import io
import torch
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

from database import create_table
from crud import insert_transaction, get_transactions
from importer import ExpenseVectorStore, BudgetMLP
# 1. 初始化与配置 FastAPI 应用
app = FastAPI()
store = ExpenseVectorStore()
le = None
model = None

# 允许跨域请求的配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # 显式允许你的前端源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法 (POST, GET, etc.)
    allow_headers=["*"],  # 允许所有 Header
)

# create_table ( only run once to create the table )
@app.on_event("startup")
def startup():
    global le, model
    create_table()
    try:
        le = joblib.load("label_encoder.joblib")
        num_classes = len(le.classes_)  
        model = BudgetMLP(input_dim=384, num_classes=num_classes)
        weights = torch.load("budget_model.pth")
        model.load_state_dict(weights)
        model.eval()  # 切换到评估模式
        print("Successfully loaded MLP model and LabelEncoder.")
    except Exception as e:
        print(f"Error loading model or LabelEncoder: {e}")
        print("Make sure you have trained the model and saved the files correctly.")

# 2. 定义 Pydantic 模型用于请求体验证
class QueryRequest(BaseModel):
    text: str

class ManualExpense(BaseModel):
    amount: float
    category: str
    date: str
    description: Optional[str] = None

# 3.核心API接口
@app.get("/")
def home():
    return {"message": "API is running"}

# query category
@app.post("/search")
async def search_expenses(request: QueryRequest):
    results = store.query(request.text, n_results=5)
    output = []
    for i in range(len(results['ids'][0])):
        output.append({
            "id": results['ids'][0][i],
            "document": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": float(results['distances'][0][i])
        })
    return {"data": output}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    raw_df = pd.read_csv(io.BytesIO(contents))
    df = raw_df.copy()

    # 1. 统一转小写进行计算
    df.columns = [c.lower() for c in df.columns]

    def merge_amounts(row):
        d_val = str(row.get('debit amount', '0')).replace(',', '').strip()
        c_val = str(row.get('credit amount', '0')).replace(',', '').strip()
        debit = pd.to_numeric(d_val, errors='coerce')
        credit = pd.to_numeric(c_val, errors='coerce')
        return float(debit) if pd.notnull(debit) and debit != 0 else float(credit if pd.notnull(credit) else 0)

    df['amount'] = df.apply(merge_amounts, axis=1)
    df['description'] = df.get('narrative', df.get('description', 'No Description'))
    df['date'] = df.get('date', 'N/A')

    # --- 核心修复：还原业务逻辑所需的字段名 ---
    # 因为你的 importer.py 内部使用的是 'Narrative' 而不是 'description'
    df['Narrative'] = df['description']
    
    # 2. 基础打标
    # 确保 manual_category 列存在
    df['manual_category'] = df['Narrative'].apply(store.rule_based_label)

    # 3. 向量传播扩展
    # ⚠️ 此时 df 内部必须有 'Narrative' 和 'manual_category'
    df = store.expand_labels_via_vector_db(df)

    # 4. MLP 模型深度预测
    if model and le:
        # 统一判断，注意这里要对齐 expand_labels 产生的列名
        # 如果 expand_labels 产生的是大写的 Manual_Category，这里也要改
        target_col = 'Manual_Category' if 'Manual_Category' in df.columns else 'manual_category'
        
        others_mask = df[target_col] == 'OTHERS'
        if others_mask.any():
            others_text = df.loc[others_mask, 'Narrative'].tolist()
            X_others = torch.tensor(store.model.encode(others_text), dtype=torch.float32)
            with torch.no_grad():
                logits = model(X_others)
                preds = torch.argmax(logits, dim=1)
                df.loc[others_mask, target_col] = le.inverse_transform(preds.numpy())
        
        # 统一回写到 manual_category 供前端使用
        df['manual_category'] = df[target_col]

    # 5. 存入向量库
    # 再次确保字段名符合 add_to_db 的硬性要求
    df['Amount'] = df['amount']
    df['Date'] = df['date']
    df['Manual_Category'] = df['manual_category']
    
    store.add_to_db(df)

    # 6. 清理并返回
    # 将所有 key 转为小写返回，这样前端 e.amount 就能拿到值
    df.columns = [c.lower() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.loc[:, ~df.columns.duplicated()] # 移除重复的列
   
    return {"status": "success", "count": len(df), "data": df.to_dict(orient="records")}
# insert expenses
@app.post("/expenses")
async def add_expense(data: dict):
    insert_transaction(
        data["amount"],
        data["category"],
        data["date"],
        data.get("description")
    )
    return {"message": "Transaction added successfully"}


# get expenses
@app.get("/expenses")
def read_all_expenses():
    # 1. 获取 SQLite 中的手动录入数据
    # 假设 SQLite 表结构为 (id, amount, category, date, description)
    manual_data = get_transactions()
    for item in manual_data:
        item["source"] = "Manual"  

    # 2. 获取 ChromaDB 中的 AI 导入数据
    # .get() 会返回 collection 中存储的所有文档和 metadata
    chroma_results = store.collection.get()
    
    ai_data = []
    if chroma_results['metadatas']:
        for i, meta in enumerate(chroma_results['metadatas']):
            # --- 关键点：字段映射 (Field Mapping) ---
            # main.py 里的 read_all_expenses 函数建议修改这段：
            ai_data.append({
                "id": chroma_results['ids'][i],
                # 统一使用小写，确保无论从哪个接口拿到的数据格式都一样
                "amount": meta.get("amount") or meta.get("Amount") or 0,
                "category": meta.get("manual_category") or meta.get("Manual_Category") or "OTHERS",
                "date": meta.get("date") or meta.get("Date") or "N/A",
                "narrative": meta.get("narrative") or meta.get("Narrative") or "",
                "source": "AI_Import"
            })
    # 3. 合并两个列表
    full_data = manual_data + ai_data

    # 4. 按日期倒序排列 (最新的消费在最上面)
    # 确保日期字符串格式一致 (如 YYYY-MM-DD)，否则排序会乱
    try:
        full_data.sort(key=lambda x: str(x['date']), reverse=True)
    except Exception as e:
        print(f"Sort failed: {e}")

    return full_data