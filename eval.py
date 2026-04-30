import torch
import json
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 假设你的 BudgetMLP 类定义在这里或从 train.py 导入
from importer import BudgetMLP, ExpenseVectorStore

def run_evaluation():
    # 1. 环境准备
    print("🧪 正在初始化评测环境...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载标签编码器和模型
    try:
        le = joblib.load("label_encoder.joblib")
        num_classes = len(le.classes_)
        
        # 初始化并加载训练好的模型权重 (假设 input_dim 为 384)
        model = BudgetMLP(input_dim=384, num_classes=num_classes)
        model.load_state_dict(torch.load("budget_model.pth", map_location=device))
        model.to(device)
        model.eval()
        
        # 初始化向量转换器 (用于将文本转为向量)
        store = ExpenseVectorStore()
    except FileNotFoundError:
        print("❌ 错误：未找到模型文件。请先运行训练脚本生成 .pth 和 .joblib 文件。")
        return

    # --- 2. 语义测试 (读取 JSON) ---
    print("\n--- 🟢 阶段 A: JSON 语义压力测试 ---")
    try:
        with open('test_cases.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            test_cases = data.get('semantic_tests', [])
    except FileNotFoundError:
        print("❌ 未找到 testcases.json 文件")
        test_cases = []

    hits = 0
    for case in test_cases:
        query = case['query']
        expected = case['expected']
        
        # 向量化并预测
        vec = torch.tensor(store.model.encode([query]), dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model(vec)
            prob = torch.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)
            actual = le.inverse_transform(pred.cpu().numpy())[0]
        
        is_correct = (actual == expected)
        if is_correct: hits += 1
        
        print(f"{'✅' if is_correct else '❌'} Query: '{query}'")
        print(f"   Predict: {actual} ({conf.item():.2%}) | Expected: {expected}")

    # --- 3. 历史数据回测 ---
    print("\n--- 🔵 阶段 B: 历史账单分类报告 ---")
    df = store.clean_dataframe()
    df['True_Label'] = df['Narrative'].apply(store.rule_based_label)
    eval_df = df[df['True_Label'] != 'OTHERS'].copy()

    if not eval_df.empty:
        texts = eval_df['Narrative'].tolist()
        y_true = le.transform(eval_df['True_Label'])
        
        X_eval = torch.tensor(store.model.encode(texts), dtype=torch.float32).to(device)
        with torch.no_grad():
            _, y_pred = torch.max(model(X_eval), 1)
            y_pred = y_pred.cpu().numpy()

        print(classification_report(y_true, y_pred, target_names=le.classes_))
    
    final_score = (hits / len(test_cases)) if test_cases else 0
    print(f"\n✨ 评测完成。语义测试通过率: {final_score:.2%}")

if __name__ == "__main__":
    run_evaluation()