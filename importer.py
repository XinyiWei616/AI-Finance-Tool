import pandas as pd
import re
import chromadb
from sentence_transformers import SentenceTransformer
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
class ExpenseVectorStore:
    def __init__(self, collection_name="budget_tracker"):
        # 1. 初始化 Embedding 模型 (小巧且对短文本效果好)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. 初始化 ChromaDB (保存在本地，重启后数据还在)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def clean_dataframe(self):
        df = pd.read_csv('west_pac.csv')
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df.dropna(subset=['Date'])
        df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)

        return df

    def clean_description(self,desc):
        if not isinstance(desc, str):
            return ""
        
        # 转换为小写，方便统一识别
        desc = desc.upper()
        
        # 1. 去掉日期：例如 01/04, 05-12
        desc = re.sub(r'\d{1,2}[/\-]\d{1,2}([/\-]\d{2,4})?', '', desc)
        
        # 2. 去掉连续的流水号数字：例如 123456789
        desc = re.sub(r'\b\d{4,}\b', '', desc)
        
        # 3. 去掉特殊符号，只保留字母和空格
        desc = re.sub(r'[^A-Z\s]', ' ', desc)
        
        # 4. 去除多余空格（比如 "WOOLWORTHS   SYDNEY" -> "WOOLWORTHS SYDNEY"）
        desc = ' '.join(desc.split())

        return desc
    
    def add_to_db(self, df):
        """将清洗后的数据存入向量数据库"""
        # 先处理 metadata，日期格式化为字符串，方便后续展示
        df_meta = df.copy()
        for col in df_meta.columns:
            if pd.api.types.is_datetime64_any_dtype(df_meta[col]):
                df_meta[col] = df_meta[col].dt.strftime('%Y-%m-%d')
        # 只取描述列进行向量化
        descriptions = df_meta['Narrative'].apply(self.clean_description).tolist()
        # 原始数据作为 metadata 存入，方便检索后展示详情
        metadatas = df_meta.to_dict('records')
        ids = [f"id_{i}" for i in range(len(df_meta))]

        # 核心：将文本变为向量并存储
        self.collection.add(
            documents=descriptions,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Success: {len(descriptions)} records added to the vector store.")

    def query(self, text, n_results=3):
        """语义搜索"""
        results = self.collection.query(
            query_texts=[text],
            n_results=n_results
        )
        return results
    
    def rule_based_label(self,desc):
        desc = desc.upper()
        if any(word in desc for word in ['WOOLWORTHS', 'COLES', 'ALDI', 'GROCER']):
            return 'GROCERIES'
        if any(word in desc for word in ['7-ELEVEN', 'BP', 'SHELL', 'TRANSPORT', 'OPAL']):
            return 'TRANSPORT'
        if any(word in desc for word in ['RESTAURANT', 'CAFE', 'KFC', 'MCDONALD', 'HUNGRY']):
            return 'DINING'
        if any(word in desc for word in ['NETFLIX', 'SPOTIFY', 'STEAM', 'NINTENDO']):
            return 'ENTERTAINMENT'
        if any(word in desc for word in ['RNT', 'RENT', 'MORTGAGE']):
            return 'HOUSING'
        return 'OTHERS'

    def expand_labels_via_vector_db(self, labeled_df):
        if 'Manual_Category' not in labeled_df.columns:
             labeled_df['Manual_Category'] = labeled_df['Narrative'].apply(self.rule_based_label)

        # 策略改进：定义每个类别的“标准自然语言种子”
        # 这样即使账单里只有 'RNT'，我们用 'Rent' 去搜也能把附近的 RNT 抓出来
        natural_language_seeds = {
            'HOUSING': ['Monthly rent payment', 'Room rent', 'Apartment rental'],
            'GROCERIES': ['Supermarket shopping', 'Weekly groceries', 'Food buy'],
            'TRANSPORT': ['Public transport fare', 'Petrol station gas', 'Opal top up'],
            'DINING': ['Restaurant bill', 'Takeaway food', 'Lunch at cafe']
        }

        for category, query_seeds in natural_language_seeds.items():
            print(f"正在通过自然语言增强类别 [{category}]...")
            for q_seed in query_seeds:
                # 使用自然语言查询去向量库里找那些简写的账单
                results = self.query(q_seed, n_results=15)
                found_ids = results['ids'][0]
                distances = results['distances'][0]

                for id_val, dist in zip(found_ids, distances):
                    if dist < 0.5: # 稍微放宽一点，允许跨语义匹配
                        idx = int(id_val.split('_')[1])
                        # 只有当原本是 OTHERS 时才覆盖，保护手动打标的准确性
                        if labeled_df.at[idx, 'Manual_Category'] == 'OTHERS':
                            labeled_df.at[idx, 'Manual_Category'] = category
        return labeled_df

class BudgetMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BudgetMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3), # 增加鲁棒性
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)
    
def train_model_pipeline(X_train, y_train, store, le, df, natural_language_seeds):
    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)
    model = BudgetMLP(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    
    # --- PHASE 1: 基础规则训练 ---
    print("\n--- Phase 1: Training on Rule-based Labels ---")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(80):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    # --- PHASE 2: 语义对齐 (使用 Seeds 强化) ---
    print("\n--- Phase 2: Strengthening Semantic Alignment ---")
    X_bridge_list = []
    y_bridge_list = []
    
    for cat, queries in natural_language_seeds.items():
        if cat in le.classes_:
            vecs = store.model.encode(queries)
            X_bridge_list.append(torch.tensor(vecs, dtype=torch.float32))
            label_idx = le.transform([cat])[0]
            y_bridge_list.append(torch.tensor([label_idx] * len(queries), dtype=torch.long))
    
    X_bridge = torch.cat(X_bridge_list)
    y_bridge = torch.cat(y_bridge_list)
    X_combined = torch.cat([X_train, X_bridge])
    y_combined = torch.cat([y_train, y_bridge])
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_combined), y_combined)
        loss.backward()
        optimizer.step()

    # --- PHASE 3: 自训练提纯 (逻辑移出循环) ---
    print("\n--- Phase 3: Self-Training on Unlabeled Data ---")
    model.eval()
    others_df = df[df['Manual_Category'] == 'OTHERS'].copy()
    if not others_df.empty:
        X_others = torch.tensor(store.model.encode(others_df['Narrative'].tolist()), dtype=torch.float32)
        with torch.no_grad():
            outputs = model(X_others)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)
        
        high_conf_mask = confidences > 0.95
        if high_conf_mask.any():
            X_pseudo = X_others[high_conf_mask]
            y_pseudo = preds[high_conf_mask]
            
            # 修正变量名：X_combined
            X_final = torch.cat([X_combined, X_pseudo])
            y_final = torch.cat([y_combined, y_pseudo])
            
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            for epoch in range(30):
                model.train()
                optimizer.zero_grad()
                loss = criterion(model(X_final), y_final)
                loss.backward()
                optimizer.step()
            print(f"Phase 3 Complete. Added {len(y_pseudo)} pseudo-labels.")

    return model
       
if __name__ == "__main__":
    if __name__ == "__main__":
    # 1. 定义种子数据 
        natural_language_seeds = {
            'HOUSING': ['Monthly rent payment', 'Room rent', 'Apartment rental'],
            'GROCERIES': ['Supermarket shopping', 'Weekly groceries', 'Food buy'],
            'TRANSPORT': ['Public transport fare', 'Petrol station gas', 'Opal top up'],
            'DINING': ['Restaurant bill', 'Takeaway food', 'Lunch at cafe']
        }

        # 2. 初始化存储与数据
        store = ExpenseVectorStore()
        df = store.clean_dataframe()
        
        # 3. 初始规则打标
        df['Manual_Category'] = df['Narrative'].apply(store.rule_based_label)
        
        # 4. 存入数据库并扩展标签 (必须按这个顺序)
        store.add_to_db(df)
        df = store.expand_labels_via_vector_db(df)
        
        # 5. 准备第一轮训练所需的数据特征 (X 和 y)
        train_df = df[df['Manual_Category'] != 'OTHERS'].copy()
        le = LabelEncoder()
        y_train = torch.tensor(le.fit_transform(train_df['Manual_Category']), dtype=torch.long)
        X_train = torch.tensor(store.model.encode(train_df['Narrative'].tolist()), dtype=torch.float32)

        # 6. 调用函数开始三阶段训练
        final_model = train_model_pipeline(X_train, y_train, store, le, df, natural_language_seeds)

        # 7. 保存结果
        torch.save(final_model.state_dict(), "budget_model.pth")
        import joblib
        joblib.dump(le, "label_encoder.joblib")
        print("✨ Model training complete and saved.")
