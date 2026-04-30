# 使用轻量级的 Python 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装必要的系统依赖（ChromaDB 某些依赖需要编译环境）
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
# 建议先复制 requirements.txt 以利用 Docker 缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目所有文件
COPY . .

# 暴露 FastAPI 运行端口
EXPOSE 8000

# 定义启动逻辑：
# 我们先运行训练脚本生成模型（AOT 模式），然后再启动 FastAPI 服务
CMD ["sh", "-c", "python importer.py && python eval.py && uvicorn main:app --host 0.0.0.0 --port 8000"]