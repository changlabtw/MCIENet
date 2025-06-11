# 開發階段優化：使用單階段構建加速開發流程
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# 設置工作目錄
WORKDIR /app

# 先複製 requirements.txt 以利用 Docker 緩存
COPY ../requirements.txt .

# 安裝系統依賴和 Python 套件
RUN apt-get update && apt-get install -y \
    git \
    wget \
    bedtools \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# 複製專案文件
COPY . .

# 設置環境變數
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# 默認命令：啟動交互式 shell
CMD ["/bin/bash"]

# 注意：此為開發用配置，若需部署到生產環境，請使用多階段構建來減小最終鏡像大小
