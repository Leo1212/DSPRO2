FROM python:3.7-slim

WORKDIR /app

LABEL org.opencontainers.image.source=https://github.com/Leo1212/DSPRO2

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "hello.py", "--server.port=8501", "--server.address=0.0.0.0"]