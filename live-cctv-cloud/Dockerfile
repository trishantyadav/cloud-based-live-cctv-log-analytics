FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV AWS_REGION=ap-south-1
ENV WORKER_DATA_DIR=/app/data
ENV PYTHONPATH=/app

WORKDIR /app

COPY live-cctv-cloud/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install RAG modules as a package
COPY pyproject.toml ./pyproject.toml
COPY *.py ./cctv_rag/
COPY *.py ./
RUN pip install --no-cache-dir --no-deps -e .

# Copy worker application
COPY live-cctv-cloud/worker.py ./worker.py

RUN mkdir -p /app/data /app/data/quarantine

CMD ["python", "-u", "worker.py"]
