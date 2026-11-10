FROM python:3.12-slim

RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy==2.3.5 pytest==9.2.0

WORKDIR /app
COPY tinyinfer /app/tinyinfer
COPY tests /app/tests
COPY README.md /app/README.md
