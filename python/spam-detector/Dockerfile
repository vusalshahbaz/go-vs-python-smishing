FROM python:3.11-alpine3.22

# Set work directory
WORKDIR /app

# Install dependencies (Alpine needs build tools for numpy & pandas)
RUN apk add --no-cache build-base openblas-dev && \
    pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Copy dataset (dataset file should be in project directory before building)
# Build with: cp ../../datasets/spam-sms.csv . && docker build --no-cache -t <name> .
RUN mkdir -p /app/datasets
COPY spam-sms.csv /app/datasets/spam-sms.csv

EXPOSE 8080

CMD ["python", "app.py", "/app/datasets/spam-sms.csv"]