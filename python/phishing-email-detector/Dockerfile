FROM python:3.11-alpine

# Set work directory
WORKDIR /app

# Install dependencies (Alpine needs build tools for numpy & pandas)
RUN apk add --no-cache build-base openblas-dev && \
    pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app and data
COPY . .

EXPOSE 8080

CMD ["python", "app.py"]