FROM golang:1.25.4-alpine3.22

# Set work directory
WORKDIR /app

# Copy go.mod and go.sum first for better caching
COPY go.mod go.sum ./
RUN go mod download

# Copy application code
COPY . .

# Copy dataset (dataset file should be in project directory before building)
# Build with: cp ../../datasets/spam-sms.csv . && docker build --no-cache -t <name> .
RUN mkdir -p /app/datasets
COPY spam-sms.csv /app/datasets/spam-sms.csv

EXPOSE 8080

CMD ["go", "run", "main.go", "/app/datasets/spam-sms.csv"]