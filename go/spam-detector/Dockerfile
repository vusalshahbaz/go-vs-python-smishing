FROM golang:1.24.4-alpine3.22

# Set work directory
WORKDIR /app

# Copy app and data
COPY . .

# Install dependencies (Alpine needs build tools for numpy & pandas)
RUN go mod download


EXPOSE 8080

CMD ["go", "run", "main.go"]