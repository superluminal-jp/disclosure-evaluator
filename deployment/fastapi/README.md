# FastAPI Deployment Guide

This guide provides instructions for deploying the FastAPI application for document disclosure evaluation.

## Prerequisites

- Docker and Docker Compose
- Python 3.12+ (for local development)
- Required API keys (OpenAI, Anthropic, or AWS)

## Quick Start

### 1. Environment Setup

Create a `.env` file in the project root:

```bash
# LLM Provider API Keys (at least one required)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# AWS (if using Bedrock)
AWS_ACCESS_KEY_ID=your_aws_access_key_id_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key_here
AWS_REGION=us-east-1

# FastAPI Configuration
DEBUG=false
ENVIRONMENT=production
SECRET_KEY=your-secret-key-here
```

### 2. Docker Deployment

```bash
# Build and start the application
cd deployment/fastapi
docker-compose up -d

# Check logs
docker-compose logs -f fastapi

# Stop the application
docker-compose down
```

### 3. Local Development

```bash
# Install dependencies
pip install -r deployment/fastapi/requirements.txt

# Run the application
cd /path/to/disclosure-evaluator
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

Once deployed, the API will be available at:

- **Health Check**: `GET /v1/health`
- **Document Evaluation**: `POST /v1/evaluation`
- **Batch Evaluation**: `POST /v1/batch`
- **Status Check**: `GET /v1/status/{request_id}`
- **API Documentation**: `GET /docs`

## Configuration

### Environment Variables

| Variable                | Description        | Default                | Required |
| ----------------------- | ------------------ | ---------------------- | -------- |
| `DEBUG`                 | Enable debug mode  | `false`                | No       |
| `ENVIRONMENT`           | Environment name   | `development`          | No       |
| `SECRET_KEY`            | Secret key for JWT | `your-secret-key-here` | Yes      |
| `OPENAI_API_KEY`        | OpenAI API key     | -                      | No\*     |
| `ANTHROPIC_API_KEY`     | Anthropic API key  | -                      | No\*     |
| `AWS_ACCESS_KEY_ID`     | AWS access key     | -                      | No\*     |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key     | -                      | No\*     |
| `AWS_REGION`            | AWS region         | `us-east-1`            | No       |

\*At least one LLM provider must be configured.

### Rate Limiting

The API includes built-in rate limiting:

- **Default**: 100 requests per minute per client
- **Configurable**: Set `RATE_LIMIT_REQUESTS` environment variable
- **Headers**: Rate limit info included in response headers

### Authentication

The API supports multiple authentication methods:

1. **API Key**: Include `X-API-Key` header
2. **JWT Token**: Include `Authorization: Bearer <token>` header

## Monitoring

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/v1/health

# Detailed health check
curl http://localhost:8000/v1/health/detailed
```

### Metrics

If Prometheus is enabled:

- **Metrics endpoint**: `http://localhost:9090`
- **Grafana dashboard**: `http://localhost:3000`

## Production Deployment

### 1. Security Considerations

- Change default secret keys
- Use HTTPS in production
- Configure proper CORS origins
- Set up proper authentication
- Enable rate limiting

### 2. Scaling

```bash
# Scale the application
docker-compose up -d --scale fastapi=3

# Use a load balancer (nginx, HAProxy, etc.)
```

### 3. Database Integration

For production, consider adding:

- **PostgreSQL**: For persistent storage
- **Redis**: For caching and session storage
- **Message Queue**: For async processing

### 4. Monitoring

- **Logging**: Structured JSON logs
- **Metrics**: Prometheus + Grafana
- **Tracing**: OpenTelemetry
- **Alerts**: Configure alerting rules

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **API Key Issues**: Verify provider configuration
3. **Rate Limiting**: Check client rate limit headers
4. **Authentication**: Verify API key or JWT token

### Debug Mode

```bash
# Enable debug mode
export DEBUG=true
export ENVIRONMENT=development

# Run with debug logging
uvicorn api.main:app --reload --log-level debug
```

### Logs

```bash
# View application logs
docker-compose logs -f fastapi

# View all service logs
docker-compose logs -f
```

## API Usage Examples

### Single Document Evaluation

```bash
curl -X POST "http://localhost:8000/v1/evaluation" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "document_content": "This document contains personal information about John Doe.",
    "context": "Request for information disclosure",
    "provider": "openai"
  }'
```

### Batch Evaluation

```bash
curl -X POST "http://localhost:8000/v1/batch" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "documents": [
      {
        "document_id": "doc_001",
        "content": "First document content"
      },
      {
        "document_id": "doc_002",
        "content": "Second document content"
      }
    ],
    "batch_options": {
      "max_concurrent": 3,
      "timeout_per_document": 300
    }
  }'
```

### Status Check

```bash
curl -X GET "http://localhost:8000/v1/status/batch_20250105_123456" \
  -H "X-API-Key: your-api-key"
```

## Support

For issues and questions:

1. Check the logs for error messages
2. Verify API key configuration
3. Test with the health check endpoint
4. Review the API documentation at `/docs`
