# Disclosure Evaluator Lambda Function

This directory contains the AWS Lambda implementation of the disclosure evaluator system, providing serverless document evaluation capabilities.

## Overview

The Lambda function provides:

- Single document evaluation
- Batch document processing
- Status tracking and monitoring
- Health checks
- API Gateway integration

## Architecture

```
API Gateway → Lambda Function → Evaluator Core
                ↓
            S3 (Batch State)
                ↓
            CloudWatch (Logs/Metrics)
```

## Features

### Single Document Evaluation

- Real-time document evaluation
- Multiple LLM provider support (OpenAI, Anthropic, AWS Bedrock)
- Structured response format
- Error handling and validation

### Batch Processing

- Asynchronous batch evaluation
- Progress tracking
- S3-based state persistence
- Retry mechanisms

### Status Monitoring

- Request status tracking
- Progress reporting
- Error details
- Completion notifications

## Configuration

### Environment Variables

| Variable                     | Description                        | Default                |
| ---------------------------- | ---------------------------------- | ---------------------- |
| `FUNCTION_NAME`              | Lambda function name               | `disclosure-evaluator` |
| `FUNCTION_VERSION`           | Function version                   | `1.0.0`                |
| `MEMORY_LIMIT_MB`            | Memory limit (MB)                  | `2048`                 |
| `TIMEOUT_SECONDS`            | Timeout (seconds)                  | `900`                  |
| `MAX_CONCURRENT_WORKERS`     | Max parallel workers               | `5`                    |
| `MAX_RETRY_ATTEMPTS`         | Max retry attempts                 | `3`                    |
| `FILE_SIZE_LIMIT`            | Max file size (bytes)              | `52428800` (50MB)      |
| `API_RATE_LIMIT_DELAY`       | API rate limit delay               | `0.1`                  |
| `RETRY_DELAY_SECONDS`        | Retry delay (seconds)              | `30`                   |
| `EXPONENTIAL_BACKOFF`        | Use exponential backoff            | `true`                 |
| `AWS_REGION`                 | AWS region                         | `us-east-1`            |
| `S3_BUCKET_NAME`             | S3 bucket for batch state          | -                      |
| `USE_S3_FOR_STATUS`          | Use S3 for status storage          | `false`                |
| `STATUS_STORAGE_PATH`        | Local status storage path          | `/tmp/status`          |
| `DEFAULT_PROVIDER`           | Default LLM provider               | `openai`               |
| `OPENAI_API_KEY`             | OpenAI API key                     | -                      |
| `ANTHROPIC_API_KEY`          | Anthropic API key                  | -                      |
| `AWS_ACCESS_KEY_ID`          | AWS access key ID                  | -                      |
| `AWS_SECRET_ACCESS_KEY`      | AWS secret access key              | -                      |
| `LOG_LEVEL`                  | Logging level                      | `INFO`                 |
| `LOG_FORMAT`                 | Log format                         | `json`                 |
| `ENABLE_STRUCTURED_LOGGING`  | Enable structured logging          | `true`                 |
| `ENABLE_PARALLEL_PROCESSING` | Enable parallel processing         | `true`                 |
| `MAX_BATCH_SIZE`             | Max batch size                     | `100`                  |
| `BATCH_TIMEOUT_SECONDS`      | Batch timeout (seconds)            | `1800`                 |
| `ENABLE_API_KEY_AUTH`        | Enable API key authentication      | `false`                |
| `API_KEY_HEADER`             | API key header name                | `X-API-Key`            |
| `ALLOWED_API_KEYS`           | Allowed API keys (comma-separated) | -                      |
| `ENABLE_METRICS`             | Enable CloudWatch metrics          | `true`                 |
| `METRICS_NAMESPACE`          | CloudWatch metrics namespace       | `DisclosureEvaluator`  |

## Deployment

### Prerequisites

1. AWS CLI configured
2. AWS SAM CLI installed
3. Python 3.12+
4. Required API keys (OpenAI, Anthropic, or AWS credentials)

### Quick Deployment

```bash
# Deploy with OpenAI
./deploy.sh --openai-key sk-xxx --s3-bucket my-bucket

# Deploy with Anthropic
./deploy.sh --anthropic-key sk-ant-xxx --use-s3-status

# Deploy with API key authentication
./deploy.sh --openai-key sk-xxx --enable-api-key-auth --allowed-api-keys key1,key2
```

### Manual Deployment

1. **Create deployment package:**

   ```bash
   mkdir -p package
   cp -r ../../api package/
   cp -r ../../evaluator.py package/
   cp -r ../../criteria package/
   cp -r ../../prompts package/
   cp -r ../../config.json package/
   cp requirements.txt package/
   cd package
   pip install -r requirements.txt -t .
   cd ..
   ```

2. **Deploy with SAM:**
   ```bash
   sam deploy \
     --template-file template.yaml \
     --stack-name disclosure-evaluator \
     --region us-east-1 \
     --parameter-overrides \
       OpenAIApiKey=sk-xxx \
       S3BucketName=my-bucket \
     --capabilities CAPABILITY_IAM
   ```

## Usage

### API Gateway Endpoints

| Endpoint                    | Method | Description                 |
| --------------------------- | ------ | --------------------------- |
| `/health`                   | GET    | Health check                |
| `/evaluation`               | POST   | Single document evaluation  |
| `/batch`                    | POST   | Batch evaluation submission |
| `/batch/{batch_id}`         | GET    | Batch status                |
| `/batch/{batch_id}/results` | GET    | Batch results               |
| `/status/{request_id}`      | GET    | Request status              |

### Direct Lambda Invocation

```python
import json
import boto3

# Initialize Lambda client
lambda_client = boto3.client('lambda')

# Single document evaluation
event = {
    "operation": "evaluation",
    "document_content": "Document content here...",
    "context": "Additional context",
    "provider": "openai"
}

response = lambda_client.invoke(
    FunctionName='disclosure-evaluator',
    InvocationType='RequestResponse',
    Payload=json.dumps(event)
)

result = json.loads(response['Payload'].read())
print(result)
```

### Batch Processing

```python
# Submit batch
batch_event = {
    "operation": "batch",
    "documents": [
        {
            "document_id": "doc_001",
            "content": "First document content",
            "context": "Context for doc 1"
        },
        {
            "document_id": "doc_002",
            "content": "Second document content",
            "context": "Context for doc 2"
        }
    ],
    "batch_options": {
        "max_concurrent": 5,
        "timeout_per_document": 300
    }
}

response = lambda_client.invoke(
    FunctionName='disclosure-evaluator',
    InvocationType='RequestResponse',
    Payload=json.dumps(batch_event)
)

result = json.loads(response['Payload'].read())
batch_id = json.loads(result['body'])['batch_id']

# Check status
status_event = {
    "operation": "status",
    "batch_id": batch_id
}

status_response = lambda_client.invoke(
    FunctionName='disclosure-evaluator',
    InvocationType='RequestResponse',
    Payload=json.dumps(status_event)
)

status_result = json.loads(status_response['Payload'].read())
print(status_result)
```

## Testing

### Local Testing

```bash
# Run test script
python test_lambda.py

# Test specific operations
python -c "
from api.lambda_handler import lambda_handler
from unittest.mock import Mock

event = {'operation': 'health'}
context = Mock()
context.aws_request_id = 'test-123'

result = lambda_handler(event, context)
print(result)
"
```

### Unit Tests

```bash
# Run unit tests
cd ../../tests
python -m pytest api/test_lambda_handler.py -v
```

## Monitoring

### CloudWatch Logs

- Log Group: `/aws/lambda/disclosure-evaluator`
- Retention: 30 days
- Structured JSON logging

### CloudWatch Metrics

- Invocations
- Errors
- Duration
- Throttles
- Concurrent Executions

### Custom Metrics

- Evaluation count
- Batch processing time
- Error rates by type

## Security

### Authentication

- API key authentication (optional)
- IAM roles for AWS services
- VPC configuration (optional)

### Data Protection

- Encryption in transit (HTTPS)
- Encryption at rest (S3)
- No persistent storage of document content
- Secure API key management

## Performance

### Limits

- Memory: 10,240 MB (10 GB)
- Timeout: 15 minutes
- Payload size: 6 MB (synchronous), 256 KB (asynchronous)
- Concurrent executions: 1,000 (default)

### Optimization

- Parallel processing for batch operations
- S3 for large batch state storage
- CloudWatch metrics for monitoring
- Exponential backoff for retries

## Troubleshooting

### Common Issues

1. **Timeout errors**: Increase timeout or optimize processing
2. **Memory errors**: Increase memory allocation
3. **API rate limits**: Adjust rate limiting settings
4. **S3 permissions**: Ensure Lambda has S3 access

### Debugging

1. Check CloudWatch logs
2. Monitor CloudWatch metrics
3. Test locally with `test_lambda.py`
4. Verify environment variables
5. Check IAM permissions

### Support

- CloudWatch Logs for detailed error information
- CloudWatch Metrics for performance monitoring
- AWS X-Ray for distributed tracing (optional)

## Cost Optimization

### Recommendations

1. Use appropriate memory allocation
2. Optimize timeout settings
3. Implement efficient batch processing
4. Monitor CloudWatch costs
5. Use S3 lifecycle policies for batch state

### Cost Factors

- Lambda invocations
- Memory allocation
- Duration
- S3 storage and requests
- CloudWatch logs and metrics

## Updates and Maintenance

### Updating the Function

1. Update code in the `api/` directory
2. Run tests to ensure compatibility
3. Update deployment package
4. Deploy with SAM
5. Test in staging environment
6. Deploy to production

### Monitoring Updates

- Monitor error rates after updates
- Check performance metrics
- Verify all endpoints work correctly
- Update documentation as needed
