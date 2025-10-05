#!/bin/bash
# Lambda deployment script for disclosure evaluator

set -e

# Configuration
STACK_NAME="disclosure-evaluator"
REGION="us-east-1"
S3_BUCKET=""
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
USE_S3_FOR_STATUS="false"
ENABLE_API_KEY_AUTH="false"
ALLOWED_API_KEYS=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --stack-name NAME          CloudFormation stack name (default: disclosure-evaluator)"
    echo "  --region REGION            AWS region (default: us-east-1)"
    echo "  --s3-bucket BUCKET         S3 bucket for batch state storage"
    echo "  --openai-key KEY           OpenAI API key"
    echo "  --anthropic-key KEY        Anthropic API key"
    echo "  --use-s3-status            Use S3 for status storage"
    echo "  --enable-api-key-auth      Enable API key authentication"
    echo "  --allowed-api-keys KEYS    Comma-separated list of allowed API keys"
    echo "  --help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --openai-key sk-xxx --s3-bucket my-bucket"
    echo "  $0 --anthropic-key sk-ant-xxx --use-s3-status --enable-api-key-auth"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --stack-name)
            STACK_NAME="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --s3-bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        --openai-key)
            OPENAI_API_KEY="$2"
            shift 2
            ;;
        --anthropic-key)
            ANTHROPIC_API_KEY="$2"
            shift 2
            ;;
        --use-s3-status)
            USE_S3_FOR_STATUS="true"
            shift
            ;;
        --enable-api-key-auth)
            ENABLE_API_KEY_AUTH="true"
            shift
            ;;
        --allowed-api-keys)
            ALLOWED_API_KEYS="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$OPENAI_API_KEY" && -z "$ANTHROPIC_API_KEY" ]]; then
    print_error "At least one API key (OpenAI or Anthropic) is required"
    exit 1
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if SAM CLI is installed
if ! command -v sam &> /dev/null; then
    print_error "AWS SAM CLI is not installed. Please install it first."
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

print_status "Starting deployment of disclosure evaluator Lambda function..."

# Create deployment package
print_status "Creating deployment package..."
if [[ -d "package" ]]; then
    rm -rf package
fi

mkdir -p package
cp -r ../../api package/
cp -r ../../evaluator.py package/
cp -r ../../criteria package/
cp -r ../../prompts package/
cp -r ../../config.json package/
cp requirements.txt package/

# Install dependencies
print_status "Installing dependencies..."
cd package
pip install -r requirements.txt -t .
cd ..

# Create SAM deployment parameters
PARAMETERS=""
if [[ -n "$OPENAI_API_KEY" ]]; then
    PARAMETERS="$PARAMETERS OpenAIApiKey=$OPENAI_API_KEY"
fi
if [[ -n "$ANTHROPIC_API_KEY" ]]; then
    PARAMETERS="$PARAMETERS AnthropicApiKey=$ANTHROPIC_API_KEY"
fi
if [[ -n "$S3_BUCKET" ]]; then
    PARAMETERS="$PARAMETERS S3BucketName=$S3_BUCKET"
fi
PARAMETERS="$PARAMETERS UseS3ForStatus=$USE_S3_FOR_STATUS"
PARAMETERS="$PARAMETERS EnableApiKeyAuth=$ENABLE_API_KEY_AUTH"
if [[ -n "$ALLOWED_API_KEYS" ]]; then
    PARAMETERS="$PARAMETERS AllowedApiKeys=$ALLOWED_API_KEYS"
fi

# Deploy with SAM
print_status "Deploying with AWS SAM..."
if sam deploy \
    --template-file template.yaml \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --parameter-overrides $PARAMETERS \
    --capabilities CAPABILITY_IAM \
    --no-confirm-changeset; then
    
    print_status "Deployment completed successfully!"
    
    # Get stack outputs
    print_status "Retrieving stack outputs..."
    FUNCTION_NAME=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`FunctionName`].OutputValue' \
        --output text)
    
    API_URL=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
        --output text)
    
    S3_BUCKET_NAME=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`S3BucketName`].OutputValue' \
        --output text)
    
    echo ""
    print_status "Deployment Summary:"
    echo "  Stack Name: $STACK_NAME"
    echo "  Region: $REGION"
    echo "  Function Name: $FUNCTION_NAME"
    echo "  API URL: $API_URL"
    echo "  S3 Bucket: $S3_BUCKET_NAME"
    echo ""
    
    print_status "Testing the function..."
    if aws lambda invoke \
        --function-name "$FUNCTION_NAME" \
        --region "$REGION" \
        --payload '{"operation":"health"}' \
        response.json; then
        
        print_status "Function test successful!"
        echo "Response:"
        cat response.json
        echo ""
        rm -f response.json
    else
        print_warning "Function test failed. Check CloudWatch logs for details."
    fi
    
    print_status "Deployment completed! You can now use the API at: $API_URL"
    
else
    print_error "Deployment failed!"
    exit 1
fi

# Cleanup
rm -rf package
