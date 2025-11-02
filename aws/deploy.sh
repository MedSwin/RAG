#!/bin/bash

# AWS EC2 Deployment Script for RAG System to Existing Instance
# This script deploys the RAG system to an existing AWS EC2 instance

set -e

# Configuration
EC2_HOST="ec2-98-93-155-138.compute-1.amazonaws.com"
EC2_USER="${EC2_USER:-ubuntu}"  # Allow override via environment variable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SSH_KEY="${PROJECT_ROOT}/key/RAGServerKey.pem"
REGION="us-east-1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting RAG System Deployment to Existing EC2 Instance${NC}"
echo -e "${YELLOW}Target: ${EC2_HOST}${NC}"

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}SSH key not found: ${SSH_KEY}${NC}"
    exit 1
fi

# Set correct permissions on SSH key
chmod 400 "$SSH_KEY"

# Check SSH connectivity
echo -e "${YELLOW}Checking SSH connectivity...${NC}"
SSH_SUCCESS=false
# Try common EC2 users
for USER in ubuntu ec2-user admin; do
    if ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "${USER}@${EC2_HOST}" "echo 'SSH connection successful'" &> /dev/null; then
        EC2_USER="$USER"
        echo -e "${GREEN}SSH connection successful as ${EC2_USER}${NC}"
        SSH_SUCCESS=true
        break
    fi
done

if [ "$SSH_SUCCESS" = false ]; then
    echo -e "${RED}Failed to connect to EC2 instance. Please check:${NC}"
    echo -e "${RED}  - Instance is running${NC}"
    echo -e "${RED}  - Security group allows SSH (port 22)${NC}"
    echo -e "${RED}  - SSH key is correct and matches the instance${NC}"
    echo -e "${YELLOW}You can manually specify the SSH user by setting EC2_USER environment variable${NC}"
    echo -e "${YELLOW}Example: EC2_USER=ubuntu ./aws/deploy.sh${NC}"
    exit 1
fi

# Prepare deployment package (exclude unnecessary files)
echo -e "${YELLOW}Preparing deployment package...${NC}"
TEMP_DIR=$(mktemp -d)
DEPLOY_DIR="${TEMP_DIR}/rag-system"

# Create deployment directory structure
mkdir -p "${DEPLOY_DIR}"
mkdir -p "${DEPLOY_DIR}/app"
mkdir -p "${DEPLOY_DIR}/lab"
mkdir -p "${DEPLOY_DIR}/aws"
mkdir -p "${DEPLOY_DIR}/nginx"

# Copy necessary files
echo -e "${YELLOW}Copying files...${NC}"
cp -r app/ "${DEPLOY_DIR}/"
cp -r lab/ "${DEPLOY_DIR}/"
cp -r nginx/ "${DEPLOY_DIR}/"
cp docker-compose.yml "${DEPLOY_DIR}/"
cp Dockerfile "${DEPLOY_DIR}/"
cp requirements.txt "${DEPLOY_DIR}/"
cp env.example "${DEPLOY_DIR}/"
cp .gitignore "${DEPLOY_DIR}/"

# Remove unnecessary files from deployment
find "${DEPLOY_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${DEPLOY_DIR}" -type f -name "*.pyc" -delete 2>/dev/null || true
find "${DEPLOY_DIR}" -type d -name ".git" -exec rm -rf {} + 2>/dev/null || true

echo -e "${GREEN}Files prepared${NC}"

# Copy files to EC2 instance
echo -e "${YELLOW}Copying files to EC2 instance...${NC}"
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -r "${DEPLOY_DIR}/" "${EC2_USER}@${EC2_HOST}:~/rag-system/"

echo -e "${GREEN}Files copied successfully${NC}"

# Deploy on EC2 instance
echo -e "${YELLOW}Deploying on EC2 instance...${NC}"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${EC2_USER}@${EC2_HOST}" << 'ENDSSH'
    set -e
    
    cd ~/rag-system
    
    # Install Docker if not installed
    if ! command -v docker &> /dev/null; then
        echo "Installing Docker..."
        sudo apt-get update
        sudo apt-get install -y ca-certificates curl gnupg lsb-release
        sudo mkdir -p /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        echo \
          "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
          $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        sudo usermod -aG docker $USER || sudo usermod -aG docker ubuntu || true
        sudo systemctl start docker
        sudo systemctl enable docker
    fi
    
    # Install docker-compose if not installed
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo "Installing docker-compose..."
        sudo apt-get update
        sudo apt-get install -y docker-compose
    fi
    
    # Stop existing containers if running
    echo "Stopping existing containers..."
    sudo docker-compose down 2>/dev/null || docker compose down 2>/dev/null || true
    
    # Create necessary directories
    mkdir -p models data logs storage nginx/ssl
    
    # Set permissions
    sudo chown -R $USER:$USER models data logs storage nginx 2>/dev/null || sudo chown -R ubuntu:ubuntu models data logs storage nginx 2>/dev/null || true
    
    # Build Docker image on EC2
    echo "Building Docker image on EC2..."
    sudo docker-compose build || docker compose build
    
    # Start services with docker-compose
    echo "Starting services..."
    sudo docker-compose up -d || docker compose up -d
    
    echo "Waiting for services to start..."
    sleep 30
    
    # Check service status
    echo "Checking service status..."
    sudo docker-compose ps || docker compose ps
    
ENDSSH

echo -e "${GREEN}Deployment completed${NC}"

# Cleanup local temp directory
rm -rf "${TEMP_DIR}"

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 30

# Test the API and Frontend
echo -e "${YELLOW}Testing deployment...${NC}"

# Test health endpoint
if curl -f -s "http://${EC2_HOST}/health" > /dev/null; then
    echo -e "${GREEN}✓ Health endpoint is accessible${NC}"
else
    echo -e "${YELLOW}⚠ Health endpoint check failed (may need more time)${NC}"
fi

# Test API root endpoint
if curl -f -s "http://${EC2_HOST}/" > /dev/null; then
    echo -e "${GREEN}✓ API root endpoint is accessible${NC}"
else
    echo -e "${YELLOW}⚠ API root endpoint check failed${NC}"
fi

# Test frontend dashboard
if curl -f -s "http://${EC2_HOST}/api/v1/dashboard/" > /dev/null; then
    echo -e "${GREEN}✓ Frontend dashboard is accessible${NC}"
    echo -e "${GREEN}Frontend URL: http://${EC2_HOST}/api/v1/dashboard/${NC}"
else
    echo -e "${YELLOW}⚠ Frontend dashboard check failed (may need more time)${NC}"
fi

# Display deployment summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Instance: ${EC2_HOST}"
echo -e "API: http://${EC2_HOST}"
echo -e "Health: http://${EC2_HOST}/health"
echo -e "Frontend: http://${EC2_HOST}/api/v1/dashboard/"
echo -e "API Docs: http://${EC2_HOST}/docs"
echo -e ""
echo -e "${YELLOW}SSH Command:${NC}"
echo -e "ssh -i ${SSH_KEY} ${EC2_USER}@${EC2_HOST}"
echo -e ""
echo -e "${YELLOW}View logs:${NC}"
echo -e "ssh -i ${SSH_KEY} ${EC2_USER}@${EC2_HOST} 'cd ~/rag-system && sudo docker-compose logs -f'"
echo -e ""
