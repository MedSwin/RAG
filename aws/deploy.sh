#!/bin/bash

# AWS EC2 Deployment Script for RAG System
# This script deploys the RAG system to AWS EC2

set -e

# Configuration
REGION="us-east-1"
INSTANCE_TYPE="g4dn.xlarge"
KEY_NAME="rag-system-key"
SECURITY_GROUP="rag-system-sg"
AMI_ID="ami-0c02fb55956c7d316"  # Deep Learning AMI (Ubuntu 20.04) Version 60.0
VOLUME_SIZE=100  # GB

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting RAG System Deployment to AWS EC2${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install it first.${NC}"
    exit 1
fi

# Create security group if it doesn't exist
echo -e "${YELLOW}Creating security group...${NC}"
if ! aws ec2 describe-security-groups --group-names $SECURITY_GROUP --region $REGION &> /dev/null; then
    aws ec2 create-security-group \
        --group-name $SECURITY_GROUP \
        --description "Security group for RAG system" \
        --region $REGION
    
    # Add inbound rules
    aws ec2 authorize-security-group-ingress \
        --group-name $SECURITY_GROUP \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region $REGION
    
    aws ec2 authorize-security-group-ingress \
        --group-name $SECURITY_GROUP \
        --protocol tcp \
        --port 80 \
        --cidr 0.0.0.0/0 \
        --region $REGION
    
    aws ec2 authorize-security-group-ingress \
        --group-name $SECURITY_GROUP \
        --protocol tcp \
        --port 443 \
        --cidr 0.0.0.0/0 \
        --region $REGION
    
    aws ec2 authorize-security-group-ingress \
        --group-name $SECURITY_GROUP \
        --protocol tcp \
        --port 8000 \
        --cidr 0.0.0.0.0/0 \
        --region $REGION
else
    echo -e "${GREEN}Security group already exists${NC}"
fi

# Create key pair if it doesn't exist
echo -e "${YELLOW}Creating key pair...${NC}"
if ! aws ec2 describe-key-pairs --key-names $KEY_NAME --region $REGION &> /dev/null; then
    aws ec2 create-key-pair \
        --key-name $KEY_NAME \
        --region $REGION \
        --query 'KeyMaterial' \
        --output text > ${KEY_NAME}.pem
    
    chmod 400 ${KEY_NAME}.pem
    echo -e "${GREEN}Key pair created: ${KEY_NAME}.pem${NC}"
else
    echo -e "${GREEN}Key pair already exists${NC}"
fi

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t rag-system:latest .

# Save Docker image
echo -e "${YELLOW}Saving Docker image...${NC}"
docker save rag-system:latest | gzip > rag-system.tar.gz

# Launch EC2 instance
echo -e "${YELLOW}Launching EC2 instance...${NC}"
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-groups $SECURITY_GROUP \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\"}}]" \
    --region $REGION \
    --query 'Instances[0].InstanceId' \
    --output text)

echo -e "${GREEN}Instance launched: $INSTANCE_ID${NC}"

# Wait for instance to be running
echo -e "${YELLOW}Waiting for instance to be running...${NC}"
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get instance public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo -e "${GREEN}Instance public IP: $PUBLIC_IP${NC}"

# Wait for SSH to be available
echo -e "${YELLOW}Waiting for SSH to be available...${NC}"
until ssh -i ${KEY_NAME}.pem -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP "echo 'SSH is ready'" &> /dev/null; do
    echo "Waiting for SSH..."
    sleep 10
done

# Copy files to instance
echo -e "${YELLOW}Copying files to instance...${NC}"
scp -i ${KEY_NAME}.pem -r . ubuntu@$PUBLIC_IP:~/rag-system/
scp -i ${KEY_NAME}.pem rag-system.tar.gz ubuntu@$PUBLIC_IP:~/rag-system/

# Install Docker on instance
echo -e "${YELLOW}Installing Docker on instance...${NC}"
ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP << 'EOF'
    sudo apt-get update
    sudo apt-get install -y docker.io docker-compose
    sudo usermod -aG docker ubuntu
    sudo systemctl start docker
    sudo systemctl enable docker
EOF

# Load Docker image on instance
echo -e "${YELLOW}Loading Docker image on instance...${NC}"
ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP << 'EOF'
    cd ~/rag-system
    sudo docker load < rag-system.tar.gz
EOF

# Start services
echo -e "${YELLOW}Starting services...${NC}"
ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP << 'EOF'
    cd ~/rag-system
    sudo docker-compose up -d
EOF

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 60

# Test the API
echo -e "${YELLOW}Testing API...${NC}"
if curl -f http://$PUBLIC_IP/health; then
    echo -e "${GREEN}API is running successfully!${NC}"
    echo -e "${GREEN}API URL: http://$PUBLIC_IP${NC}"
    echo -e "${GREEN}API Docs: http://$PUBLIC_IP/docs${NC}"
else
    echo -e "${RED}API is not responding${NC}"
fi

echo -e "${GREEN}Deployment completed!${NC}"
echo -e "${GREEN}Instance ID: $INSTANCE_ID${NC}"
echo -e "${GREEN}Public IP: $PUBLIC_IP${NC}"
echo -e "${GREEN}SSH Command: ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP${NC}"
