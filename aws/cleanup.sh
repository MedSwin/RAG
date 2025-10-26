#!/bin/bash

# AWS EC2 Cleanup Script for RAG System
# This script cleans up AWS resources created for the RAG system

set -e

# Configuration
REGION="us-east-1"
KEY_NAME="rag-system-key"
SECURITY_GROUP="rag-system-sg"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting RAG System Cleanup${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Get instance ID
echo -e "${YELLOW}Finding RAG system instance...${NC}"
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=rag-system" "Name=instance-state-name,Values=running" \
    --region $REGION \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text)

if [ "$INSTANCE_ID" = "None" ] || [ -z "$INSTANCE_ID" ]; then
    echo -e "${YELLOW}No running RAG system instance found${NC}"
else
    echo -e "${GREEN}Found instance: $INSTANCE_ID${NC}"
    
    # Terminate instance
    echo -e "${YELLOW}Terminating instance...${NC}"
    aws ec2 terminate-instances \
        --instance-ids $INSTANCE_ID \
        --region $REGION
    
    # Wait for instance to be terminated
    echo -e "${YELLOW}Waiting for instance to be terminated...${NC}"
    aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID --region $REGION
    echo -e "${GREEN}Instance terminated${NC}"
fi

# Delete security group
echo -e "${YELLOW}Deleting security group...${NC}"
if aws ec2 describe-security-groups --group-names $SECURITY_GROUP --region $REGION &> /dev/null; then
    aws ec2 delete-security-group --group-name $SECURITY_GROUP --region $REGION
    echo -e "${GREEN}Security group deleted${NC}"
else
    echo -e "${YELLOW}Security group not found${NC}"
fi

# Delete key pair
echo -e "${YELLOW}Deleting key pair...${NC}"
if aws ec2 describe-key-pairs --key-names $KEY_NAME --region $REGION &> /dev/null; then
    aws ec2 delete-key-pair --key-name $KEY_NAME --region $REGION
    rm -f ${KEY_NAME}.pem
    echo -e "${GREEN}Key pair deleted${NC}"
else
    echo -e "${YELLOW}Key pair not found${NC}"
fi

# Clean up local files
echo -e "${YELLOW}Cleaning up local files...${NC}"
rm -f rag-system.tar.gz
rm -f ${KEY_NAME}.pem

echo -e "${GREEN}Cleanup completed!${NC}"
