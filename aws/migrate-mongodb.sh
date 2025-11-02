#!/bin/bash

# Script to migrate RAG application from local MongoDB to remote MongoDB instance
# This script helps configure the security group and update the RAG application

set -e

# Configuration
RAG_HOST="ec2-98-93-155-138.compute-1.amazonaws.com"
RAG_REGION="us-east-1"
MONGODB_HOST="ec2-3-25-73-98.ap-southeast-2.compute.amazonaws.com"
MONGODB_REGION="ap-southeast-2"
MONGODB_DATABASE="medicaldiagnosissystem"
MONGODB_PORT="27017"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== MongoDB Migration Script ===${NC}"
echo ""

# Step 1: Find RAG instance IP
echo -e "${YELLOW}Step 1: Finding RAG instance information...${NC}"
RAG_IP=$(dig +short $RAG_HOST | head -1)
if [ -z "$RAG_IP" ]; then
    echo -e "${RED}Failed to resolve RAG instance IP${NC}"
    exit 1
fi
echo -e "${GREEN}RAG instance IP: $RAG_IP${NC}"

# Step 2: Find MongoDB instance
echo -e "${YELLOW}Step 2: Finding MongoDB instance...${NC}"
MONGODB_INSTANCE_ID=$(aws ec2 describe-instances \
    --region $MONGODB_REGION \
    --filters "Name=instance-state-name,Values=running" \
    --query "Reservations[*].Instances[?contains(PublicDnsName, '3-25-73-98')].InstanceId" \
    --output text 2>/dev/null | head -1)

if [ -z "$MONGODB_INSTANCE_ID" ] || [ "$MONGODB_INSTANCE_ID" == "None" ]; then
    echo -e "${RED}Could not find MongoDB instance automatically${NC}"
    echo "Please find it manually and set MONGODB_INSTANCE_ID environment variable"
    exit 1
fi

echo -e "${GREEN}Found MongoDB instance: $MONGODB_INSTANCE_ID${NC}"

# Step 3: Get MongoDB security group
echo -e "${YELLOW}Step 3: Getting MongoDB security group...${NC}"
MONGODB_SG_ID=$(aws ec2 describe-instances \
    --region $MONGODB_REGION \
    --instance-ids $MONGODB_INSTANCE_ID \
    --query "Reservations[0].Instances[0].SecurityGroups[0].GroupId" \
    --output text)

echo -e "${GREEN}MongoDB Security Group: $MONGODB_SG_ID${NC}"

# Step 4: Check current security group rules
echo -e "${YELLOW}Step 4: Checking current security group rules...${NC}"
aws ec2 describe-security-groups \
    --region $MONGODB_REGION \
    --group-ids $MONGODB_SG_ID \
    --query "SecurityGroups[0].IpPermissions[?FromPort==\`$MONGODB_PORT\`]" \
    --output table

# Step 5: Check if rule already exists
echo ""
echo -e "${YELLOW}Step 5: Checking if port $MONGODB_PORT is accessible from RAG instance...${NC}"
HAS_RULE=$(aws ec2 describe-security-groups \
    --region $MONGODB_REGION \
    --group-ids $MONGODB_SG_ID \
    --query "SecurityGroups[0].IpPermissions[?FromPort==\`$MONGODB_PORT\` && (IpRanges[?CidrIp==\`$RAG_IP/32\`] || IpRanges[?CidrIp==\`0.0.0.0/0\`])]" \
    --output text)

if [ -n "$HAS_RULE" ]; then
    echo -e "${GREEN}✓ Port $MONGODB_PORT is already accessible${NC}"
else
    echo -e "${RED}✗ Port $MONGODB_PORT is NOT accessible from RAG instance${NC}"
    echo ""
    read -p "Do you want to add a security group rule to allow access? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Adding security group rule...${NC}"
        aws ec2 authorize-security-group-ingress \
            --region $MONGODB_REGION \
            --group-id $MONGODB_SG_ID \
            --protocol tcp \
            --port $MONGODB_PORT \
            --cidr $RAG_IP/32 && \
            echo -e "${GREEN}✓ Added rule for RAG instance IP ($RAG_IP/32)${NC}" || \
            aws ec2 authorize-security-group-ingress \
                --region $MONGODB_REGION \
                --group-id $MONGODB_SG_ID \
                --protocol tcp \
                --port $MONGODB_PORT \
                --cidr 0.0.0.0/0 && \
                echo -e "${GREEN}✓ Added rule for all IPs (0.0.0.0/0)${NC}" || \
                echo -e "${YELLOW}⚠ Rule may already exist${NC}"
    else
        echo -e "${YELLOW}To add the rule manually, run:${NC}"
        echo ""
        echo "aws ec2 authorize-security-group-ingress \\"
        echo "    --region $MONGODB_REGION \\"
        echo "    --group-id $MONGODB_SG_ID \\"
        echo "    --protocol tcp \\"
        echo "    --port $MONGODB_PORT \\"
        echo "    --cidr $RAG_IP/32"
        echo ""
    fi
fi

# Step 6: Update RAG application
echo ""
echo -e "${YELLOW}Step 6: Updating RAG application configuration...${NC}"
echo "MongoDB connection string will be:"
echo "mongodb://$MONGODB_HOST:$MONGODB_PORT/$MONGODB_DATABASE"
echo ""

read -p "Do you want to update the RAG application now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Connecting to RAG instance and updating configuration...${NC}"
    
    SSH_KEY="/Users/khoale/Downloads/COS30018/RAG/key/RAGServerKey.pem"
    
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ec2-user@$RAG_HOST << EOF
        cd ~/rag-system
        
        # Stop and remove old containers
        docker stop rag_api rag_mongodb 2>/dev/null || true
        docker rm rag_api rag_mongodb 2>/dev/null || true
        
        # Start RAG API with new MongoDB connection
        docker run -d \\
            --name rag_api \\
            --network rag_network \\
            -p 8000:8000 \\
            -e MONGODB_URL=mongodb://$MONGODB_HOST:$MONGODB_PORT/$MONGODB_DATABASE \\
            -e MONGODB_DATABASE=$MONGODB_DATABASE \\
            -e EMBEDDING_MODEL_PATH=/app/models/MedEmbed-large-v0.1 \\
            -e RERANKER_MODEL_PATH=/app/models/bge-reranker-v2-m3 \\
            -e HNSW_INDEX_PATH=/app/data/hnsw_index.bin \\
            -e HNSW_MAPPING_PATH=/app/data/hnsw_mapping.json \\
            -e DATA_DIR=/app/data \\
            -e DEBUG=false \\
            -e LOG_LEVEL=INFO \\
            -v \$(pwd)/models:/app/models \\
            -v \$(pwd)/data:/app/data \\
            -v \$(pwd)/logs:/app/logs \\
            -v \$(pwd)/storage:/app/storage \\
            --restart unless-stopped \\
            rag-api:latest
        
        echo "Waiting for application to start..."
        sleep 20
        
        # Test connection
        docker logs rag_api 2>&1 | tail -15
        echo ""
        echo "Testing MongoDB connection..."
        docker exec rag_api python3 -c "
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def test():
    try:
        client = AsyncIOMotorClient('mongodb://$MONGODB_HOST:$MONGODB_PORT/$MONGODB_DATABASE', serverSelectionTimeoutMS=10000)
        await client.admin.command('ping')
        print('✓ MongoDB connection successful')
        db = client.$MONGODB_DATABASE
        collections = await db.list_collection_names()
        print(f'✓ Collections found: {collections}')
        client.close()
    except Exception as e:
        print(f'✗ Connection failed: {e}')

asyncio.run(test())
" 2>&1
        
        # Test health endpoint
        sleep 5
        curl -s http://localhost/health | python3 -m json.tool 2>/dev/null || echo "Health check pending..."
EOF
    
    echo ""
    echo -e "${GREEN}Migration complete!${NC}"
    echo ""
    echo "Please verify:"
    echo "1. Security group allows connections on port $MONGODB_PORT"
    echo "2. MongoDB is accessible from RAG instance"
    echo "3. Application health check: http://$RAG_HOST/health"
else
    echo -e "${YELLOW}To update manually, connect to RAG instance and run:${NC}"
    echo ""
    echo "docker stop rag_api rag_mongodb"
    echo "docker rm rag_api rag_mongodb"
    echo "docker run -d --name rag_api --network rag_network -p 8000:8000 \\"
    echo "  -e MONGODB_URL=mongodb://$MONGODB_HOST:$MONGODB_PORT/$MONGODB_DATABASE \\"
    echo "  -e MONGODB_DATABASE=$MONGODB_DATABASE \\"
    echo "  [other environment variables] \\"
    echo "  rag-api:latest"
fi

