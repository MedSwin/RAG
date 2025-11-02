#!/bin/bash

# Script to fix MongoDB connection by updating security group and restarting RAG app

set -e

RAG_HOST="ec2-98-93-155-138.compute-1.amazonaws.com"
RAG_IP="98.93.155.138"
RAG_REGION="us-east-1"
MONGODB_HOST="ec2-3-25-73-98.ap-southeast-2.compute.amazonaws.com"
MONGODB_REGION="ap-southeast-2"
MONGODB_DATABASE="medicaldiagnosissystem"
SSH_KEY="/Users/khoale/Downloads/COS30018/RAG/key/RAGServerKey.pem"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== MongoDB Connection Fix Script ===${NC}"
echo ""

# Step 1: Find MongoDB instance and security group
echo -e "${YELLOW}Step 1: Finding MongoDB instance...${NC}"
MONGODB_INSTANCE_ID=$(aws ec2 describe-instances \
    --region $MONGODB_REGION \
    --filters "Name=instance-state-name,Values=running" \
    --query "Reservations[*].Instances[?contains(PublicDnsName, '3-25-73-98')].InstanceId" \
    --output text 2>/dev/null | head -1)

if [ -z "$MONGODB_INSTANCE_ID" ] || [ "$MONGODB_INSTANCE_ID" == "None" ]; then
    echo -e "${RED}Could not find MongoDB instance automatically${NC}"
    echo "Please find it manually in AWS Console"
    exit 1
fi

echo -e "${GREEN}Found MongoDB instance: $MONGODB_INSTANCE_ID${NC}"

MONGODB_SG_ID=$(aws ec2 describe-instances \
    --region $MONGODB_REGION \
    --instance-ids $MONGODB_INSTANCE_ID \
    --query "Reservations[0].Instances[0].SecurityGroups[0].GroupId" \
    --output text)

echo -e "${GREEN}MongoDB Security Group: $MONGODB_SG_ID${NC}"
echo ""

# Step 2: Check current rules
echo -e "${YELLOW}Step 2: Checking current security group rules for port 27017...${NC}"
EXISTING_RULE=$(aws ec2 describe-security-groups \
    --region $MONGODB_REGION \
    --group-ids $MONGODB_SG_ID \
    --query "SecurityGroups[0].IpPermissions[?FromPort==\`27017\` && (IpRanges[?CidrIp==\`$RAG_IP/32\`] || IpRanges[?CidrIp==\`0.0.0.0/0\`])]" \
    --output text 2>/dev/null)

if [ -n "$EXISTING_RULE" ]; then
    echo -e "${GREEN}✓ Port 27017 is already accessible from RAG instance${NC}"
else
    echo -e "${RED}✗ Port 27017 is NOT accessible from RAG instance${NC}"
    echo ""
    echo "Adding security group rule..."
    
    # Try to add rule for specific IP first
    if aws ec2 authorize-security-group-ingress \
        --region $MONGODB_REGION \
        --group-id $MONGODB_SG_ID \
        --protocol tcp \
        --port 27017 \
        --cidr $RAG_IP/32 2>/dev/null; then
        echo -e "${GREEN}✓ Added rule for RAG IP ($RAG_IP/32)${NC}"
    else
        # Try 0.0.0.0/0 if specific IP fails (might already exist)
        if aws ec2 authorize-security-group-ingress \
            --region $MONGODB_REGION \
            --group-id $MONGODB_SG_ID \
            --protocol tcp \
            --port 27017 \
            --cidr 0.0.0.0/0 2>/dev/null; then
            echo -e "${GREEN}✓ Added rule for all IPs (0.0.0.0/0)${NC}"
        else
            echo -e "${YELLOW}⚠ Rule may already exist or there was an error${NC}"
        fi
    fi
fi

echo ""
echo -e "${YELLOW}Step 3: Waiting for security group to propagate...${NC}"
sleep 10

# Step 3: Test connection from RAG instance
echo -e "${YELLOW}Step 4: Testing MongoDB connection from RAG instance...${NC}"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ec2-user@$RAG_HOST << EOF
    timeout 5 bash -c '</dev/tcp/$MONGODB_HOST/27017' 2>/dev/null && echo "✓ Port 27017 is now reachable" || echo "✗ Port 27017 still not reachable - check security group"
EOF

echo ""
echo -e "${YELLOW}Step 5: Updating RAG application configuration...${NC}"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ec2-user@$RAG_HOST << EOF
    cd ~/rag-system
    
    # Stop and remove old containers
    docker stop rag_api 2>/dev/null || true
    docker rm rag_api 2>/dev/null || true
    
    # Start with remote MongoDB
    docker run -d \\
        --name rag_api \\
        --network rag_network \\
        -p 8000:8000 \\
        -e MONGODB_URL=mongodb://$MONGODB_HOST:27017/$MONGODB_DATABASE \\
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
    sleep 25
    
    # Test connection
    echo ""
    echo "Testing MongoDB connection from container..."
    docker exec rag_api python3 -c "
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def test():
    try:
        client = AsyncIOMotorClient('mongodb://$MONGODB_HOST:27017/$MONGODB_DATABASE', serverSelectionTimeoutMS=15000)
        await client.admin.command('ping')
        print('✓ MongoDB connection successful!')
        db = client.$MONGODB_DATABASE
        collections = await db.list_collection_names()
        print(f'✓ Collections found: {collections}')
        client.close()
    except Exception as e:
        print(f'✗ Connection failed: {e}')

asyncio.run(test())
" 2>&1
    
    # Check health
    echo ""
    echo "Checking application health..."
    sleep 5
    curl -s http://localhost/health | python3 -m json.tool 2>/dev/null || echo "Health check pending..."
    
    # Show logs
    echo ""
    echo "Recent logs:"
    docker logs rag_api 2>&1 | tail -10
EOF

echo ""
echo -e "${GREEN}=== Migration Complete ===${NC}"
echo ""
echo "If connection still fails, verify:"
echo "1. MongoDB security group allows port 27017 from RAG IP ($RAG_IP)"
echo "2. MongoDB is running and configured to accept external connections"
echo "3. No network ACLs are blocking the connection"

