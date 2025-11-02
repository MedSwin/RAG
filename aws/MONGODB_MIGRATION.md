# MongoDB Migration Guide

## Overview
This guide helps you migrate the RAG application from using a local MongoDB container to a remote MongoDB instance on a different EC2 instance.

## Configuration

- **Remote MongoDB Host**: `ec2-3-25-73-98.ap-southeast-2.compute.amazonaws.com`
- **MongoDB Region**: `ap-southeast-2` (Asia Pacific Sydney)
- **Database Name**: `medicaldiagnosissystem`
- **Collections**: `accounts`, `patients`, `emr`
- **RAG Instance**: `ec2-98-93-155-138.compute-1.amazonaws.com`
- **RAG Region**: `us-east-1` (US East N. Virginia)

## Prerequisites

1. **AWS CLI configured** with appropriate permissions
2. **Security Group Access**: The MongoDB EC2 instance security group must allow inbound connections on port 27017 from the RAG instance

## Migration Steps

### Option 1: Use the Automated Script

Run the migration script:
```bash
./aws/migrate-mongodb.sh
```

This script will:
1. Find both EC2 instances
2. Check and update security group rules
3. Update the RAG application configuration
4. Test the connection

### Option 2: Manual Migration

#### Step 1: Update MongoDB Security Group

The MongoDB EC2 instance security group needs to allow inbound connections on port 27017 from the RAG instance.

**Via AWS Console:**
1. Go to EC2 Console → Instances
2. Find the MongoDB instance (search for `3-25-73-98`)
3. Click on Security tab → Security Group
4. Click "Edit inbound rules"
5. Add rule:
   - **Type**: Custom TCP
   - **Port**: 27017
   - **Source**: IP of RAG instance (`98.93.155.138/32`) OR the RAG security group
6. Click "Save rules"

**Via AWS CLI:**
```bash
# Find MongoDB instance and security group
MONGODB_INSTANCE_ID=$(aws ec2 describe-instances \
    --region ap-southeast-2 \
    --filters "Name=instance-state-name,Values=running" \
    --query "Reservations[*].Instances[?contains(PublicDnsName, '3-25-73-98')].InstanceId" \
    --output text)

MONGODB_SG_ID=$(aws ec2 describe-instances \
    --region ap-southeast-2 \
    --instance-ids $MONGODB_INSTANCE_ID \
    --query "Reservations[0].Instances[0].SecurityGroups[0].GroupId" \
    --output text)

# Add rule to allow RAG instance IP
aws ec2 authorize-security-group-ingress \
    --region ap-southeast-2 \
    --group-id $MONGODB_SG_ID \
    --protocol tcp \
    --port 27017 \
    --cidr 98.93.155.138/32
```

#### Step 2: Update RAG Application

SSH into the RAG instance and update the container:

```bash
ssh -i key/RAGServerKey.pem ec2-user@ec2-98-93-155-138.compute-1.amazonaws.com

# Stop and remove old containers
docker stop rag_api rag_mongodb
docker rm rag_api rag_mongodb

# Start RAG API with new MongoDB connection
docker run -d \
    --name rag_api \
    --network rag_network \
    -p 8000:8000 \
    -e MONGODB_URL=mongodb://ec2-3-25-73-98.ap-southeast-2.compute.amazonaws.com:27017/medicaldiagnosissystem \
    -e MONGODB_DATABASE=medicaldiagnosissystem \
    -e EMBEDDING_MODEL_PATH=/app/models/MedEmbed-large-v0.1 \
    -e RERANKER_MODEL_PATH=/app/models/bge-reranker-v2-m3 \
    -e HNSW_INDEX_PATH=/app/data/hnsw_index.bin \
    -e HNSW_MAPPING_PATH=/app/data/hnsw_mapping.json \
    -e DATA_DIR=/app/data \
    -e DEBUG=false \
    -e LOG_LEVEL=INFO \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/storage:/app/storage \
    --restart unless-stopped \
    rag-api:latest
```

#### Step 3: Verify Connection

Test the connection from inside the container:

```bash
docker exec rag_api python3 -c "
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def test():
    client = AsyncIOMotorClient('mongodb://ec2-3-25-73-98.ap-southeast-2.compute.amazonaws.com:27017/medicaldiagnosissystem')
    await client.admin.command('ping')
    print('✓ Connection successful')
    db = client.medicaldiagnosissystem
    collections = await db.list_collection_names()
    print(f'Collections: {collections}')
    client.close()

asyncio.run(test())
"
```

Check the health endpoint:
```bash
curl http://localhost/health
```

## Troubleshooting

### Connection Timeout

If you see `ServerSelectionTimeoutError`, check:

1. **Security Group**: Verify MongoDB security group allows port 27017 from RAG instance
2. **MongoDB Configuration**: Ensure MongoDB is configured to accept external connections (not just localhost)
3. **Network ACLs**: Check if Network ACLs are blocking traffic
4. **Firewall**: Verify no host-level firewall is blocking port 27017

### Authentication Errors

If MongoDB requires authentication, update the connection string:
```
mongodb://username:password@ec2-3-25-73-98.ap-southeast-2.compute.amazonaws.com:27017/medicaldiagnosissystem?authSource=admin
```

### Region Connectivity

Since instances are in different regions (us-east-1 and ap-southeast-2), ensure:
- Both instances have public IPs or are in a VPC with peering
- Security groups allow cross-region communication
- No route tables are blocking the connection

## Connection String Format

**Without Authentication:**
```
mongodb://ec2-3-25-73-98.ap-southeast-2.compute.amazonaws.com:27017/medicaldiagnosissystem
```

**With Authentication:**
```
mongodb://username:password@ec2-3-25-73-98.ap-southeast-2.compute.amazonaws.com:27017/medicaldiagnosissystem?authSource=admin
```

## Post-Migration

After successful migration:
- The local MongoDB container (`rag_mongodb`) is no longer needed
- All RAG data will be stored in the remote `medicaldiagnosissystem` database
- Make sure to backup the remote MongoDB instance regularly

