#!/bin/bash

# Quick script to update RAG application to use remote MongoDB
# Run this AFTER updating the MongoDB security group

set -e

RAG_HOST="ec2-98-93-155-138.compute-1.amazonaws.com"
SSH_KEY="/Users/khoale/Downloads/COS30018/RAG/key/RAGServerKey.pem"
MONGODB_HOST="ec2-3-25-73-98.ap-southeast-2.compute.amazonaws.com"
MONGODB_DATABASE="medicaldiagnosissystem"

echo "Updating RAG application to use remote MongoDB..."
echo "MongoDB: $MONGODB_HOST"
echo "Database: $MONGODB_DATABASE"
echo ""

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ec2-user@$RAG_HOST << EOF
    cd ~/rag-system
    
    echo "Stopping old containers..."
    docker stop rag_api rag_mongodb 2>/dev/null || true
    docker rm rag_api rag_mongodb 2>/dev/null || true
    
    echo "Starting RAG API with remote MongoDB..."
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
    sleep 20
    
    echo ""
    echo "Checking logs..."
    docker logs rag_api 2>&1 | tail -20
    
    echo ""
    echo "Testing MongoDB connection..."
    docker exec rag_api python3 -c "
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def test():
    try:
        client = AsyncIOMotorClient('mongodb://$MONGODB_HOST:27017/$MONGODB_DATABASE', serverSelectionTimeoutMS=10000)
        await client.admin.command('ping')
        print('✓ MongoDB connection successful')
        db = client.$MONGODB_DATABASE
        collections = await db.list_collection_names()
        print(f'✓ Collections found: {collections}')
        client.close()
        exit(0)
    except Exception as e:
        print(f'✗ Connection failed: {e}')
        exit(1)

asyncio.run(test())
" 2>&1
    
    echo ""
    echo "Testing health endpoint..."
    sleep 5
    curl -s http://localhost/health | python3 -m json.tool 2>/dev/null || echo "Health check pending..."
    
    echo ""
    echo "Container status:"
    docker ps --format 'table {{.Names}}\t{{.Status}}'
EOF

echo ""
echo "Migration complete!"

