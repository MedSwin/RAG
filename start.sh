#!/bin/bash

# RAG System Startup Script
# This script starts the RAG system with proper configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Medical RAG System${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Check if MongoDB is running
echo -e "${YELLOW}Checking MongoDB connection...${NC}"
if ! python3 -c "import pymongo; pymongo.MongoClient('mongodb://localhost:27017').admin.command('ping')" &> /dev/null; then
    echo -e "${YELLOW}MongoDB is not running. Starting MongoDB with Docker...${NC}"
    docker run -d -p 27017:27017 --name rag_mongodb mongo:6.0
    sleep 10
fi

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p models data logs storage

# Check if models exist
if [ ! -d "models/MedEmbed-large-v0.1" ]; then
    echo -e "${YELLOW}Embedding model not found. Please download and place it in models/MedEmbed-large-v0.1${NC}"
    echo -e "${YELLOW}You can download it from Hugging Face or use the original model path${NC}"
fi

if [ ! -d "models/bge-reranker-v2-m3" ]; then
    echo -e "${YELLOW}Reranker model not found. Please download and place it in models/bge-reranker-v2-m3${NC}"
    echo -e "${YELLOW}This model is optional but recommended for better retrieval quality${NC}"
fi

# Set environment variables
export MONGODB_URL="mongodb://localhost:27017"
export MONGODB_DATABASE="medical_rag_db"
export EMBEDDING_MODEL_PATH="./models/MedEmbed-large-v0.1"
export RERANKER_MODEL_PATH="./models/bge-reranker-v2-m3"
export HNSW_INDEX_PATH="./data/hnsw_index.bin"
export HNSW_MAPPING_PATH="./data/hnsw_mapping.json"
export DATA_DIR="./data"
export DEBUG="true"
export LOG_LEVEL="INFO"

# Start the application
echo -e "${GREEN}Starting RAG API server...${NC}"
echo -e "${GREEN}API will be available at: http://localhost:8000${NC}"
echo -e "${GREEN}API documentation: http://localhost:8000/docs${NC}"
echo -e "${GREEN}Health check: http://localhost:8000/health${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"

python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
