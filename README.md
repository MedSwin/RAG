# Medical RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for medical document processing, built with FastAPI and designed for deployment on AWS EC2.

## Features

- **Document Processing**: Upload and process medical documents (CSV, JSON, TXT)
- **Intelligent Chunking**: Smart text chunking with multiple strategies
- **Embedding Generation**: Generate embeddings using medical-specific models
- **Vector Search**: Fast similarity search using HNSW indexing
- **Reranking**: Improve retrieval quality with advanced reranking models
- **RESTful API**: Complete FastAPI-based API with automatic documentation
- **Docker Support**: Containerized deployment with Docker and Docker Compose
- **AWS Ready**: Pre-configured for AWS EC2 deployment

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   MongoDB       │    │   HNSW Index    │
│                 │    │                 │    │                 │
│ - Preprocessing │◄──►│ - Chunks        │◄──►│ - Vector Search │
│ - Embedding     │    │ - Metadata      │    │ - Similarity    │
│ - Retrieval     │    │ - Relationships │    │ - Mapping       │
│ - Storage       │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
RAG/
├── app/                    # FastAPI production application
│   ├── api/               # API endpoints
│   ├── core/              # Core configuration
│   ├── services/          # Business logic services
│   └── main.py            # Application entry point
├── lab/                    # Lab environment (HPC/research)
│   ├── preprocessing/     # Data preprocessing and chunking
│   ├── embedding/         # Embedding model operations
│   ├── storage/           # Data storage and indexing
│   ├── retrieval/         # Document retrieval
│   ├── reranking/         # Document reranking
│   ├── database/          # Database schema and utilities
│   └── README.md          # Lab setup documentation
├── aws/                   # AWS deployment scripts
├── nginx/                 # Nginx configuration
├── docker-compose.yml     # Docker Compose setup
├── Dockerfile             # Docker configuration
└── requirements.txt       # Python dependencies
```

## 🔬 Lab Environment

The `lab/` directory contains the original research and development components designed for HPC environments. These modules provide:

- **Research Tools**: Direct access to preprocessing, embedding, and retrieval functions
- **HPC Optimization**: Optimized for high-performance computing environments
- **Experimentation**: Easy testing of new chunking strategies and models
- **Development**: Isolated development environment for algorithm improvements

For detailed lab setup and usage instructions, see [lab/README.md](lab/README.md).

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Start MongoDB**
   ```bash
   docker run -d -p 27017:27017 --name mongodb mongo:6.0
   ```

5. **Run the application**
   ```bash
   python -m uvicorn app.main:app --reload
   ```

6. **Access the API**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Access the application**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs

### AWS EC2 Deployment

1. **Prerequisites**
   - AWS CLI configured
   - Docker installed locally
   - EC2 key pair created

2. **Deploy to AWS**
   ```bash
   chmod +x aws/deploy.sh
   ./aws/deploy.sh
   ```

3. **Clean up resources**
   ```bash
   chmod +x aws/cleanup.sh
   ./aws/cleanup.sh
   ```

## API Endpoints

### Preprocessing
- `POST /api/v1/preprocessing/chunk` - Chunk text data
- `POST /api/v1/preprocessing/upload-and-chunk` - Upload and chunk files
- `GET /api/v1/preprocessing/info` - Get preprocessing information

### Embedding
- `POST /api/v1/embedding/embed` - Embed single text
- `POST /api/v1/embedding/embed/batch` - Embed multiple texts
- `GET /api/v1/embedding/info` - Get embedding model information

### Retrieval
- `POST /api/v1/retrieval/search` - Search for documents
- `GET /api/v1/retrieval/search` - Search with query parameters
- `GET /api/v1/retrieval/index/info` - Get index information

### Storage
- `POST /api/v1/storage/chunks` - Store chunks
- `POST /api/v1/storage/index/build` - Build HNSW index
- `GET /api/v1/storage/stats` - Get storage statistics
- `DELETE /api/v1/storage/chunks` - Clear chunks

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URL` | MongoDB connection string | `mongodb://localhost:27017` |
| `EMBEDDING_MODEL_PATH` | Path to embedding model | `/app/models/MedEmbed-large-v0.1` |
| `RERANKER_MODEL_PATH` | Path to reranker model | `/app/models/bge-reranker-v2-m3` |
| `HNSW_INDEX_PATH` | Path to HNSW index file | `/app/data/hnsw_index.bin` |
| `DEFAULT_TOP_K` | Default number of results | `5` |
| `TARGET_CHUNK_SIZE` | Target chunk size in tokens | `400` |

### Model Requirements

The system requires the following models to be placed in the `/app/models/` directory:

1. **Embedding Model**: `MedEmbed-large-v0.1`
   - Medical-specific embedding model
   - Should be compatible with Hugging Face transformers

2. **Reranker Model**: `bge-reranker-v2-m3`
   - Optional reranking model
   - Improves retrieval quality

## Usage Examples

### 1. Upload and Process Documents

```python
import requests

# Upload a CSV file
with open('medical_data.csv', 'rb') as f:
    files = {'file': f}
    data = {
        'chunking_strategy': 'auto',
        'target_chunk_size': 400
    }
    response = requests.post(
        'http://localhost:8000/api/v1/preprocessing/upload-and-chunk',
        files=files,
        data=data
    )
    chunks = response.json()
```

### 2. Search for Relevant Documents

```python
# Search for relevant documents
query = "What are the symptoms of diabetes?"
response = requests.post(
    'http://localhost:8000/api/v1/retrieval/search',
    json={
        'query': query,
        'top_k': 5,
        'use_reranking': True
    }
)
results = response.json()
```

### 3. Generate Embeddings

```python
# Generate embeddings for text
response = requests.post(
    'http://localhost:8000/api/v1/embedding/embed',
    json={
        'text': 'Patient has high blood pressure',
        'normalize': True
    }
)
embedding = response.json()
```

## Development

### Project Structure

```
RAG/
├── app/                    # FastAPI application
│   ├── api/               # API endpoints
│   ├── core/              # Core configuration
│   └── services/          # Business logic services
├── aws/                   # AWS deployment scripts
├── database/              # Database initialization
├── nginx/                 # Nginx configuration
├── preprocessing/         # Original preprocessing code
├── embedding/             # Original embedding code
├── retrieval/             # Original retrieval code
├── storage/               # Original storage code
├── reranking/             # Original reranking code
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile            # Docker configuration
└── requirements.txt      # Python dependencies
```

### Adding New Features

1. **New API Endpoints**: Add to `app/api/v1/endpoints/`
2. **New Services**: Add to `app/services/`
3. **New Models**: Add to `app/models/`
4. **Configuration**: Update `app/core/config.py`

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app
```

## Monitoring and Logging

- **Health Check**: `/health` endpoint
- **Metrics**: Prometheus metrics available at `/metrics`
- **Logs**: Structured logging to `/app/logs/app.log`
- **API Documentation**: Available at `/docs`

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure models are in the correct directory
   - Check model compatibility with transformers

2. **MongoDB Connection Issues**
   - Verify MongoDB is running
   - Check connection string in configuration

3. **Memory Issues**
   - Reduce batch size in configuration
   - Use smaller models for development

4. **GPU Issues**
   - Ensure CUDA is properly installed
   - Check GPU availability in Docker

### Logs

Check application logs for detailed error information:

```bash
# Docker logs
docker-compose logs rag_api

# Application logs
tail -f /app/logs/app.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the documentation at `/docs`
- Review the troubleshooting section
