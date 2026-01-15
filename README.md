# MedSwin - Medical RAG System with Multi-Agent Conversation

A comprehensive Retrieval-Augmented Generation (RAG) system for medical document processing with **Multi-Agent Conversation (MAC)** orchestration, evidence sufficiency policies, two-stage retrieval + calibrated reranking, and enterprise-grade auditability. Built with FastAPI and designed for deployment on AWS EC2.

## Features

### Core RAG Features
- **Document Processing**: Upload and process medical documents (CSV, JSON, TXT, PDF)
- **Intelligent Chunking**: Smart text chunking with multiple strategies
- **Embedding Generation**: Generate embeddings using medical-specific models
- **Dynamic Vector Search**: Fast similarity search with automatic index selection (HNSW, FAISS-ivf, Tree), see [Indexing Strategy](docs/INDEXING.md)  
- **Reranking**: Improve retrieval quality with advanced reranking models
- **RESTful API**: Complete FastAPI-based API with automatic documentation
- **Docker Support**: Containerized deployment with Docker and Docker Compose
- **AWS Ready**: Pre-configured for AWS EC2 deployment

### MedSwin Features (New)
- **Multi-Agent Conversation (MAC)**: Supervisor + 3 specialist agents orchestration
- **Evidence Sufficiency Policies**: Deterministic gates with retrieve-more loops
- **Two-Stage Retrieval**: Dense vector search + optional BM25 lexical union
- **Calibrated Reranking**: Pointwise reranking with calibrated probabilities
- **Fusion Scoring**: Multi-factor scoring (reranker, dense, lexical, recency, section, source)
- **MMR Selection**: Maximal Marginal Relevance for diverse evidence selection
- **Enterprise Auditability**: Full audit trails with traces, sessions, and provenance
- **PHI-Safe Logging**: Configurable PHI redaction for compliance
- **Environment-Driven Endpoints**: All model endpoints configurable via environment variables

## Architecture

### MedSwin Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI App (Port 8100)                  │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              MedSwin Orchestrator                         │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │ │
│  │  │ Supervisor   │  │ Agent 1      │  │ Agent 2      │   │ │
│  │  │ (Port 8000)  │  │ Evidence     │  │ EMR          │   │ │
│  │  │              │  │ Retriever    │  │ Summariser   │   │ │
│  │  │              │  │ (Port 8001)  │  │ (Port 8002)  │   │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │ │
│  │                              │                            │ │
│  │                    ┌──────────────┐                      │ │
│  │                    │ Agent 3      │                      │ │
│  │                    │ Guideline    │                      │ │
│  │                    │ Synthesiser  │                      │ │
│  │                    │ (Port 8003)  │                      │ │
│  │                    └──────────────┘                      │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         Retrieval Pipeline                               │ │
│  │  • Two-stage retrieval (Dense + BM25)                    │ │
│  │  • Reranker (Port 8004)                                  │ │
│  │  • Fusion scoring                                        │ │
│  │  • MMR selection                                         │ │
│  │  • Evidence sufficiency checks                           │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MongoDB                                  │
│  • documents, chunks, sessions, traces                          │
│  • org-aware partitioning                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Dynamic Index (HNSW/FAISS/Tree)                     │
│  • Vector search                                                 │
│  • BM25 lexical index (optional)                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
rag/
├── app/                           # FastAPI production application
│   ├── api/                      # API endpoints
│   │   └── v1/
│   │       ├── endpoints/        # Endpoint modules
│   │       │   ├── dashboard.py  # Dashboard endpoint
│   │       │   ├── embedding.py  # Embedding endpoints
│   │       │   ├── medswin.py    # MedSwin endpoints (NEW)
│   │       │   ├── preprocessing.py
│   │       │   ├── retrieval.py
│   │       │   └── storage.py
│   │       └── router.py         # API router
│   ├── core/                     # Core configuration and infrastructure
│   │   ├── config.py             # Application settings (updated for MedSwin)
│   │   ├── database.py           # MongoDB connection
│   │   ├── state.py              # Global state management
│   │   └── indexing/             # Dynamic indexing
│   │       ├── base.py           # Base index interface
│   │       ├── hnsw.py           # HNSW index builder
│   │       ├── faiss.py          # FAISS index builder
│   │       └── tree.py           # Tree index builder
│   ├── models/                   # Data models (NEW)
│   │   ├── medswin.py            # MedSwin Pydantic models
│   │   ├── manager.py            # Model manager
│   │   └── download.py           # Model download utilities
│   ├── repositories/             # MongoDB repositories (NEW)
│   │   ├── base.py               # Base repository with org-aware partitioning
│   │   ├── chunks.py             # Chunk repository
│   │   ├── documents.py          # Document repository
│   │   ├── sessions.py           # Session repository
│   │   └── traces.py             # Trace repository
│   ├── services/                 # Business logic services
│   │   ├── adapters/             # External service adapters (NEW)
│   │   │   ├── llm.py            # LLM client (OpenAI-compatible)
│   │   │   ├── embedding.py      # Embedding service client
│   │   │   └── reranker.py       # Reranker service client
│   │   ├── medswin/              # MedSwin services (NEW)
│   │   │   ├── orchestrator.py   # Multi-agent orchestrator
│   │   │   ├── policy.py         # Evidence sufficiency policy
│   │   │   └── retrieval.py      # Two-stage retrieval pipeline
│   │   ├── dataset.py            # HuggingFace dataset service
│   │   ├── ingestion.py          # Ingestion pipeline service
│   │   ├── preprocessing.py      # Preprocessing service
│   │   ├── reranker.py           # Reranker service
│   │   ├── storage.py            # Storage service
│   │   └── strategy.py           # Index strategy manager
│   └── main.py                   # Application entry point
├── lab/                          # Lab environment (HPC/research)
│   ├── preprocessing/            # Data preprocessing and chunking
│   ├── embedding/                # Embedding model operations
│   ├── storage/                  # Data storage and indexing
│   ├── retrieval/                # Document retrieval
│   ├── reranking/                # Document reranking
│   ├── database/                 # Database schema and utilities
│   └── README.md                 # Lab setup documentation
├── docs/                         # Documentation (NEW)
│   ├── MEDSWIN.md                # MedSwin architecture guide
│   └── INDEXING.md               # Dynamic indexing strategy
├── aws/                          # AWS deployment scripts
│   ├── deploy.sh
│   └── cleanup.sh
├── nginx/                         # Nginx configuration
│   └── nginx.conf
├── docker-compose.yml             # Docker Compose setup
├── Dockerfile                     # Docker configuration
├── env.example                    # Environment variables template (updated)
├── requirements.txt               # Python dependencies (updated)
├── README.md                      # This file
└── TODO.md                        # Implementation requirements
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

5. **Set up model endpoints** (required for MedSwin)
   
   MedSwin requires external model endpoints. Configure in `.env`:
   - Supervisor: `SUPERVISOR_URL=http://localhost:8000/v1/chat/completions`
   - Agent 1: `AGENT1_URL=http://localhost:8001/v1/chat/completions`
   - Agent 2: `AGENT2_URL=http://localhost:8002/v1/chat/completions`
   - Agent 3: `AGENT3_URL=http://localhost:8003/v1/chat/completions`
   - Reranker: `RERANKER_URL=http://localhost:8004/rerank`
   - Embedding: `EMBEDDING_URL=http://localhost:8005/embeddings`
   
   See [MedSwin Setup Guide](docs/MEDSWIN.md) for details.

6. **Run the application**
   ```bash
   python -m uvicorn app.main:app --reload
   ```

7. **Access the API**
   - API: http://localhost:8100 (changed from 8000 to avoid conflict with supervisor)
   - Documentation: http://localhost:8100/docs
   - Health Check: http://localhost:8100/health

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Access the application**
   - API: http://localhost:8100
   - Documentation: http://localhost:8100/docs

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

### MedSwin Endpoints (New)
- `POST /api/v1/medswin/chat` - Process chat query through multi-agent pipeline
- `GET /api/v1/medswin/sessions/{session_id}` - Get session summary
- `GET /api/v1/medswin/traces/{trace_id}` - Get audit trace (admin only)
- `POST /api/v1/medswin/ingest` - Ingest CPG or EMR documents

### Legacy RAG Endpoints (Preserved)
#### Preprocessing
- `POST /api/v1/preprocessing/chunk` - Chunk text data
- `POST /api/v1/preprocessing/upload-and-chunk` - Upload and chunk files
- `GET /api/v1/preprocessing/info` - Get preprocessing information

#### Embedding
- `POST /api/v1/embedding/embed` - Embed single text
- `POST /api/v1/embedding/embed/batch` - Embed multiple texts
- `GET /api/v1/embedding/info` - Get embedding model information

#### Retrieval
- `POST /api/v1/retrieval/search` - Search for documents
- `GET /api/v1/retrieval/search` - Search with query parameters
- `GET /api/v1/retrieval/index/info` - Get index information

#### Storage
- `POST /api/v1/storage/chunks` - Store chunks
- `POST /api/v1/storage/index/build` - Build dynamic index
- `GET /api/v1/storage/stats` - Get storage statistics
- `DELETE /api/v1/storage/chunks` - Clear chunks

## Configuration

### Environment Variables

See `env.example` for complete configuration. Key variables:

#### Core Application
| Variable | Description | Default |
|----------|-------------|---------|
| `APP_HOST` | Application host | `0.0.0.0` |
| `APP_PORT` | Application port | `8100` |
| `MONGODB_URL` | MongoDB connection string | `mongodb://localhost:27017` |
| `MONGODB_DB` | MongoDB database name | `medswin` |

#### Model Endpoints (OpenAI-compatible)
| Variable | Description | Default |
|----------|-------------|---------|
| `SUPERVISOR_URL` | Supervisor LLM endpoint | `http://localhost:8000/v1/chat/completions` |
| `AGENT1_URL` | Agent 1 (Evidence Retriever) endpoint | `http://localhost:8001/v1/chat/completions` |
| `AGENT2_URL` | Agent 2 (EMR Summariser) endpoint | `http://localhost:8002/v1/chat/completions` |
| `AGENT3_URL` | Agent 3 (Guideline Synthesiser) endpoint | `http://localhost:8003/v1/chat/completions` |
| `RERANKER_URL` | Reranker service endpoint | `http://localhost:8004/rerank` |
| `EMBEDDING_URL` | Embedding service endpoint | `http://localhost:8005/embeddings` |

#### Retrieval Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `CANDIDATE_K` | Initial candidate pool size | `80` |
| `CANDIDATE_K_PRIME` | Expanded candidate pool | `120` |
| `MAX_RETRIEVE_LOOPS` | Max sufficiency loop iterations | `3` |
| `TOKEN_BUDGET_B` | Token budget for evidence | `1800` |
| `ENABLE_BM25` | Enable BM25 lexical retrieval | `true` |
| `DEFAULT_INDEX_TYPE` | Default index type | `hnsw` |

#### Evidence Sufficiency Policy
| Variable | Description | Default |
|----------|-------------|---------|
| `SUFF_T_CPG` | Target CPG passages | `2` |
| `SUFF_T_EMR` | Target EMR passages | `2` |
| `SUFF_T_INCLUSION` | Inclusion threshold | `0.55` |
| `SUFF_T_MEAN_CONF` | Mean confidence threshold | `0.60` |

#### Fusion Weights (must sum to 1.0)
| Variable | Description | Default |
|----------|-------------|---------|
| `W_RERANK` | Reranker weight | `0.45` |
| `W_DENSE` | Dense similarity weight | `0.25` |
| `W_LEX` | Lexical (BM25) weight | `0.10` |
| `W_RECENCY` | Recency weight | `0.07` |
| `W_SECTION` | Section weight | `0.08` |
| `W_SOURCE` | Source type weight | `0.05` |

For complete configuration, see `env.example`.

### Model Requirements

MedSwin supports two deployment modes:

#### Mode 1: External Model Endpoints (Recommended for MedSwin)
All models run as separate services with OpenAI-compatible APIs:
- **Supervisor LLM**: Port 8000 (`/v1/chat/completions`)
- **Agent 1-3 LLMs**: Ports 8001-8003 (`/v1/chat/completions`)
- **Reranker**: Port 8004 (`/rerank`)
- **Embedding**: Port 8005 (`/embeddings`)

Configure endpoints in `.env` file. See [MedSwin Setup Guide](docs/MEDSWIN.md).

#### Mode 2: Local Models (Legacy RAG)
For legacy RAG endpoints, models can be placed locally:

1. **Embedding Model**: `MedEmbed-large-v0.1`
   - Medical-specific embedding model
   - Should be compatible with Hugging Face transformers
   - Path: `EMBEDDING_MODEL_PATH` in `.env`

2. **Reranker Model**: `bge-reranker-v2-m3`
   - Optional reranking model
   - Improves retrieval quality
   - Path: `RERANKER_MODEL_PATH` in `.env`

## Usage Examples

### 1. MedSwin Chat (Multi-Agent Conversation)

```python
import requests

# Process a medical query through MedSwin pipeline
response = requests.post(
    'http://localhost:8100/api/v1/medswin/chat',
    json={
        'query': 'What are the treatment guidelines for type 2 diabetes in elderly patients?',
        'user_id': 'user123',
        'org_id': 'org456',
        'patient_id': 'patient789',  # Optional
        'constraints': {
            'guideline_only': True  # Optional constraints
        }
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Evidence: {len(result['evidence_bundle']['passages'])} passages")
print(f"Trace ID: {result['trace_id']}")
```

### 2. Upload and Process Documents

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
        'http://localhost:8100/api/v1/preprocessing/upload-and-chunk',
        files=files,
        data=data
    )
    chunks = response.json()
```

### 3. Ingest CPG or EMR Documents

```python
# Ingest clinical practice guidelines
response = requests.post(
    'http://localhost:8100/api/v1/medswin/ingest',
    params={
        'source_type': 'CPG',
        'org_id': 'org456'
    },
    json={
        'documents': [
            {
                'doc_id': 'guideline_001',
                'title': 'Type 2 Diabetes Management',
                'text': '...',
                'version': '2024.1',
                'metadata': {}
            }
        ]
    }
)
```

### 4. Search for Relevant Documents (Legacy)

```python
# Search for relevant documents
query = "What are the symptoms of diabetes?"
response = requests.post(
    'http://localhost:8100/api/v1/retrieval/search',
    json={
        'query': query,
        'top_k': 5,
        'use_reranking': True
    }
)
results = response.json()
```

### 5. Get Audit Trace

```python
# Get audit trace for a request
response = requests.get(
    'http://localhost:8100/api/v1/medswin/traces/{trace_id}',
    params={'org_id': 'org456'}
)
trace = response.json()
```

## Development

### Project Structure Details

#### Core Application (`app/`)
- **`app/api/v1/endpoints/`**: All API endpoint modules
  - `medswin.py`: MedSwin multi-agent chat endpoints
  - `dashboard.py`, `embedding.py`, `preprocessing.py`, `retrieval.py`, `storage.py`: Legacy RAG endpoints
  
- **`app/core/`**: Core infrastructure
  - `config.py`: Centralized configuration (all environment variables)
  - `database.py`: MongoDB connection and initialization
  - `indexing/`: Dynamic index builders (HNSW, FAISS, Tree)

- **`app/models/`**: Pydantic data models
  - `medswin.py`: MedSwin typed artifacts (QuerySpec, EvidenceBundle, AuditTrace, etc.)
  - `manager.py`, `download.py`: Model management utilities

- **`app/repositories/`**: MongoDB data access layer
  - All repositories support org-aware partitioning for multi-tenancy
  - `base.py`: Base repository with common functionality
  - `chunks.py`, `documents.py`, `sessions.py`, `traces.py`: Domain repositories

- **`app/services/`**: Business logic services
  - **`adapters/`**: External HTTP service clients
    - `llm.py`: OpenAI-compatible LLM client with retries
    - `embedding.py`: Embedding service client
    - `reranker.py`: Reranker service client
  - **`medswin/`**: MedSwin-specific services
    - `orchestrator.py`: Multi-agent orchestration (supervisor + 3 agents)
    - `policy.py`: Evidence sufficiency policy with deterministic gates
    - `retrieval.py`: Two-stage retrieval pipeline (dense + BM25, reranking, fusion, MMR)
  - Legacy services: `preprocessing.py`, `storage.py`, `reranker.py`, `strategy.py`, etc.

### Adding New Features

1. **New API Endpoints**: Add to `app/api/v1/endpoints/` and register in `router.py`
2. **New Services**: Add to `app/services/` (use `adapters/` for external services)
3. **New Models**: Add Pydantic models to `app/models/`
4. **New Repositories**: Extend `app/repositories/base.py` for new data access patterns
5. **Configuration**: Update `app/core/config.py` and `env.example`

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

1. **Model Endpoint Connection Errors (MedSwin)**
   - Verify all model endpoints are running and accessible
   - Check `SUPERVISOR_URL`, `AGENT1_URL`, `AGENT2_URL`, `AGENT3_URL`, `RERANKER_URL`, `EMBEDDING_URL` in `.env`
   - Test endpoints with `curl` or `httpx` to verify OpenAI-compatible format
   - Check timeout settings (`LLM_TIMEOUT_S`, `RERANK_TIMEOUT_S`, `EMBED_TIMEOUT_S`)

2. **MongoDB Connection Issues**
   - Verify MongoDB is running: `docker ps | grep mongo`
   - Check connection string: `MONGODB_URL` in `.env`
   - Ensure database name matches: `MONGODB_DB=medswin`
   - Check network connectivity if using remote MongoDB

3. **Evidence Sufficiency Loops Not Triggering**
   - Verify `SUFF_T_CPG` and `SUFF_T_EMR` targets are set appropriately
   - Check `SUFF_T_MEAN_CONF` threshold (default: 0.60)
   - Ensure sufficient evidence exists in database for query
   - Review `MAX_RETRIEVE_LOOPS` setting (default: 3)

4. **Retrieval Performance Issues**
   - Increase `CANDIDATE_K` for better recall (default: 80)
   - Enable BM25: `ENABLE_BM25=true` for rare terms/acronyms
   - Adjust fusion weights if scoring is unbalanced
   - Monitor token budget usage: `TOKEN_BUDGET_B` (default: 1800)

5. **Memory Issues**
   - Reduce batch size in configuration
   - Use smaller models for development
   - Limit `MMR_MAX_EVIDENCE_CHUNKS` (default: 10)

6. **GPU Issues** (for local models)
   - Ensure CUDA is properly installed
   - Check GPU availability in Docker
   - Use CPU mode if GPU unavailable

### Logs

Check application logs for detailed error information:

```bash
# Docker logs
docker-compose logs rag_api

# Application logs
tail -f /app/logs/app.log

# Check specific service logs
docker-compose logs rag_api | grep -i "medswin\|orchestrator\|retrieval"
```

### Debugging MedSwin Pipeline

1. **Check Trace**: Use `/api/v1/medswin/traces/{trace_id}` to inspect full pipeline execution
2. **Verify Sufficiency Checks**: Review `sufficiency_checks` in trace to see why loops triggered/failed
3. **Inspect Evidence Bundle**: Check `evidence_bundle` in response for passage counts and coverage
4. **Review Agent Messages**: Check `messages` in trace to see agent interactions
5. **Monitor Degraded Mode**: Check `degraded_mode` flags in response for service failures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Documentation

### Comprehensive Guides
- **[MedSwin Architecture Guide](docs/MEDSWIN.md)**: Complete guide to MedSwin multi-agent architecture, retrieval pipeline, evidence sufficiency, and enterprise features
- **[Dynamic Indexing Strategy](docs/INDEXING.md)**: Guide to HNSW, FAISS, and Tree indexing with MedSwin integration
- **[API Documentation](http://localhost:8100/docs)**: Interactive Swagger/OpenAPI documentation

### Key Concepts

#### Multi-Agent Conversation (MAC)
- **Supervisor**: Orchestrates workflow, normalizes queries, performs safety critique, generates final answer
- **Agent 1 (Evidence Retriever)**: Searches CPG + EMR indices with tool calls
- **Agent 2 (EMR Summariser)**: Produces structured patient state summaries
- **Agent 3 (Guideline Synthesiser)**: Extracts actionable recommendations and contraindications

#### Retrieval Pipeline
1. **Stage 1**: Dense vector search (HNSW/FAISS/Tree) + optional BM25 lexical union
2. **Reranking**: Pointwise reranking with calibrated probabilities
3. **Fusion Scoring**: Multi-factor scoring combining reranker, dense, lexical, recency, section, and source scores
4. **MMR Selection**: Maximal Marginal Relevance for diverse evidence under token budget
5. **Sufficiency Checks**: Deterministic gates with retrieve-more loops

#### Enterprise Features
- **Audit Traces**: Complete request traces with messages, tool calls, evidence bundle, sufficiency checks
- **Org-Aware Partitioning**: All data access isolated by `org_id` for multi-tenancy
- **PHI-Safe Logging**: Configurable PHI redaction for HIPAA compliance
- **Environment-Driven**: All model endpoints configurable via environment variables

## Key Files Reference

### Configuration Files
- **`env.example`**: Complete environment variable template with all MedSwin settings
- **`app/core/config.py`**: Centralized configuration with validation (fusion weights, etc.)

### Core Implementation Files
- **`app/services/medswin/orchestrator.py`**: Main orchestrator coordinating supervisor + agents
- **`app/services/medswin/retrieval.py`**: Two-stage retrieval pipeline with BM25, reranking, fusion, MMR
- **`app/services/medswin/policy.py`**: Evidence sufficiency policy with deterministic gates
- **`app/services/adapters/llm.py`**: LLM client with retries and timeouts
- **`app/services/adapters/embedding.py`**: Embedding service client
- **`app/services/adapters/reranker.py`**: Reranker service client

### Data Models
- **`app/models/medswin.py`**: All Pydantic models (QuerySpec, EvidenceBundle, AuditTrace, etc.)

### Repositories
- **`app/repositories/base.py`**: Base repository with org-aware partitioning
- **`app/repositories/chunks.py`**, **`documents.py`**, **`sessions.py`**, **`traces.py`**: Domain repositories

### API Endpoints
- **`app/api/v1/endpoints/medswin.py`**: MedSwin chat, sessions, traces, ingest endpoints

## Migration Notes

### From Legacy RAG to MedSwin

1. **Port Change**: Application now runs on port **8100** (was 8000) to avoid conflict with supervisor
2. **Database Name**: Default database changed to **`medswin`** (was `medical_rag_db`)
3. **New Dependencies**: Added `rank-bm25`, `tiktoken`, `tenacity` to `requirements.txt`
4. **Environment Variables**: Many new variables added - see `env.example`
5. **Backward Compatibility**: All legacy RAG endpoints (`/preprocessing`, `/embedding`, `/retrieval`, `/storage`) remain functional

### Breaking Changes
- None - all changes are additive. Legacy endpoints continue to work.

## Support

For support and questions:
- Create an issue in the repository
- Check the documentation:
  - [MedSwin Architecture Guide](docs/MEDSWIN.md)
  - [Indexing Strategy](docs/INDEXING.md)
  - [Interactive API Docs](http://localhost:8100/docs)
- Review the troubleshooting section above
