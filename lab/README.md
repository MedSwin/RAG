# Lab Environment Setup

This directory contains the original lab environment components for the Medical RAG System. These modules were designed for HPC (High Performance Computing) environments and local development before being integrated into the production FastAPI application.

## 📁 Directory Structure

```
lab/
├── database/           # Database schema and utilities
├── embedding/          # Embedding model operations
├── preprocessing/      # Data preprocessing and chunking
├── reranking/         # Document reranking functionality
├── retrieval/          # Document retrieval operations
├── storage/           # Data storage and indexing
└── README.md          # This file
```

## 🔧 Components Overview

### 1. Database (`database/`)
**Purpose**: MongoDB schema management and database utilities

**Files**:
- `create_schema.py` - Creates MongoDB collections with proper schemas and indexes
- `init-mongo.js` - MongoDB initialization script for Docker
- `utils_db/schema_check.py` - Database schema validation utilities

**Key Features**:
- Collection schemas for `chunks`, `chunk_relationships`, `search_indexes`
- Automatic index creation for performance optimization
- Schema validation and integrity checks

**Usage**:
```bash
cd lab/database
python create_schema.py
```

### 2. Embedding (`embedding/`)
**Purpose**: Text embedding generation using Hugging Face models

**Files**:
- `embedding.py` - Main embedding functions and model management
- `embed_query.py` - Query embedding functionality
- `__init__.py` - Module exports

**Key Features**:
- Support for multiple embedding models (MedEmbed-large-v0.1, etc.)
- Batch embedding generation for efficiency
- GPU acceleration support
- Token counting and text preprocessing

**Usage**:
```python
from lab.embedding import load_embed_model, embedding_text

# Load model
model, tokenizer = load_embed_model("MedAI-COS30018/MedEmbed-large-v0.1")

# Generate embeddings
embeddings = embedding_text(["text1", "text2"], model, tokenizer)
```

### 3. Preprocessing (`preprocessing/`)
**Purpose**: Data preprocessing, cleaning, and intelligent chunking

**Files**:
- `chunker.py` - Advanced chunking strategies and text splitting
- `data_preprocessing.py` - Data cleaning and preparation
- `reader.py` - Data reading utilities
- `__init__.py` - Module exports

**Key Features**:
- Multiple chunking strategies:
  - `single_chunk`: For small dialogues
  - `split_input_keep_output`: Split question, keep answer intact
  - `keep_input_split_output`: Keep question, split answer
  - `split_both_fields`: Split both question and answer
- Token-aware text splitting
- Content validation and deduplication
- QCA (Question, Context, Answer) format support

**Usage**:
```python
from lab.preprocessing import chunk_medical_dialogues

# Chunk data with specific strategy
chunks = chunk_medical_dialogues(
    df, 
    target_chunk_size=512,
    chunking_strategy="adaptive"
)
```

### 4. Reranking (`reranking/`)
**Purpose**: Document reranking for improved retrieval accuracy

**Files**:
- `reranker.py` - Main reranking functionality
- `test_reranker.py` - Reranking tests and validation
- `__init__.py` - Module exports

**Key Features**:
- Support for BGE and Gemma reranking models
- Cross-encoder architecture for relevance scoring
- Batch processing capabilities
- Integration with retrieval pipeline

**Usage**:
```python
from lab.reranking import DocumentReranker

# Initialize reranker
reranker = DocumentReranker("BAAI/bge-reranker-v2-m3")

# Rerank documents
reranked_docs = reranker.rerank_documents(query, documents, top_k=10)
```

### 5. Retrieval (`retrieval/`)
**Purpose**: Document retrieval using vector similarity and HNSW indexing

**Files**:
- `retrieve.py` - Basic retrieval functionality
- `retrieval_with_rerank.py` - Advanced retrieval with reranking
- `__init__.py` - Module exports

**Key Features**:
- HNSW (Hierarchical Navigable Small World) vector indexing
- Cosine similarity search
- Integration with MongoDB for metadata retrieval
- Support for reranking integration

**Usage**:
```python
from lab.retrieval import retrieve_chunks, retrieve_with_reranking

# Basic retrieval
results = retrieve_chunks(query, top_k=10)

# Retrieval with reranking
results = retrieve_with_reranking(query, top_k=10, reranker=reranker)
```

### 6. Storage (`storage/`)
**Purpose**: Data storage, indexing, and persistence management

**Files**:
- `store_chunks.py` - MongoDB chunk storage
- `build_hnsw_index.py` - HNSW index construction
- `pre_store_validation.py` - Data validation before storage
- `__init__.py` - Module exports

**Key Features**:
- Batch MongoDB operations for efficiency
- HNSW index building and management
- Data validation and integrity checks
- Progress tracking for large datasets

**Usage**:
```python
from lab.storage import store_chunks_to_mongodb, build_hnsw_index

# Store chunks
store_chunks_to_mongodb(chunks, batch_size=1000)

# Build index
build_hnsw_index(collection_name="chunks")
```

## 🚀 Lab Environment Setup

### Prerequisites
- Python 3.8+
- MongoDB 4.4+
- CUDA-capable GPU (recommended)
- Hugging Face account with API token

### Installation

1. **Clone and navigate to lab directory**:
```bash
cd lab
```

2. **Install dependencies**:
```bash
pip install -r ../requirements.txt
```

3. **Set up environment variables**:
```bash
cp ../env.example .env
# Edit .env with your HF_TOKEN and MongoDB settings
```

4. **Initialize MongoDB**:
```bash
cd database
python create_schema.py
```

### Running Lab Components

#### 1. Data Preprocessing Pipeline
```bash
cd preprocessing
python -c "
from chunker import chunk_medical_dialogues
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Process and chunk
chunks = chunk_medical_dialogues(df, target_chunk_size=512)
print(f'Generated {len(chunks)} chunks')
"
```

#### 2. Embedding Generation
```bash
cd embedding
python -c "
from embedding import load_embed_model, embedding_text

# Load model
model, tokenizer = load_embed_model('MedAI-COS30018/MedEmbed-large-v0.1')

# Generate embeddings for chunks
texts = [chunk['content'] for chunk in chunks]
embeddings = embedding_text(texts, model, tokenizer)
print(f'Generated {len(embeddings)} embeddings')
"
```

#### 3. Storage and Indexing
```bash
cd storage
python -c "
from store_chunks import store_chunks_to_mongodb
from build_hnsw_index import build_hnsw_index

# Store chunks with embeddings
store_chunks_to_mongodb(chunks_with_embeddings)

# Build HNSW index
build_hnsw_index()
print('Index built successfully')
"
```

#### 4. Retrieval Testing
```bash
cd retrieval
python -c "
from retrieve import retrieve_chunks

# Test retrieval
query = 'What are the symptoms of diabetes?'
results = retrieve_chunks(query, top_k=5)

for i, result in enumerate(results):
    print(f'{i+1}. {result[\"content\"][:100]}...')
"
```

## 🔬 Research and Development

### Custom Chunking Strategies
The lab environment allows for easy experimentation with custom chunking strategies:

```python
from preprocessing.chunker import ChunkingStrategy

class CustomChunkingStrategy(ChunkingStrategy):
    def chunk(self, text, target_size):
        # Implement your custom chunking logic
        return custom_chunks
```

### Model Comparison
Test different embedding and reranking models:

```python
# Compare embedding models
models = [
    "MedAI-COS30018/MedEmbed-large-v0.1",
    "sentence-transformers/all-MiniLM-L6-v2"
]

for model_name in models:
    model, tokenizer = load_embed_model(model_name)
    # Run evaluation...
```

### Performance Optimization
The lab environment provides tools for performance analysis:

```python
import time
from lab.preprocessing import chunk_medical_dialogues

# Benchmark chunking performance
start_time = time.time()
chunks = chunk_medical_dialogues(large_dataset)
end_time = time.time()

print(f"Chunking time: {end_time - start_time:.2f} seconds")
print(f"Chunks per second: {len(chunks) / (end_time - start_time):.2f}")
```

## 📊 Integration with Production

The lab components are designed to be easily integrated into the production FastAPI application:

1. **Service Layer**: Lab functions are wrapped in service classes
2. **API Endpoints**: Lab functionality exposed via REST APIs
3. **Configuration**: Lab settings managed through environment variables
4. **Monitoring**: Lab operations monitored through application logs

### Migration Path
```python
# Lab usage
from lab.preprocessing import chunk_medical_dialogues

# Production usage
from app.services.preprocessing_service import PreprocessingService
preprocessing_service = PreprocessingService()
chunks = preprocessing_service.chunk_data(df)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in embedding generation
   - Use CPU fallback for large datasets

2. **MongoDB Connection Issues**:
   - Check MongoDB service status
   - Verify connection string in environment variables

3. **Model Download Failures**:
   - Verify HF_TOKEN is set correctly
   - Check internet connectivity
   - Use model caching for offline development

### Debug Mode
Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your lab operations will now show detailed logs
```

## 📈 Performance Benchmarks

Typical performance metrics for lab components:

- **Chunking**: ~1000 documents/second
- **Embedding**: ~100 documents/second (GPU), ~10 documents/second (CPU)
- **Storage**: ~5000 chunks/second (batch mode)
- **Retrieval**: ~100 queries/second (with HNSW index)

## 🤝 Contributing

When adding new features to lab components:

1. Maintain backward compatibility
2. Add comprehensive tests
3. Update documentation
4. Follow the existing code structure
5. Add performance benchmarks for new features

## 📚 Additional Resources

- [Hugging Face Models](https://huggingface.co/models)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [HNSW Algorithm Paper](https://arxiv.org/abs/1603.09320)
- [Medical NLP Best Practices](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6568068/)

---

**Note**: This lab environment is designed for research and development. For production deployment, use the integrated FastAPI application in the main directory.
