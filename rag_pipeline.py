import pandas as pd
from pathlib import Path
from pymongo import MongoClient
import logging
from preprocessing import chunk_medical_dialogues
from embedding import load_embed_model, embedding_text, add_embeddings_to_chunks
from storage import pre_store_validate, store_chunks_to_mongodb, build_hnsw_index
from preprocessing import DocumentIngestionManager
import argparse

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="RAG pipeline")
    parser.add_argument("--data-dir", type=str, default="/fred/oz446/HenryNguyen/data/", 
                       help="Data directory containing CSV files")
    parser.add_argument("--embed-model-path", type=str, 
                       default="/fred/oz446/HenryNguyen/EmbeddingModel/PubMedBERT-MNLI-MedNLI",
                       help="Path to the model directory")
    parser.add_argument("--index-path", type=str,
                       help="Path to save the HNSW index file")
    
    args = parser.parse_args()
    
    # Set default index path if not provided
    if args.index_path is None:
        args.index_path = str(Path(args.data_dir) / "hnsw_index.bin")
    
    return args

def run_pipeline(data_dir: str, model_path: str, index_path: str):
    # Load embedding model
    tokenizer, embed_model, device = load_embed_model(Path(model_path))
    logger.info(f"Loaded embedding model on {device}")
    
    # Use DocumentIngestionManager for incremental loading
    logger.info("Ingesting data with DocumentIngestionManager...")
    document_manager = DocumentIngestionManager(data_dir)
    df = document_manager.incremental_load(data_dir)
    
    if df is None or df.empty:
        logger.error("No data loaded from DocumentIngestionManager")
        return
    
    logger.info(f"Loaded {len(df)} rows using incremental load")
    
    # Chunk data
    chunks = chunk_medical_dialogues(df, tokenizer)
    if not chunks:
        logger.error("No valid chunks generated")
        return
    
    logger.info(f"Total chunks generated: {len(chunks)}")
    
    # Generate embeddings
    embeddings, valid_indices = embedding_text(chunks, tokenizer, embed_model, device, batch_size=64)
    if embeddings.shape[0] == 0:
        logger.error("No valid embeddings generated")
        return
    
    # Add embeddings to chunks
    chunks_with_embeddings = add_embeddings_to_chunks(chunks, embeddings, valid_indices)
    if not chunks_with_embeddings:
        logger.error("No chunks with valid embeddings")
        return
    
    # Validate chunks
    validated_chunks = pre_store_validate(chunks_with_embeddings)
    if not validated_chunks:
        logger.error("No chunks passed validation")
        return
    
    # Store to MongoDB
    success_count = store_chunks_to_mongodb(validated_chunks)
    logger.info(f"Stored {success_count}/{len(validated_chunks)} chunks to MongoDB")
    
    # Build HNSW index
    build_hnsw_index(index_path=index_path)
    
    # Verify storage
    client = MongoClient('mongodb://localhost:27017/')
    db = client['medical_rag_db']
    coll = db['chunks']
    stored_count = coll.count_documents({})
    logger.info(f"Total documents in MongoDB: {stored_count}")
    client.close()

def main():
    logger.info("Starting pipeline...")
    args = parse_args()
    logger.info(f"Args: {args}")
    
    # Run the pipeline with parsed arguments
    run_pipeline(
        data_dir=args.data_dir,
        model_path=args.embed_model_path,
        index_path=args.index_path
    )
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()