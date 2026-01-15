import logging
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json

from app.core.config import settings
from app.core.database import get_sync_database
from app.services.dataset import HuggingFaceDatasetService
from app.services.preprocessing import PreprocessingService
from app.services.storage import StorageService
from app.core.state import get_model_manager

logger = logging.getLogger(__name__)

class IngestionPipelineService:
    """Comprehensive ingestion pipeline service."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.hf_service = HuggingFaceDatasetService()
        self.storage_service = StorageService()
    
    async def run_full_ingestion_pipeline(self, dataset_name: str) -> Dict[str, Any]:
        """Run the complete ingestion pipeline for a dataset."""
        try:
            logger.info(f"Starting full ingestion pipeline for {dataset_name}")
            
            # Step 1: Crawl dataset from Hugging Face
            logger.info(f"Step 1: Crawling dataset {dataset_name}")
            crawl_result = await self.hf_service.crawl_dataset(dataset_name)
            
            if not crawl_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to crawl dataset: {crawl_result.get('error', 'Unknown error')}"
                }
            
            # Step 2: Parse and preprocess data
            logger.info(f"Step 2: Parsing and preprocessing {dataset_name}")
            processed_data = crawl_result["processed_data"]
            
            # Convert to DataFrame for preprocessing
            import pandas as pd
            df = pd.DataFrame(processed_data)
            
            # Step 3: Chunk data
            logger.info(f"Step 3: Chunking data for {dataset_name}")
            tokenizer, _, _, _ = get_model_manager().get_embedding_model()
            preprocessing_service = PreprocessingService(tokenizer)
            
            chunks = await preprocessing_service.chunk_medical_dialogues(df)
            
            # Step 4: Generate embeddings
            logger.info(f"Step 4: Generating embeddings for {dataset_name}")
            embeddings_result = await self._generate_embeddings_for_chunks(chunks)
            
            if not embeddings_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to generate embeddings: {embeddings_result.get('error', 'Unknown error')}"
                }
            
            chunks_with_embeddings = embeddings_result["chunks_with_embeddings"]
            
            # Step 5: Deduplication
            logger.info(f"Step 5: Deduplicating chunks for {dataset_name}")
            deduplicated_chunks = await self._deduplicate_chunks(chunks_with_embeddings)
            
            # Step 6: Store to MongoDB
            logger.info(f"Step 6: Storing chunks to MongoDB for {dataset_name}")
            storage_result = await self.storage_service.store_chunks(
                chunks=deduplicated_chunks,
                collection_name="chunks",
                batch_size=100
            )
            
            # Step 7: Build/Update HNSW index
            logger.info(f"Step 7: Building HNSW index for {dataset_name}")
            index_result = await self.storage_service.build_hnsw_index_async(force_rebuild=True)
            
            # Step 8: Update dataset status
            await self.hf_service._update_dataset_status(dataset_name, "processed", 100)
            
            logger.info(f"Successfully completed ingestion pipeline for {dataset_name}")
            
            return {
                "success": True,
                "dataset_name": dataset_name,
                "total_rows": len(processed_data),
                "total_chunks": len(deduplicated_chunks),
                "stored_chunks": storage_result["success_count"],
                "index_built": index_result["success"],
                "message": f"Successfully processed {len(processed_data)} rows into {len(deduplicated_chunks)} chunks"
            }
            
        except Exception as e:
            logger.error(f"Error in ingestion pipeline for {dataset_name}: {e}")
            await self.hf_service._update_dataset_status(dataset_name, "error", 0)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate embeddings for chunks."""
        try:
            # Get embedding model
            tokenizer, embed_model, device, embedding_dim = get_model_manager().get_embedding_model()
            
            # Run embedding generation in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._generate_embeddings_sync,
                chunks,
                tokenizer,
                embed_model,
                device,
                embedding_dim
            )
            
            return {
                "success": True,
                "chunks_with_embeddings": result
            }
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_embeddings_sync(self, chunks, tokenizer, embed_model, device, embedding_dim):
        """Synchronous embedding generation."""
        import torch
        import torch.nn.functional as F
        import numpy as np
        
        chunks_with_embeddings = []
        batch_size = settings.BATCH_SIZE
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_texts = [chunk['content'] for chunk in batch_chunks]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=settings.MAX_SEQUENCE_LENGTH,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = embed_model(**inputs)
                attention_mask = inputs['attention_mask']
                embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                embeddings = embeddings.cpu().numpy().astype(np.float64)
            
            # Add embeddings to chunks
            for j, chunk in enumerate(batch_chunks):
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding['embedding'] = embeddings[j].tolist()
                chunk_with_embedding['embedding_model'] = settings.EMBEDDING_MODEL_PATH
                chunk_with_embedding['embedding_dim'] = embedding_dim
                chunks_with_embeddings.append(chunk_with_embedding)
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return chunks_with_embeddings
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling function."""
        import torch
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * input_mask_expanded
        summed_embeddings = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = summed_embeddings / summed_mask
        return mean_embeddings
    
    async def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate chunks based on content hash."""
        try:
            seen_hashes = set()
            deduplicated_chunks = []
            
            for chunk in chunks:
                # Calculate content hash
                content_hash = self._calculate_content_hash(chunk['content'])
                
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    chunk['content_hash'] = content_hash
                    deduplicated_chunks.append(chunk)
                else:
                    logger.debug(f"Removed duplicate chunk: {chunk['metadata']['chunk_id']}")
            
            logger.info(f"Deduplication: {len(chunks)} -> {len(deduplicated_chunks)} chunks")
            
            return deduplicated_chunks
            
        except Exception as e:
            logger.error(f"Error in deduplication: {e}")
            return chunks  # Return original chunks if deduplication fails
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash for content deduplication."""
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get_ingestion_status(self, dataset_name: str) -> Dict[str, Any]:
        """Get ingestion status for a dataset."""
        try:
            db = get_sync_database()
            status_collection = db['dataset_status']
            status_doc = status_collection.find_one({"dataset_name": dataset_name})
            
            if not status_doc:
                return {
                    "dataset_name": dataset_name,
                    "status": "not_processed",
                    "progress": 0,
                    "last_updated": None
                }
            
            return {
                "dataset_name": dataset_name,
                "status": status_doc["status"],
                "progress": status_doc.get("processing_progress", 0),
                "last_updated": status_doc.get("last_updated"),
                "last_processed": status_doc.get("last_processed")
            }
            
        except Exception as e:
            logger.error(f"Error getting ingestion status: {e}")
            return {
                "dataset_name": dataset_name,
                "status": "error",
                "progress": 0,
                "error": str(e)
            }
    
    async def get_all_ingestion_statuses(self) -> List[Dict[str, Any]]:
        """Get ingestion status for all datasets."""
        dataset_names = list(self.hf_service.datasets.keys())
        tasks = [self.get_ingestion_status(name) for name in dataset_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        statuses = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error getting ingestion status: {result}")
                continue
            statuses.append(result)
        
        return statuses
    
    def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        self.hf_service.cleanup()
        self.storage_service.cleanup()
