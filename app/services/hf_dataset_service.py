import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import re

from app.core.config import settings
from app.core.database import get_sync_database

logger = logging.getLogger(__name__)

# Global cache for dataset statistics
_dataset_stats_cache = None
_cache_timestamp = None

class HuggingFaceDatasetService:
    """Service for crawling and processing Hugging Face datasets."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.hf_token = os.getenv('HF_TOKEN')
        
        # Dataset configurations
        self.datasets = {
            "HealthCareMagic": {
                "repo_id": "MedAI-COS30018/HealthCareMagic",
                "description": "~112.165 real conversations between patients and doctors from HealthCareMagic.com",
                "url": "https://huggingface.co/datasets/MedAI-COS30018/HealthCareMagic",
                "expected_rows": 112165,
                "status": "not_processed"
            },
            "iCliniq": {
                "repo_id": "MedAI-COS30018/iCliniq", 
                "description": "~11k collection including lavita/ChatDoctor-iCliniq",
                "url": "https://huggingface.co/datasets/MedAI-COS30018/iCliniq",
                "expected_rows": 11000,
                "status": "not_processed"
            },
            "PubMedQA-u": {
                "repo_id": "MedAI-COS30018/PubMedQA-u-RAG",
                "description": "61.160 unlabelled PubMedQA questions, derived with Knowledge Distillation from MedGemma-27b-it model",
                "url": "https://huggingface.co/datasets/MedAI-COS30018/PubMedQA-u-RAG", 
                "expected_rows": 61160,
                "status": "not_processed"
            },
            "PubMedQA-l": {
                "repo_id": "MedAI-COS30018/PubMedQA-l-RAG",
                "description": "998 specialist labelled PubMedQA topics",
                "url": "https://huggingface.co/datasets/MedAI-COS30018/PubMedQA-l-RAG",
                "expected_rows": 998,
                "status": "not_processed"
            },
            "PubMedQA-map": {
                "repo_id": "MedAI-COS30018/PubMedQA-map",
                "description": "286,519 data sources, both unlabelled, labeled and synthesis from PubMedQA sources",
                "url": "https://huggingface.co/datasets/MedAI-COS30018/PubMedQA-map",
                "expected_rows": 286519,
                "status": "not_processed"
            }
        }
    
    async def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a specific dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_config = self.datasets[dataset_name]
        
        # Try to get dataset info from Hugging Face first
        dataset_info = None
        try:
            # Load dataset info without downloading
            loop = asyncio.get_event_loop()
            dataset_info = await loop.run_in_executor(
                self.executor,
                self._get_dataset_info_sync,
                dataset_config["repo_id"]
            )
        except Exception as e:
            logger.error(f"Error loading dataset info from Hugging Face for {dataset_name}: {e}")
            # Continue to return basic info even if HF loading fails
        
        # Try to get processing status from database (but don't fail if DB isn't connected)
        current_status = "not_processed"
        last_processed = None
        processing_progress = 0
        try:
            db = get_sync_database()
            if db is not None:
                status_collection = db['dataset_status']
                status_doc = status_collection.find_one({"dataset_name": dataset_name})
                if status_doc:
                    current_status = status_doc.get("status", "not_processed")
                    last_processed = status_doc.get("last_processed")
                    processing_progress = status_doc.get("processing_progress", 0)
        except Exception as e:
            logger.warning(f"Could not get dataset status from database for {dataset_name}: {e}")
            # Continue with default status
        
        # Return dataset info
        if dataset_info:
            return {
                "name": dataset_name,
                "description": dataset_config["description"],
                "url": dataset_config["url"],
                "repo_id": dataset_config["repo_id"],
                "expected_rows": dataset_config["expected_rows"],
                "actual_rows": dataset_info["num_rows"],
                "size_gb": dataset_info["size_gb"],
                "status": current_status,
                "last_processed": last_processed,
                "processing_progress": processing_progress
            }
        else:
            # Return info with default values if HF loading failed
            return {
                "name": dataset_name,
                "description": dataset_config["description"],
                "url": dataset_config["url"],
                "repo_id": dataset_config["repo_id"],
                "expected_rows": dataset_config["expected_rows"],
                "actual_rows": 0,
                "size_gb": 0,
                "status": current_status if current_status != "not_processed" else "error",
                "last_processed": last_processed,
                "processing_progress": processing_progress
            }
    
    def _get_dataset_info_sync(self, repo_id: str) -> Dict[str, Any]:
        """Synchronous function to get dataset info."""
        try:
            # Try to load dataset info using dataset builder (faster, metadata only)
            from datasets import load_dataset_builder
            builder = load_dataset_builder(repo_id, token=self.hf_token)
            
            # Get dataset info from builder
            if hasattr(builder.info, 'splits') and builder.info.splits:
                total_rows = sum(split.num_examples for split in builder.info.splits.values())
                # Estimate size from config if available
                if hasattr(builder.info, 'dataset_size') and builder.info.dataset_size:
                    size_bytes = builder.info.dataset_size
                    size_gb = size_bytes / (1024 ** 3)
                else:
                    size_gb = total_rows * 0.001  # Rough estimate
                
                return {
                    "num_rows": total_rows,
                    "size_gb": round(size_gb, 2),
                    "splits": list(builder.info.splits.keys())
                }
            
            # If builder doesn't have splits info, return defaults
            logger.warning(f"Dataset builder for {repo_id} doesn't have splits info, using defaults")
            return {
                "num_rows": 0,
                "size_gb": 0,
                "splits": []
            }
            
        except Exception as e:
            logger.error(f"Error loading dataset info for {repo_id}: {e}")
            # Return defaults instead of raising exception
            return {
                "num_rows": 0,
                "size_gb": 0,
                "splits": []
            }
    
    async def crawl_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Crawl and download a dataset from Hugging Face."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_config = self.datasets[dataset_name]
        
        try:
            # Update status to processing
            await self._update_dataset_status(dataset_name, "processing", 0)
            
            # Download dataset
            loop = asyncio.get_event_loop()
            dataset_data = await loop.run_in_executor(
                self.executor,
                self._download_dataset_sync,
                dataset_config["repo_id"]
            )
            
            # Process data into QCA format
            processed_data = await self._process_dataset_to_qca(dataset_data, dataset_name)
            
            # Update status to processed
            await self._update_dataset_status(dataset_name, "processed", 100)
            
            return {
                "success": True,
                "dataset_name": dataset_name,
                "total_rows": len(processed_data),
                "processed_data": processed_data[:100],  # Return first 100 rows as sample
                "message": f"Successfully crawled {len(processed_data)} rows from {dataset_name}"
            }
            
        except Exception as e:
            logger.error(f"Error crawling dataset {dataset_name}: {e}")
            await self._update_dataset_status(dataset_name, "error", 0)
            return {
                "success": False,
                "dataset_name": dataset_name,
                "error": str(e)
            }
    
    def _download_dataset_sync(self, repo_id: str) -> List[Dict[str, Any]]:
        """Synchronous function to download dataset."""
        try:
            dataset = load_dataset(repo_id, token=self.hf_token)
            
            # Combine all splits into one list
            all_data = []
            for split_name, split_data in dataset.items():
                for row in split_data:
                    row_dict = dict(row)
                    row_dict['_split'] = split_name
                    all_data.append(row_dict)
            
            return all_data
            
        except Exception as e:
            logger.error(f"Error downloading dataset {repo_id}: {e}")
            raise
    
    async def _process_dataset_to_qca(self, raw_data: List[Dict[str, Any]], dataset_name: str) -> List[Dict[str, Any]]:
        """Process raw dataset data into QCA (Question, Context, Answer) format."""
        processed_data = []
        
        for i, row in enumerate(raw_data):
            try:
                # Extract QCA from input/output fields
                qca_data = self._extract_qca_from_row(row, dataset_name)
                
                if qca_data:
                    # Add metadata
                    qca_data.update({
                        "id": f"{dataset_name}_{i}",
                        "source": dataset_name,
                        "original_row": i,
                        "created_timestamp": datetime.now(timezone.utc),
                        "content_hash": self._calculate_content_hash(qca_data)
                    })
                    
                    processed_data.append(qca_data)
                
                # Update progress every 1000 rows
                if i % 1000 == 0:
                    progress = min(90, (i / len(raw_data)) * 90)  # Reserve 10% for final processing
                    await self._update_dataset_status(dataset_name, "processing", progress)
                
            except Exception as e:
                logger.warning(f"Error processing row {i} from {dataset_name}: {e}")
                continue
        
        return processed_data
    
    def _extract_qca_from_row(self, row: Dict[str, Any], dataset_name: str) -> Optional[Dict[str, Any]]:
        """Extract Question, Context, Answer from a dataset row."""
        try:
            # Handle different dataset formats
            if dataset_name in ["HealthCareMagic", "iCliniq"]:
                return self._extract_qca_from_dialogue(row)
            elif dataset_name.startswith("PubMedQA"):
                return self._extract_qca_from_pubmed(row)
            else:
                return self._extract_qca_generic(row)
                
        except Exception as e:
            logger.warning(f"Error extracting QCA from row: {e}")
            return None
    
    def _extract_qca_from_dialogue(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract QCA from dialogue datasets."""
        try:
            # Look for input/output or question/answer fields
            input_text = row.get('input', '') or row.get('question', '') or row.get('sft', {}).get('input', '')
            output_text = row.get('output', '') or row.get('answer', '') or row.get('sft', {}).get('output', '')
            
            if not input_text or not output_text:
                return None
            
            # Parse input to extract question and context
            question, context = self._parse_input_field(input_text)
            
            return {
                "question": question,
                "context": context,
                "answer": output_text
            }
            
        except Exception as e:
            logger.warning(f"Error extracting QCA from dialogue: {e}")
            return None
    
    def _extract_qca_from_pubmed(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract QCA from PubMed datasets."""
        try:
            # PubMed datasets might have different field names
            question = row.get('question', '') or row.get('input', '')
            context = row.get('context', '') or row.get('long_answer', '')
            answer = row.get('answer', '') or row.get('output', '') or row.get('final_decision', '')
            
            if not question:
                return None
            
            return {
                "question": question,
                "context": context or "No additional context provided",
                "answer": answer or "No answer provided"
            }
            
        except Exception as e:
            logger.warning(f"Error extracting QCA from PubMed: {e}")
            return None
    
    def _extract_qca_generic(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generic QCA extraction for unknown formats."""
        try:
            # Try common field names
            question = row.get('question', '') or row.get('input', '') or row.get('query', '')
            context = row.get('context', '') or row.get('passage', '') or row.get('text', '')
            answer = row.get('answer', '') or row.get('output', '') or row.get('response', '')
            
            if not question:
                return None
            
            return {
                "question": question,
                "context": context or "No additional context provided",
                "answer": answer or "No answer provided"
            }
            
        except Exception as e:
            logger.warning(f"Error extracting QCA generically: {e}")
            return None
    
    def _parse_input_field(self, input_text: str) -> Tuple[str, str]:
        """Parse input field to extract question and context."""
        # Look for "Question:" and "Context:" patterns
        question_pattern = r"Question:\s*(.*?)(?=Context:|$)"
        context_pattern = r"Context:\s*(.*?)(?=Question:|$)"
        
        question_match = re.search(question_pattern, input_text, re.DOTALL | re.IGNORECASE)
        context_match = re.search(context_pattern, input_text, re.DOTALL | re.IGNORECASE)
        
        question = question_match.group(1).strip() if question_match else input_text.strip()
        context = context_match.group(1).strip() if context_match else "No additional context provided"
        
        return question, context
    
    def _calculate_content_hash(self, data: Dict[str, Any]) -> str:
        """Calculate hash for content deduplication."""
        content_str = f"{data['question']}|{data['context']}|{data['answer']}"
        return hashlib.md5(content_str.encode()).hexdigest()
    
    async def _update_dataset_status(self, dataset_name: str, status: str, progress: float):
        """Update dataset processing status in database."""
        try:
            db = get_sync_database()
            status_collection = db['dataset_status']
            
            status_doc = {
                "dataset_name": dataset_name,
                "status": status,
                "processing_progress": progress,
                "last_updated": datetime.now(timezone.utc)
            }
            
            if status == "processed":
                status_doc["last_processed"] = datetime.now(timezone.utc)
            
            status_collection.replace_one(
                {"dataset_name": dataset_name},
                status_doc,
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Error updating dataset status: {e}")
    
    async def get_all_datasets_info(self) -> List[Dict[str, Any]]:
        """Get information about all datasets."""
        tasks = []
        for dataset_name in self.datasets.keys():
            tasks.append(self.get_dataset_info(dataset_name))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        dataset_infos = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error getting dataset info: {result}")
                continue
            dataset_infos.append(result)
        
        return dataset_infos
    
    async def get_total_statistics(self, use_cache: bool = True) -> Dict[str, Any]:
        """Get total statistics across all datasets."""
        global _dataset_stats_cache, _cache_timestamp
        
        # Check cache first (valid for 10 minutes)
        if use_cache and _dataset_stats_cache is not None and _cache_timestamp is not None:
            import time
            if time.time() - _cache_timestamp < 600:  # 10 minute cache
                logger.info("Returning cached dataset statistics")
                return _dataset_stats_cache
        
        # Load fresh data
        dataset_infos = await self.get_all_datasets_info()
        
        total_rows = sum(info.get("actual_rows", 0) for info in dataset_infos)
        total_size_gb = sum(info.get("size_gb", 0) for info in dataset_infos)
        
        # Count processing statuses
        status_counts = {}
        for info in dataset_infos:
            status = info.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        result = {
            "total_datasets": len(dataset_infos),
            "total_rows": total_rows,
            "total_size_gb": round(total_size_gb, 2),
            "status_counts": status_counts,
            "datasets": dataset_infos
        }
        
        # Cache the result
        import time
        _dataset_stats_cache = result
        _cache_timestamp = time.time()
        
        return result
    
    @staticmethod
    def clear_cache():
        """Clear the dataset statistics cache."""
        global _dataset_stats_cache, _cache_timestamp
        _dataset_stats_cache = None
        _cache_timestamp = None
    
    def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
