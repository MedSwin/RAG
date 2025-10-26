import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timezone
import re
from concurrent.futures import ThreadPoolExecutor
import asyncio

from app.core.config import settings

logger = logging.getLogger(__name__)

class PreprocessingService:
    """Service for data preprocessing and chunking."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    async def chunk_medical_dialogues(
        self, 
        df: pd.DataFrame, 
        target_chunk_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Chunk medical dialogues asynchronously."""
        try:
            target_size = target_chunk_size or settings.TARGET_CHUNK_SIZE
            
            # Run chunking in thread pool
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(
                self.executor,
                self._chunk_medical_dialogues_sync,
                df,
                target_size
            )
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking medical dialogues: {e}")
            raise
    
    def _chunk_medical_dialogues_sync(self, df: pd.DataFrame, target_chunk_size: int) -> List[Dict[str, Any]]:
        """Synchronous chunking function."""
        # Import the original chunking functions
        from preprocessing.chunker import chunk_medical_dialogues
        
        # Process data
        processed_df = self._preprocessing_data(df)
        df_with_strategy = self._chunk_field_based_splitting(processed_df, target_chunk_size)
        
        # Create chunks using different strategies
        single_chunks = self._single_chunk(df_with_strategy)
        split_input_chunks = self._split_input_chunk(df_with_strategy, target_chunk_size)
        split_output_chunks = self._split_output_chunk(df_with_strategy, target_chunk_size)
        split_both_chunks = self._create_split_both_chunks(df_with_strategy, target_chunk_size)
        
        # Combine all chunks
        all_chunks = self._combine_all_chunks(
            single_chunks, 
            split_input_chunks, 
            split_output_chunks, 
            split_both_chunks
        )
        
        # Validate chunks
        final_chunks = self._validate_chunks(all_chunks)
        
        return final_chunks
    
    def _preprocessing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input data."""
        # Basic data cleaning
        df = df.copy()
        
        # Handle QCA format - preserve original fields for embedding
        if 'question' in df.columns and 'context' in df.columns and 'answer' in df.columns:
            # Keep original QCA fields for embedding
            df['question'] = df['question'].fillna('')
            df['context'] = df['context'].fillna('')
            df['answer'] = df['answer'].fillna('')
            
            # Create combined input/output for chunking compatibility
            df['input'] = df['question'] + '\n\nContext: ' + df['context']
            df['output'] = df['answer']
        else:
            # Fill missing values for traditional input/output format
            df['input'] = df['input'].fillna('')
            df['output'] = df['output'].fillna('')
        
        # Remove rows with empty input or output
        df = df[(df['input'].str.strip() != '') & (df['output'].str.strip() != '')]
        
        # Add required columns if they don't exist
        if 'id' not in df.columns:
            df['id'] = range(len(df))
        if 'source' not in df.columns:
            df['source'] = 'unknown'
        if 'task' not in df.columns:
            df['task'] = 'medical_dialogue'
        
        return df
    
    def _chunk_field_based_splitting(self, df: pd.DataFrame, target_chunk_size: int) -> pd.DataFrame:
        """Determine chunking strategy for each row."""
        strategies = []
        
        for _, row in df.iterrows():
            input_text = str(row.get('input', '')).strip()
            output_text = str(row.get('output', '')).strip()
            
            if not input_text or not output_text:
                strategies.append("skip")
                continue
            
            input_tokens = self._count_tokens(input_text)
            output_tokens = self._count_tokens(output_text)
            total_single_chunk_tokens = input_tokens + output_tokens + 20
            
            if total_single_chunk_tokens <= target_chunk_size:
                strategy = "single_chunk"
            elif input_tokens > target_chunk_size and output_tokens <= target_chunk_size:
                strategy = "split_input_keep_output"
            elif input_tokens <= target_chunk_size and output_tokens > target_chunk_size:
                strategy = "keep_input_split_output"
            else:
                strategy = "split_both_fields"
            
            strategies.append(strategy)
        
        df['chunking_strategy'] = strategies
        return df
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text, add_special_tokens=True))
    
    def _single_chunk(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create single chunks for rows that fit in one chunk."""
        chunks = []
        chunk_single = df[df["chunking_strategy"] == "single_chunk"]
        
        for _, row in chunk_single.iterrows():
            # Use raw text for embedding (QCA format if available)
            if 'question' in row and 'context' in row and 'answer' in row:
                # For QCA format, combine question + context + answer for embedding
                content_for_embedding = f"{row['question']}\n\n{row['context']}\n\n{row['answer']}"
                
                # Store QCA separately for retrieval
                qca_data = {
                    "question": row['question'],
                    "context": row['context'],
                    "answer": row['answer']
                }
            else:
                # Traditional input/output format
                content_for_embedding = f"{row['input']}\n\n{row['output']}"
                qca_data = None
            
            if not content_for_embedding.strip():
                continue
            
            chunk_metadata = {
                "chunk_id": f"{row['source']}_{row['id']}_single",
                "parent_id": str(row["id"]),
                "source": str(row["source"]),
                "sequence": 1,
                "total_chunks": 1,
                "content_type": "complete_dialogue",
                "task": str(row["task"]),
                "chunk_length": len(content_for_embedding),
                "token_count": self._count_tokens(content_for_embedding),
                "created_timestamp": datetime.now(timezone.utc),
                "related_chunks": []
            }
            
            # Add QCA data to metadata if available
            if qca_data:
                chunk_metadata.update(qca_data)
            
            chunks.append({
                "content": content_for_embedding,
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def _split_input_chunk(self, df: pd.DataFrame, target_chunk_size: int) -> List[Dict[str, Any]]:
        """Create chunks by splitting input text."""
        chunks = []
        chunks_split_input = df[df["chunking_strategy"] == "split_input_keep_output"]
        
        for _, row in chunks_split_input.iterrows():
            # Handle QCA format
            if 'question' in row and 'context' in row and 'answer' in row:
                question_text = str(row['question']).strip()
                context_text = str(row['context']).strip()
                answer_text = str(row['answer']).strip()
                
                if not question_text or not answer_text:
                    continue
                
                # Split question + context for embedding
                input_text = f"{question_text}\n\n{context_text}"
                input_parts = self._split_text_by_token(input_text, max_tokens=target_chunk_size//2)
                
                qca_data = {
                    "question": question_text,
                    "context": context_text,
                    "answer": answer_text
                }
            else:
                # Traditional format
                input_text = str(row.get('input', '')).strip()
                output_text = str(row.get('output', '')).strip()
                
                if not input_text or not output_text:
                    continue
                
                input_parts = self._split_text_by_token(input_text, max_tokens=target_chunk_size//2)
                qca_data = None
            
            if not input_parts:
                continue
            
            total_chunks = len(input_parts)
            for i, input_part in enumerate(input_parts, 1):
                if i == total_chunks:
                    # Include answer in final chunk
                    if qca_data:
                        content_for_embedding = f"{input_part}\n\n{qca_data['answer']}"
                    else:
                        content_for_embedding = f"{input_part}\n\n{output_text}"
                    content_type = "question_final_and_answer"
                else:
                    content_for_embedding = input_part
                    content_type = f"question_part{i}"
                
                if not content_for_embedding.strip():
                    continue
                
                chunk_metadata = {
                    "chunk_id": f"{row['source']}_{row['id']}_{i:02d}",
                    "parent_id": str(row["id"]),
                    "sequence": i,
                    "total_chunks": total_chunks,
                    "content_type": content_type,
                    "source": str(row["source"]),
                    "task": str(row["task"]),
                    "related_chunks": [f"{row['source']}_{row['id']}_{j:02d}" for j in range(1, total_chunks+1) if j != i],
                    "chunk_length": len(content_for_embedding),
                    "token_count": self._count_tokens(content_for_embedding),
                    "created_timestamp": datetime.now(timezone.utc)
                }
                
                # Add QCA data to metadata if available
                if qca_data:
                    chunk_metadata.update(qca_data)
                
                chunks.append({"content": content_for_embedding, "metadata": chunk_metadata})
        
        return chunks
    
    def _split_output_chunk(self, df: pd.DataFrame, target_chunk_size: int) -> List[Dict[str, Any]]:
        """Create chunks by splitting output text."""
        chunks = []
        chunks_split_output = df[df["chunking_strategy"] == "keep_input_split_output"]
        
        for _, row in chunks_split_output.iterrows():
            # Handle QCA format
            if 'question' in row and 'context' in row and 'answer' in row:
                question_text = str(row['question']).strip()
                context_text = str(row['context']).strip()
                answer_text = str(row['answer']).strip()
                
                if not question_text or not answer_text:
                    continue
                
                # Split answer for embedding
                answer_parts = self._split_text_by_token(answer_text, max_tokens=target_chunk_size//2)
                
                qca_data = {
                    "question": question_text,
                    "context": context_text,
                    "answer": answer_text
                }
                
                # Use question + context as input reference
                input_reference = f"{question_text}\n\n{context_text}"
            else:
                # Traditional format
                input_text = str(row.get('input', '')).strip()
                output_text = str(row.get('output', '')).strip()
                
                if not input_text or not output_text:
                    continue
                
                answer_parts = self._split_text_by_token(output_text, max_tokens=target_chunk_size//2)
                qca_data = None
                input_reference = input_text
            
            if not answer_parts:
                continue
            
            total_chunks = len(answer_parts)
            
            for i, answer_part in enumerate(answer_parts, 1):
                if i == 1:
                    # First chunk includes question + context + first part of answer
                    content_for_embedding = f"{input_reference}\n\n{answer_part}"
                else:
                    # Subsequent chunks include reference + answer part
                    input_ref = self._get_first_words(input_reference, 15)
                    content_for_embedding = f"{input_ref}\n\n{answer_part}"
                
                if not content_for_embedding.strip():
                    continue
                
                chunk_metadata = {
                    "chunk_id": f"{row['source']}_{row['id']}_{i:02d}",
                    "parent_id": str(row["id"]),
                    "sequence": i,
                    "total_chunks": total_chunks,
                    "content_type": "question_and_answer_part1" if i == 1 else "answer_continuation",
                    "source": str(row["source"]),
                    "task": str(row["task"]),
                    "related_chunks": [f"{row['source']}_{row['id']}_{j:02d}" for j in range(1, total_chunks+1) if j != i],
                    "chunk_length": len(content_for_embedding),
                    "token_count": self._count_tokens(content_for_embedding),
                    "created_timestamp": datetime.now(timezone.utc)
                }
                
                # Add QCA data to metadata if available
                if qca_data:
                    chunk_metadata.update(qca_data)
                
                chunks.append({"content": content_for_embedding, "metadata": chunk_metadata})
        
        return chunks
    
    def _create_split_both_chunks(self, df: pd.DataFrame, target_chunk_size: int) -> List[Dict[str, Any]]:
        """Create chunks by splitting both input and output text."""
        chunks = []
        split_both_fields = df[df["chunking_strategy"] == "split_both_fields"]
        
        for _, row in split_both_fields.iterrows():
            # Handle QCA format
            if 'question' in row and 'context' in row and 'answer' in row:
                question_text = str(row['question']).strip()
                context_text = str(row['context']).strip()
                answer_text = str(row['answer']).strip()
                
                if not question_text or not answer_text:
                    continue
                
                # Split both question+context and answer
                input_text = f"{question_text}\n\n{context_text}"
                input_parts = self._split_text_by_token(input_text, max_tokens=target_chunk_size//3)
                output_parts = self._split_text_by_token(answer_text, max_tokens=target_chunk_size//3)
                
                qca_data = {
                    "question": question_text,
                    "context": context_text,
                    "answer": answer_text
                }
                
                patient_summary = self._get_first_words(input_text, 15)
            else:
                # Traditional format
                input_text = str(row.get('input', '')).strip()
                output_text = str(row.get('output', '')).strip()
                
                if not input_text or not output_text:
                    continue
                
                input_parts = self._split_text_by_token(input_text, max_tokens=target_chunk_size//3)
                output_parts = self._split_text_by_token(output_text, max_tokens=target_chunk_size//3)
                qca_data = None
                patient_summary = self._get_first_words(input_text, 15)
            
            if not input_parts or not output_parts:
                continue
            
            total_chunks = len(input_parts) + len(output_parts)
            chunk_sequence = 1
            
            # Process input parts
            for i, input_part in enumerate(input_parts):
                content_for_embedding = input_part
                
                chunk_metadata = {
                    "chunk_id": f"{row['source']}_{row['id']}_{chunk_sequence:02d}",
                    "parent_id": str(row["id"]),
                    "sequence": chunk_sequence,
                    "total_chunks": total_chunks,
                    "content_type": f"question_part{i+1}",
                    "source": str(row["source"]),
                    "task": str(row["task"]),
                    "related_chunks": [f"{row['source']}_{row['id']}_{j:02d}" for j in range(1, total_chunks+1) if j != chunk_sequence],
                    "chunk_length": len(content_for_embedding),
                    "token_count": self._count_tokens(content_for_embedding),
                    "created_timestamp": datetime.now(timezone.utc)
                }
                
                # Add QCA data to metadata if available
                if qca_data:
                    chunk_metadata.update(qca_data)
                
                chunks.append({"content": content_for_embedding, "metadata": chunk_metadata})
                chunk_sequence += 1
            
            # Process output parts
            for i, output_part in enumerate(output_parts):
                content_for_embedding = f"{patient_summary}\n\n{output_part}"
                
                chunk_metadata = {
                    "chunk_id": f"{row['source']}_{row['id']}_{chunk_sequence:02d}",
                    "parent_id": str(row["id"]),
                    "sequence": chunk_sequence,
                    "total_chunks": total_chunks,
                    "content_type": f"answer_part{i+1}",
                    "source": str(row["source"]),
                    "task": str(row["task"]),
                    "related_chunks": [f"{row['source']}_{row['id']}_{j:02d}" for j in range(1, total_chunks+1) if j != chunk_sequence],
                    "chunk_length": len(content_for_embedding),
                    "token_count": self._count_tokens(content_for_embedding),
                    "created_timestamp": datetime.now(timezone.utc)
                }
                
                # Add QCA data to metadata if available
                if qca_data:
                    chunk_metadata.update(qca_data)
                
                chunks.append({"content": content_for_embedding, "metadata": chunk_metadata})
                chunk_sequence += 1
        
        return chunks
    
    def _combine_all_chunks(self, single_chunks, split_input_chunks, split_output_chunks, split_both_chunks) -> List[Dict[str, Any]]:
        """Combine all chunks from different strategies."""
        all_chunks = []
        all_chunks.extend(single_chunks)
        all_chunks.extend(split_input_chunks)
        all_chunks.extend(split_output_chunks)
        all_chunks.extend(split_both_chunks)
        
        logger.info(f"Chunk Summaries:")
        logger.info(f"Single chunks: {len(single_chunks)}")
        logger.info(f"Split input chunks: {len(split_input_chunks)}")
        logger.info(f"Split output chunks: {len(split_output_chunks)}")
        logger.info(f"Split both chunks: {len(split_both_chunks)}")
        logger.info(f"Total chunks: {len(all_chunks)}")
        
        return all_chunks
    
    def _validate_chunks(self, all_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate chunks for quality and consistency."""
        valid_chunks = []
        
        for chunk in all_chunks:
            if not chunk.get('content', '').strip():
                continue
            
            # Check token count
            token_count = chunk['metadata'].get('token_count', 0)
            if token_count > settings.MAX_SEQUENCE_LENGTH:
                logger.warning(f"Chunk {chunk['metadata']['chunk_id']} exceeds token limit: {token_count}")
                continue
            
            valid_chunks.append(chunk)
        
        return valid_chunks
    
    def _split_text_by_token(self, text: str, max_tokens: int) -> List[str]:
        """Split text by token count."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + (". " if current_chunk else "") + sentence + "."
            if self._count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = sentence + "."
                else:
                    chunks.extend(self._force_split_by_words(sentence, max_tokens))
                    current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return [c for c in chunks if c.strip()]
    
    def _force_split_by_words(self, text: str, max_tokens: int) -> List[str]:
        """Force split text by words when sentence splitting fails."""
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            test_chunk = current_chunk + (" " if current_chunk else "") + word
            if self._count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk + "...")
                    current_chunk = "..." + word
                else:
                    chunks.append(word)
                    current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return [c for c in chunks if c.strip()]
    
    def _create_safe_chunk_content(self, content_parts: Dict[str, str]) -> str:
        """Create safe chunk content from parts."""
        base_parts = []
        
        for key, value in content_parts.items():
            if value and str(value).strip():
                base_parts.append(f"{key}: {value}")
        
        content = "\n\n".join(base_parts)
        return content.strip()
    
    def _get_first_words(self, text: str, num_words: int) -> str:
        """Get first N words from text."""
        words = str(text).split()
        if len(words) <= num_words:
            return text
        return ' '.join(words[:num_words]) + '...'
    
    async def validate_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate chunks and return validation results."""
        valid_chunks = []
        invalid_chunks = []
        errors = []
        
        for i, chunk in enumerate(chunks):
            chunk_errors = []
            
            # Check required fields
            if not chunk.get('content', '').strip():
                chunk_errors.append("Empty content")
            
            if 'metadata' not in chunk:
                chunk_errors.append("Missing metadata")
            else:
                metadata = chunk['metadata']
                required_fields = ['chunk_id', 'parent_id', 'source', 'task', 'sequence', 'total_chunks', 'content_type']
                for field in required_fields:
                    if field not in metadata:
                        chunk_errors.append(f"Missing metadata field: {field}")
            
            # Check token count
            if 'metadata' in chunk:
                token_count = chunk['metadata'].get('token_count', 0)
                if token_count > settings.MAX_SEQUENCE_LENGTH:
                    chunk_errors.append(f"Token count exceeds limit: {token_count}")
            
            if chunk_errors:
                invalid_chunks.append(chunk)
                errors.extend([f"Chunk {i}: {error}" for error in chunk_errors])
            else:
                valid_chunks.append(chunk)
        
        # Calculate statistics
        statistics = {
            "total_chunks": len(chunks),
            "valid_chunks": len(valid_chunks),
            "invalid_chunks": len(invalid_chunks),
            "validation_errors": len(errors)
        }
        
        return {
            "valid_chunks": valid_chunks,
            "invalid_chunks": invalid_chunks,
            "errors": errors,
            "statistics": statistics
        }
