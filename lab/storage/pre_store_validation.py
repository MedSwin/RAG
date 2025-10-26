from typing import List
from datetime import datetime, timezone
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pre_store_validate(chunks_with_embeddings: List[dict]) -> List[dict]:
    required_top = ['chunk_id', 'content', 'embedding', 'metadata', 'embedding_model', 'embedding_dim']
    required_meta = ['parent_id', 'source', 'task', 'sequence', 'total_chunks', 'content_type', 'related_chunks', 'chunk_length', 'created_timestamp']
    
    valid_chunks = []
    for chunk in chunks_with_embeddings:
        chunk_id = chunk.get('chunk_id', 'unknown')
        
        # Top-level validation
        if not all(key in chunk for key in required_top):
            logger.error(f"Missing top-level fields in chunk {chunk_id}: {set(required_top) - set(chunk)}")
            continue
        if not isinstance(chunk['chunk_id'], str) or not chunk['chunk_id'].strip():
            logger.error(f"Invalid chunk_id in chunk {chunk_id}: {chunk['chunk_id']}")
            continue
        if not isinstance(chunk['content'], str) or not chunk['content'].strip():
            logger.error(f"Invalid content in chunk {chunk_id}: {chunk['content'][:50]}...")
            continue
        if not isinstance(chunk['embedding'], list) or len(chunk['embedding']) != chunk.get('embedding_dim', 768):
            logger.error(f"Invalid embedding in chunk {chunk_id}: len={len(chunk['embedding']) if isinstance(chunk['embedding'], list) else 'not list'}, expected={chunk.get('embedding_dim', 768)}")
            continue
        if not all(isinstance(x, float) and np.isfinite(x) for x in chunk['embedding']):
            logger.error(f"Non-finite values in embedding for chunk {chunk_id}")
            continue
        if not isinstance(chunk['metadata'], dict):
            logger.error(f"Invalid metadata in chunk {chunk_id}: type={type(chunk['metadata'])}")
            continue
        if not isinstance(chunk['embedding_model'], str):
            logger.error(f"Invalid embedding_model in chunk {chunk_id}: {chunk['embedding_model']}")
            continue
        if not isinstance(chunk['embedding_dim'], int) or chunk['embedding_dim'] < 1:
            logger.error(f"Invalid embedding_dim in chunk {chunk_id}: {chunk['embedding_dim']}")
            continue
        
        # Metadata validation
        meta = chunk['metadata']
        if not all(key in meta for key in required_meta):
            logger.error(f"Missing metadata fields in chunk {chunk_id}: {set(required_meta) - set(meta)}")
            continue
        if not isinstance(meta['parent_id'], str) or not meta['parent_id'].strip():
            logger.error(f"Invalid parent_id in chunk {chunk_id}: {meta['parent_id']}")
            continue
        if not isinstance(meta['source'], str) or not meta['source'].strip():
            logger.error(f"Invalid source in chunk {chunk_id}: {meta['source']}")
            continue
        if not isinstance(meta['task'], str) or not meta['task'].strip():
            logger.error(f"Invalid task in chunk {chunk_id}: {meta['task']}")
            continue
        if not isinstance(meta['sequence'], int) or meta['sequence'] < 1:
            logger.error(f"Invalid sequence in chunk {chunk_id}: {meta['sequence']}")
            continue
        if not isinstance(meta['total_chunks'], int) or meta['total_chunks'] < 1:
            logger.error(f"Invalid total_chunks in chunk {chunk_id}: {meta['total_chunks']}")
            continue
        if not isinstance(meta['content_type'], str) or not meta['content_type'].strip():
            logger.error(f"Invalid content_type in chunk {chunk_id}: {meta['content_type']}")
            continue
        if not isinstance(meta['related_chunks'], list) or not all(isinstance(c, str) for c in meta['related_chunks']):
            logger.error(f"Invalid related_chunks in chunk {chunk_id}: {meta['related_chunks']}")
            continue
        if not isinstance(meta['chunk_length'], int) or meta['chunk_length'] < 1:
            logger.error(f"Invalid chunk_length in chunk {chunk_id}: {meta['chunk_length']}")
            continue
        if not isinstance(meta['created_timestamp'], datetime):
            logger.error(f"Invalid created_timestamp in chunk {chunk_id}: {meta['created_timestamp']}")
            continue
        
        valid_chunks.append(chunk)
    
    logger.info(f"Validation passed for {len(valid_chunks)}/{len(chunks_with_embeddings)} chunks")
    if len(valid_chunks) < len(chunks_with_embeddings):
        logger.warning(f"Dropped {len(chunks_with_embeddings) - len(valid_chunks)} invalid chunks")
    return valid_chunks