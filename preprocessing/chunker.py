import pandas as pd
from typing import List
import re
from datetime import datetime, timezone
import logging
from transformers import PreTrainedTokenizer
from .data_preprocessing import preprocessing_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_tokens(text: str, tokenizer: PreTrainedTokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))

def chunk_field_based_splitting(df: pd.DataFrame, tokenizer: PreTrainedTokenizer) -> pd.DataFrame:
    TARGET_CHUNK_SIZE = 400
    
    strategies = []
    for _, row in df.iterrows():
        input_text = str(row.get('input', '')).strip()
        output_text = str(row.get('output', '')).strip()
        instruction_text = str(row.get('instruction', '')).strip()
        if not input_text or not output_text or not instruction_text:
            logger.warning(f"Skipping row {row.get('id', 'unknown')} due to empty input/output/instruction")
            strategies.append("skip")
            continue
        input_tokens = count_tokens(input_text, tokenizer)
        output_tokens = count_tokens(output_text, tokenizer)
        instruction_tokens = count_tokens(f"Medical Task: {instruction_text}", tokenizer)
        
        total_single_chunk_tokens = instruction_tokens + input_tokens + output_tokens + 20
        
        if total_single_chunk_tokens <= TARGET_CHUNK_SIZE:
            strategy = "single_chunk"
        elif input_tokens > TARGET_CHUNK_SIZE and output_tokens <= TARGET_CHUNK_SIZE:
            strategy = "split_input_keep_output"
        elif input_tokens <= TARGET_CHUNK_SIZE and output_tokens > TARGET_CHUNK_SIZE:
            strategy = "keep_input_split_output"
        else:
            strategy = "split_both_fields"
        strategies.append(strategy)
    
    df['chunking_strategy'] = strategies
    logger.info("Chunking strategies summary:")
    for strategy, count in df['chunking_strategy'].value_counts().items():
        logger.info(f"  {strategy}: {count} rows")
            
    return df

def force_split_by_words(text: str, tokenizer: PreTrainedTokenizer, max_tokens: int) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = ""
    
    for word in words:
        test_chunk = current_chunk + (" " if current_chunk else "") + word
        if count_tokens(test_chunk, tokenizer) <= max_tokens:
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

def split_text_by_token(text: str, tokenizer: PreTrainedTokenizer, max_tokens: int) -> List[str]:
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        test_chunk = current_chunk + (". " if current_chunk else "") + sentence + "."
        if count_tokens(test_chunk, tokenizer) <= max_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence + "."
            else:
                chunks.extend(force_split_by_words(sentence, tokenizer, max_tokens))
                current_chunk = ""
    
    if current_chunk:
        chunks.append(current_chunk)
    return [c for c in chunks if c.strip()]

def create_safe_chunk_content(instruction: str, content_parts: dict, tokenizer: PreTrainedTokenizer, max_tokens: int = 400) -> str:
    instruction = str(instruction).strip()
    if not instruction:
        logger.warning("Empty instruction, using default")
        instruction = "Medical dialogue"
    base_parts = [f"Medical Task: {instruction}"]
    
    for key, value in content_parts.items():
        if value and str(value).strip():
            base_parts.append(f"{key}: {value}")
    
    content = "\n\n".join(base_parts)
    if not content.strip():
        logger.error("Generated empty content, returning empty string")
        return ""
    
    while count_tokens(content, tokenizer) > max_tokens:
        longest_key = max(content_parts.keys(), key=lambda k: len(str(content_parts.get(k, ''))) if content_parts.get(k) else 0)
        if content_parts[longest_key]:
            words = content_parts[longest_key].split()
            content_parts[longest_key] = " ".join(words[:-5]) + "..." if len(words) > 5 else ""
            content = "\n\n".join([f"Medical Task: {instruction}"] + 
                                 [f"{k}: {v}" for k, v in content_parts.items() if v])
        else:
            break
    
    return content.strip()

def single_chunk(df: pd.DataFrame, tokenizer: PreTrainedTokenizer) -> List[dict]:
    chunks = []
    chunk_single = df[df["chunking_strategy"] == "single_chunk"]
    
    for _, row in chunk_single.iterrows():
        content_parts = {
            "Input": str(row.get('input', '')).strip(),
            "Output": str(row.get('output', '')).strip()
        }
        if not content_parts["Input"] or not content_parts["Output"]:
            logger.warning(f"Skipping row {row.get('id', 'unknown')} due to empty input/output")
            continue
        
        chunk_text = create_safe_chunk_content(row.get('instruction', ''), content_parts, tokenizer)
        if not chunk_text:
            logger.warning(f"Skipping row {row.get('id', 'unknown')} due to empty chunk text")
            continue
        
        chunk_metadata = {
            "chunk_id": f"{row['source']}_{row['id']}_single",
            "parent_id": str(row["id"]),
            "source": str(row["source"]),
            "sequence": 1,
            "total_chunks": 1,
            "content_type": "complete_dialogue",
            "task": str(row["task"]),
            "chunk_length": len(chunk_text),
            "token_count": count_tokens(chunk_text, tokenizer),
            "created_timestamp": datetime.now(timezone.utc),
            "related_chunks": []
        }
        
        chunks.append({
            "content": chunk_text,
            "metadata": chunk_metadata
        })

    return chunks

def get_first_words(text: str, num_words: int) -> str:
    words = str(text).split()
    if len(words) <= num_words:
        return text
    return ' '.join(words[:num_words]) + '...'

def split_input_chunk(df: pd.DataFrame, tokenizer: PreTrainedTokenizer) -> List[dict]:
    chunks = []
    chunks_split_input = df[df["chunking_strategy"] == "split_input_keep_output"]
    
    for _, row in chunks_split_input.iterrows():
        input_text = str(row.get('input', '')).strip()
        output_text = str(row.get('output', '')).strip()
        if not input_text or not output_text:
            logger.warning(f"Skipping row {row.get('id', 'unknown')} due to empty input/output")
            continue
        
        input_parts = split_text_by_token(input_text, tokenizer, max_tokens=200)
        if not input_parts:
            logger.warning(f"Skipping row {row.get('id', 'unknown')} due to empty input parts")
            continue
        
        total_chunks = len(input_parts)
        for i, input_part in enumerate(input_parts, 1):
            content_parts = {"Patient Question": input_part} if i == 1 else {"Patient Question (continued)": input_part}
            if i == total_chunks:
                content_parts["Doctor Response"] = output_text
                content_type = "patient_question_final_and_response"
            else:
                content_type = f"patient_question_part{i}"
            
            chunk_text = create_safe_chunk_content(row.get('instruction', ''), content_parts, tokenizer)
            if not chunk_text:
                logger.warning(f"Skipping row {row.get('id', 'unknown')} chunk {i} due to empty chunk text")
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
                "chunk_length": len(chunk_text),
                "token_count": count_tokens(chunk_text, tokenizer),
                "created_timestamp": datetime.now(timezone.utc)
            }
            
            chunks.append({"content": chunk_text, "metadata": chunk_metadata})
        
    logger.info(f"Created {len(chunks)} chunks from split input strategy")
    return chunks

def split_output_chunk(df: pd.DataFrame, tokenizer: PreTrainedTokenizer) -> List[dict]:
    chunks = []
    chunks_split_output = df[df["chunking_strategy"] == "keep_input_split_output"]
    
    for _, row in chunks_split_output.iterrows():
        input_text = str(row.get('input', '')).strip()
        output_text = str(row.get('output', '')).strip()
        if not input_text or not output_text:
            logger.warning(f"Skipping row {row.get('id', 'unknown')} due to empty input/output")
            continue
        
        output_parts = split_text_by_token(output_text, tokenizer, max_tokens=200)
        if not output_parts:
            logger.warning(f"Skipping row {row.get('id', 'unknown')} due to empty output parts")
            continue
        
        patient_input_ref = get_first_words(input_text, 15)
        total_chunks = len(output_parts)
        
        for i, output_part in enumerate(output_parts, 1):
            content_parts = {
                "Patient": input_text if i == 1 else patient_input_ref,
                "Doctor Response" if i == 1 else "Doctor Response (continued)": output_part
            }
            chunk_text = create_safe_chunk_content(row.get('instruction', ''), content_parts, tokenizer)
            if not chunk_text:
                logger.warning(f"Skipping row {row.get('id', 'unknown')} chunk {i} due to empty chunk text")
                continue
            
            chunk_metadata = {
                "chunk_id": f"{row['source']}_{row['id']}_{i:02d}",
                "parent_id": str(row["id"]),
                "sequence": i,
                "total_chunks": total_chunks,
                "content_type": "patient_and_response_part1" if i == 1 else "response_continuation",
                "source": str(row["source"]),
                "task": str(row["task"]),
                "related_chunks": [f"{row['source']}_{row['id']}_{j:02d}" for j in range(1, total_chunks+1) if j != i],
                "chunk_length": len(chunk_text),
                "token_count": count_tokens(chunk_text, tokenizer),
                "created_timestamp": datetime.now(timezone.utc)
            }
            
            chunks.append({"content": chunk_text, "metadata": chunk_metadata})
        
    logger.info(f"Created {len(chunks)} chunks from split output strategy")
    return chunks

def create_split_both_chunks(df: pd.DataFrame, tokenizer: PreTrainedTokenizer) -> List[dict]:
    chunks = []
    split_both_fields = df[df["chunking_strategy"] == "split_both_fields"]
    
    for _, row in split_both_fields.iterrows():
        input_text = str(row.get('input', '')).strip()
        output_text = str(row.get('output', '')).strip()
        if not input_text or not output_text:
            logger.warning(f"Skipping row {row.get('id', 'unknown')} due to empty input/output")
            continue
        
        input_parts = split_text_by_token(input_text, tokenizer, max_tokens=150)
        output_parts = split_text_by_token(output_text, tokenizer, max_tokens=150)
        if not input_parts or not output_parts:
            logger.warning(f"Skipping row {row.get('id', 'unknown')} due to empty input/output parts")
            continue
        
        patient_summary = get_first_words(input_text, 15)
        total_chunks = len(input_parts) + len(output_parts)
        chunk_sequence = 1
        
        for i, input_part in enumerate(input_parts):
            content_parts = {"Patient Question": input_part} if i == 0 else {"Patient Question (continued)": input_part}
            chunk_text = create_safe_chunk_content(row.get('instruction', ''), content_parts, tokenizer)
            if not chunk_text:
                logger.warning(f"Skipping row {row.get('id', 'unknown')} chunk {chunk_sequence} due to empty chunk text")
                continue
            
            chunk_metadata = {
                "chunk_id": f"{row['source']}_{row['id']}_{chunk_sequence:02d}",
                "parent_id": str(row["id"]),
                "sequence": chunk_sequence,
                "total_chunks": total_chunks,
                "content_type": f"patient_question_part{i+1}",
                "source": str(row["source"]),
                "task": str(row["task"]),
                "related_chunks": [f"{row['source']}_{row['id']}_{j:02d}" for j in range(1, total_chunks+1) if j != chunk_sequence],
                "chunk_length": len(chunk_text),
                "token_count": count_tokens(chunk_text, tokenizer),
                "created_timestamp": datetime.now(timezone.utc)
            }
            
            chunks.append({"content": chunk_text, "metadata": chunk_metadata})
            chunk_sequence += 1
        
        for i, output_part in enumerate(output_parts):
            response_key = "Doctor Response" if i == 0 else "Doctor Response (continued)"
            content_parts = {
                "Regarding patient concern": patient_summary,
                response_key: output_part
            }
            chunk_text = create_safe_chunk_content(row.get('instruction', ''), content_parts, tokenizer)
            if not chunk_text:
                logger.warning(f"Skipping row {row.get('id', 'unknown')} chunk {chunk_sequence} due to empty chunk text")
                continue
            
            chunk_metadata = {
                "chunk_id": f"{row['source']}_{row['id']}_{chunk_sequence:02d}",
                "parent_id": str(row["id"]),
                "sequence": chunk_sequence,
                "total_chunks": total_chunks,
                "content_type": f"response_part{i+1}",
                "source": str(row["source"]),
                "task": str(row["task"]),
                "related_chunks": [f"{row['source']}_{row['id']}_{j:02d}" for j in range(1, total_chunks+1) if j != chunk_sequence],
                "chunk_length": len(chunk_text),
                "token_count": count_tokens(chunk_text, tokenizer),
                "created_timestamp": datetime.now(timezone.utc)
            }
            
            chunks.append({"content": chunk_text, "metadata": chunk_metadata})
            chunk_sequence += 1
    
    logger.info(f"Created {len(chunks)} chunks from split both strategy")
    return chunks

def combine_all_chunks(single_chunks, split_input_chunks, split_output_chunks, split_both_chunks) -> List[dict]:
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

def validate_chunks(all_chunks, tokenizer):
    logger.info("\nCHUNK VALIDATION:")
    
    chunks_with_content = sum(1 for chunk in all_chunks if 'content' in chunk and chunk['content'].strip())
    chunks_with_metadata = sum(1 for chunk in all_chunks if 'metadata' in chunk and isinstance(chunk['metadata'], dict))
    
    logger.info(f"Chunks with content: {chunks_with_content}/{len(all_chunks)}")
    logger.info(f"Chunks with metadata: {chunks_with_metadata}/{len(all_chunks)}")
    
    chunk_lengths = [len(chunk['content']) for chunk in all_chunks if chunk['content'].strip()]
    if chunk_lengths:
        avg_length = sum(chunk_lengths) / len(chunk_lengths)
        min_length = min(chunk_lengths)
        max_length = max(chunk_lengths)
        logger.info(f"Average chunk length: {avg_length:.0f} characters")
        logger.info(f"Shortest chunk: {min_length} characters")
        logger.info(f"Longest chunk: {max_length} characters")
    
    token_counts = [chunk['metadata'].get('token_count', count_tokens(chunk['content'], tokenizer)) 
                   for chunk in all_chunks if chunk['content'].strip()]
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
        over_limit = sum(1 for count in token_counts if count > 512)
        logger.info(f"\nTOKEN ANALYSIS:")
        logger.info(f"Average tokens: {avg_tokens:.0f}")
        logger.info(f"Shortest chunk: {min_tokens} tokens")
        logger.info(f"Longest chunk: {max_tokens} tokens")
        logger.info(f"Chunks over 512 tokens: {over_limit}/{len(all_chunks)} ({over_limit/len(all_chunks)*100:.1f}%)")
        if over_limit > 0:
            logger.warning(f"{over_limit} chunks exceed 512 token limit! Adjusting max_tokens recommended.")

    return [chunk for chunk in all_chunks if chunk['content'].strip()]

def chunk_medical_dialogues(df: pd.DataFrame, tokenizer) -> List[dict]:

    processed_df = preprocessing_data(df)
    df_with_strategy = chunk_field_based_splitting(processed_df, tokenizer)
    single_chunks = single_chunk(df_with_strategy, tokenizer)
    split_input_chunks = split_input_chunk(df_with_strategy, tokenizer)
    split_output_chunks = split_output_chunk(df_with_strategy, tokenizer)
    split_both_fields_chunks = create_split_both_chunks(df_with_strategy, tokenizer)
    all_chunks = combine_all_chunks(single_chunks, split_input_chunks, split_output_chunks, split_both_fields_chunks)
    final_chunks = validate_chunks(all_chunks, tokenizer)
    return final_chunks