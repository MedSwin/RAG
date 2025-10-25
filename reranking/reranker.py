import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Union
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentReranker:
    # DocumentReranker class to rerank documents based on their relevance to a query
    # using different BGE reranker models
    def __init__(self, model_path: str, model_type: str = "auto"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = self._detect_model_type() if model_type == "auto" else model_type
        
        logger.info(f"Loading reranker model from {model_path} on {self.device}")
        logger.info(f"Detected model type: {self.model_type}")
        
        self.tokenizer, self.model = self._load_model()
        
    def _detect_model_type(self) -> str:
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise ValueError(f"Config file not found at {config_path}")
            
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        model_type = config.get("model_type", "").lower()
        architectures = config.get("architectures", [])
        
        if "xlm-roberta" in model_type or "XLMRobertaForSequenceClassification" in architectures:
            return "bge-m3"
        elif "gemma" in model_type or "GemmaForCausalLM" in architectures:
            return "gemma"
        else:
            raise ValueError(f"Unsupported model type: {model_type}, architectures: {architectures}")
    
    def _load_model(self) -> Tuple[AutoTokenizer, Union[AutoModelForSequenceClassification, AutoModelForCausalLM]]:
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.model_type == "bge-m3":
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        elif self.model_type == "gemma":
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        model = model.to(self.device)
        model.eval()
        
        return tokenizer, model
    
    def _prepare_input_bge_m3(self, query: str, document: str) -> Dict[str, torch.Tensor]:
        # For BGE rerankers, we typically concatenate query and document
        text = f"{query} [SEP] {document}"
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def _prepare_input_gemma(self, query: str, document: str) -> Dict[str, torch.Tensor]:
        # For Gemma models, we use a specific prompt format
        prompt = f"Query: {query}\nDocument: {document}\nRelevant:"
        
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def _compute_score_bge_m3(self, inputs: Dict[str, torch.Tensor]) -> float:
        with torch.no_grad():
            outputs = self.model(**inputs)
            # For sequence classification, get the logits and apply softmax
            logits = outputs.logits
            scores = F.softmax(logits, dim=-1)
            # Assuming positive relevance is the second class (index 1)
            # If it's a single output, use sigmoid
            if scores.shape[-1] == 1:
                score = torch.sigmoid(logits).item()
            else:
                score = scores[0, 1].item() if scores.shape[-1] > 1 else scores[0, 0].item()
            
        return score
    
    def _compute_score_gemma(self, inputs: Dict[str, torch.Tensor]) -> float:
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get the last token's logits
            last_token_logits = logits[0, -1, :]
            
            # Find tokens for "Yes" and "No" 
            yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
            
            yes_score = last_token_logits[yes_token_id].item()
            no_score = last_token_logits[no_token_id].item()
            
            # Apply softmax to get probability
            scores = F.softmax(torch.tensor([no_score, yes_score]), dim=0)
            score = scores[1].item()  # Probability of "Yes"
            
        return score
    
    def compute_relevance_score(self, query: str, document: str) -> float:
        if self.model_type == "bge-m3":
            inputs = self._prepare_input_bge_m3(query, document)
            return self._compute_score_bge_m3(inputs)
        elif self.model_type == "gemma":
            inputs = self._prepare_input_gemma(query, document)
            return self._compute_score_gemma(inputs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def rerank_documents(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:    
        logger.info(f"Reranking {len(documents)} documents for query: {query[:100]}...")
        
        scored_documents = []
        
        for doc in documents:
            content = doc.get('content', '')
            if not content:
                logger.warning(f"Empty content for document {doc.get('chunk_id', 'unknown')}")
                continue
                
            try:
                score = self.compute_relevance_score(query, content)
                doc_with_score = doc.copy()
                doc_with_score['rerank_score'] = score
                scored_documents.append(doc_with_score)
                
                logger.debug(f"Document {doc.get('chunk_id', 'unknown')}: score={score:.4f}")
                
            except Exception as e:
                logger.error(f"Error scoring document {doc.get('chunk_id', 'unknown')}: {e}")
                continue
        
        # Sort by rerank score (descending)
        scored_documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Return top_k documents
        top_documents = scored_documents[:top_k]
        
        logger.info(f"Reranking complete. Top {len(top_documents)} documents selected.")
        for i, doc in enumerate(top_documents[:3], 1):  # Log top 3 scores
            logger.info(f"Rank {i}: score={doc['rerank_score']:.4f}, chunk_id={doc.get('chunk_id', 'unknown')}")
        
        return top_documents

def load_reranker(model_name: str = "bge-reranker-v2-m3") -> DocumentReranker:

    base_path = Path("/fred/oz446/HenryNguyen/reranker")
    model_path = base_path / model_name
    
    if not model_path.exists():
        raise ValueError(f"Model not found at {model_path}")
    
    return DocumentReranker(str(model_path))


