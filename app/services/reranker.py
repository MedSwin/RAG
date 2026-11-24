import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Union
import json

from app.models.pooling import inference_device
from app.services.medswin.governance import redact_phi_text

logger = logging.getLogger(__name__)

class DocumentReranker:
    """Document reranker for improving retrieval quality."""
    
    def __init__(self, model_path: str, model_type: str = "auto"):
        self.model_path = Path(model_path)
        self.device = inference_device()
        self.model_type = self._detect_model_type() if model_type == "auto" else model_type
        
        logger.info(f"Loading reranker model from {model_path} on {self.device}")
        logger.info(f"Detected model type: {self.model_type}")
        
        self.tokenizer, self.model = self._load_model()
    
    def _detect_model_type(self) -> str:
        """Detect model type from config."""
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise ValueError(f"Config file not found at {config_path}")
        
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
        """Load the reranker model."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.model_type == "bge-m3":
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                dtype=torch.float32,
                trust_remote_code=True
            )
        elif self.model_type == "gemma":
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.float32,
                trust_remote_code=True
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        model = model.to(self.device)
        model.eval()
        
        return tokenizer, model
    
    def _prepare_input_bge_m3(self, query: str, document: str) -> Dict[str, torch.Tensor]:
        """Prepare input for BGE-M3 model."""
        inputs = self.tokenizer(
            query,
            document,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def _prepare_input_gemma(self, query: str, document: str) -> Dict[str, torch.Tensor]:
        """Prepare input for Gemma model."""
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
        """Compute relevance score for BGE-M3 model."""
        with torch.no_grad():
            logits = self.model(**inputs).logits
            if logits.shape[-1] == 1:
                score = torch.sigmoid(logits.view(-1)[0]).item()
            else:
                score = F.softmax(logits, dim=-1)[0, 1].item()
        return score
    
    def _compute_score_gemma(self, inputs: Dict[str, torch.Tensor]) -> float:
        """Compute relevance score for Gemma model."""
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            last_token_logits = logits[0, -1, :]
            
            yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
            
            yes_score = last_token_logits[yes_token_id].item()
            no_score = last_token_logits[no_token_id].item()
            
            scores = F.softmax(torch.tensor([no_score, yes_score]), dim=0)
            score = scores[1].item()
        
        return score
    
    def compute_relevance_score(self, query: str, document: str) -> float:
        """Compute relevance score between query and document."""
        if self.model_type == "bge-m3":
            inputs = self._prepare_input_bge_m3(query, document)
            return self._compute_score_bge_m3(inputs)
        elif self.model_type == "gemma":
            inputs = self._prepare_input_gemma(query, document)
            return self._compute_score_gemma(inputs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def compute_relevance_scores(
        self, query: str, documents: List[str], batch_size: int = 16
    ) -> List[float]:
        """Compute relevance scores in batches on GPU/MPS/CPU."""
        if not documents:
            return []
        if self.model_type != "bge-m3":
            return [self.compute_relevance_score(query, doc) for doc in documents]
        scores: List[float] = []
        for start in range(0, len(documents), max(1, batch_size)):
            batch_docs = documents[start : start + max(1, batch_size)]
            inputs = self.tokenizer(
                [query] * len(batch_docs),
                batch_docs,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
                if logits.shape[-1] == 1:
                    probs = torch.sigmoid(logits.view(-1)).detach().cpu().tolist()
                else:
                    probs = F.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
                scores.extend(float(p) for p in probs)
        return scores
    
    def rerank_documents(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """Rerank documents based on relevance to query."""
        query = redact_phi_text(query or "")
        logger.info(
            "Reranking %s documents for query: %s",
            len(documents),
            query[:100],
        )
        
        scored_documents = []
        
        for doc in documents:
            content = redact_phi_text(doc.get('text') or doc.get('content', '') or "")
            if not content:
                logger.warning(f"Empty content for document {doc.get('chunk_id', 'unknown')}")
                continue
            
            try:
                score = self.compute_relevance_score(query, content)
                doc_with_score = doc.copy()
                doc_with_score['rerank_score'] = score
                scored_documents.append(doc_with_score)
                
            except Exception as e:
                logger.error(
                    "Error scoring document %s: %s",
                    doc.get("chunk_id", "unknown"),
                    type(e).__name__,
                )
                continue
        
        # Sort by rerank score (descending)
        scored_documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Return top_k documents
        top_documents = scored_documents[:top_k]
        
        logger.info(f"Reranking complete. Top {len(top_documents)} documents selected.")
        return top_documents

