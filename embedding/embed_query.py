import numpy as np 
import torch
from embedding import mean_pooling
import torch.nn.functional as F

def embed_query(query: str,tokenizer, embed_model, device) -> np.ndarray:
    inputs = tokenizer([query], padding=True, truncation=True, return_tensors='pt')
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = embed_model(**inputs)
    
    attention_mask = inputs["attention_mask"]
    query_embedding = mean_pooling(outputs.last_hidden_state, attention_mask)
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    
    return query_embedding.cpu().numpy().astype(np.float64)



