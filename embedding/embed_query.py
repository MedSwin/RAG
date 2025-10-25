import numpy as np 
import torch
from embedding import mean_pooling
import torch.nn.functional as F

def embed_query(query: str,tokenizer, embed_model, device) -> np.ndarray:
    """Embed a query using a tokenizer and embed model.

    Args:
        query (str): The query to embed.
        tokenizer (AutoTokenizer): The tokenizer to use => Remember that this is the tokenizer for the embed model.
        embed_model (AutoModel): The embed model to use => Remember that this is the embed model.
        device (torch.device): The device to use => Remember that this is the device to use.

    Returns:
        np.ndarray: The embedded query.
    """
    inputs = tokenizer([query], padding=True, truncation=True, return_tensors='pt')
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = embed_model(**inputs)
    
    attention_mask = inputs["attention_mask"]
    query_embedding = mean_pooling(outputs.last_hidden_state, attention_mask)
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    
    return query_embedding.cpu().numpy().astype(np.float64)



