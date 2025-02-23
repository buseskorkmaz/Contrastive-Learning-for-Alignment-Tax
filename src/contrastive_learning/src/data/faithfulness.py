from typing import List, Optional
import torch
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
import numpy as np

class FaithfulnessScorer:
    """Score summary faithfulness using semantic similarity."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None
    ):
        if not device:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print("Faithfulness Scorer Device:", self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _get_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> torch.Tensor:
        """Get embeddings for a list of texts."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                all_embeddings.append(embeddings)
                
        return np.vstack(all_embeddings)

    def score_batch(
        self,
        sources: List[str],
        summaries: List[str],
        batch_size: int = 32
    ) -> List[float]:
        """Score faithfulness using semantic similarity."""
        source_embeddings = self._get_embeddings(sources, batch_size)
        summary_embeddings = self._get_embeddings(summaries, batch_size)
        print("Computing faithfulness scores...", flush=True)
        scores = []
        for src_emb, sum_emb in zip(source_embeddings, summary_embeddings):
            similarity = 1 - cosine(src_emb, sum_emb)
            scores.append(float(similarity))
            
        return scores
