from typing import List, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

class ToxicityScorer:
    """Score text toxicity using a pre-trained model."""
    
    def __init__(
        self,
        model_name: str = "s-nlp/roberta_toxicity_classifier",
        device: Optional[str] = None
    ):
        device='cuda:0'
        if not device:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print("Toxicity Scorer Device:", self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer Loaded.", flush=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map='cuda:0')
        print("Model Loaded.", flush=True)
        self.model.to(self.device)
        print("Model Moved to Device.", flush=True)
        self.model.eval()
        print("Model Set to Evaluation Mode.", flush=True)

    @torch.no_grad()
    def score_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[float]:
        """Score a batch of texts for toxicity."""
        scores = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            print(f"processing batch {i} to {i + batch_size}", flush=True)
            batch = texts[i:i + batch_size]
            print("Batch len:", len(batch), flush=True)
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )

            # Move inputs to device and set dtype
            inputs = {
                k: v.to(self.device, dtype=torch.long) 
                for k, v in inputs.items()
            }
            print("Model device:", next(self.model.parameters()).device, flush=True)
            print("Input devices:", {k: v.device for k, v in inputs.items()}, flush=True)
            print("Input shape:", inputs['input_ids'].shape, flush=True)
            print("Inputs Created by Tokenizer.", flush=True)
  
            outputs = self.model(**inputs)
            print("Model Outputs Received.", flush=True)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            print("Probabilities Calculated.", flush=True)
            toxic_probs = probabilities[:, 1].cpu().numpy()
            print("Toxicity Probabilities Extracted.", flush=True)
            scores.extend(toxic_probs.tolist())
            print("Scores Extended.", flush=True)

        print("Done for batch size:", batch_size, flush=True)
        print("Toxicity Scores:", scores, flush=True)
        return scores


import pytest

class TestToxicityScorer:
    @pytest.fixture
    def scorer(self):
        """Create a ToxicityScorer instance with CPU device for testing"""
        return ToxicityScorer(device="cpu")

    def test_initialization(self):
        """Test that scorer initializes correctly"""
        scorer = ToxicityScorer(device="cpu")
        assert scorer.device == "cpu"
        assert scorer.model is not None
        assert scorer.tokenizer is not None
        assert scorer.model.training is False  # Should be in eval mode

    def test_score_batch_single_text(self, scorer):
        """Test scoring a single text"""
        texts = ["This is a positive and friendly message."]
        scores = scorer.score_batch(texts, batch_size=1)
        
        assert isinstance(scores, list)
        assert len(scores) == 1
        assert 0 <= scores[0] <= 1  # Toxicity score should be between 0 and 1

    def test_score_batch_multiple_texts(self, scorer):
        """Test scoring multiple texts"""
        texts = [
            "This is a positive message.",
            "I hate everything!",
            "Have a nice day."
        ]
        scores = scorer.score_batch(texts, batch_size=2)
        
        assert isinstance(scores, list)
        assert len(scores) == 3
        assert all(0 <= score <= 1 for score in scores)
        # The second text should have a higher toxicity score
        assert scores[1] > scores[0]
        assert scores[1] > scores[2]

    def test_empty_input(self, scorer):
        """Test handling of empty input"""
        texts = []
        scores = scorer.score_batch(texts)
        assert scores == []

    def test_batch_size_handling(self, scorer):
        """Test that different batch sizes produce consistent results"""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        
        # Score with batch_size=2
        scores_batch_2 = scorer.score_batch(texts, batch_size=2)
        # Score with batch_size=4
        scores_batch_4 = scorer.score_batch(texts, batch_size=4)
        
        assert len(scores_batch_2) == len(scores_batch_4)
        # Scores should be very close regardless of batch size
        assert all(abs(s1 - s2) < 1e-5 for s1, s2 in zip(scores_batch_2, scores_batch_4))

    def test_long_text_handling(self, scorer):
        """Test handling of very long text"""
        long_text = "word " * 1000  # Text longer than max_length
        scores = scorer.score_batch([long_text])
        
        assert isinstance(scores, list)
        assert len(scores) == 1
        assert 0 <= scores[0] <= 1

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_different_batch_sizes(self, scorer, batch_size):
        """Test that scorer works with different batch sizes"""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        scores = scorer.score_batch(texts, batch_size=batch_size)
        
        assert len(scores) == len(texts)
        assert all(0 <= score <= 1 for score in scores)

# scorer = ToxicityScorer()
# TestToxicityScorer().test_long_text_handling(scorer=scorer)
# TestToxicityScorer().test_initialization()
# TestToxicityScorer().test_score_batch_single_text(scorer=scorer)
# TestToxicityScorer().test_score_batch_multiple_texts(scorer=scorer)  
# TestToxicityScorer().test_empty_input(scorer=scorer)
# TestToxicityScorer().test_batch_size_handling(scorer=scorer) 