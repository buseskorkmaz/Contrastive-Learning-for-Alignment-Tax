import random
from typing import List, Dict



def generate_incorrect_summary(
        self,
        text: str,
        source: str,
        global_idx: int,
        num_samples: int = 1
    ) -> List[Dict]:
        """
        Creates 'unfaithful' or 'incorrect' versions by messing up the text:
        - Shuffle sentences
        - Replace or randomise numbers
        - Insert contradictory phrases
        ...
        """
        if not text:
            return []
        doc = self.nlp(text)
        sents = list(doc.sents)

        results = []
        for _ in range(num_samples):
            corrupted = text

            # Example 1: partial sentence shuffle if multiple sentences
            if len(sents) > 1 and random.random() < 0.6:
                random_sents = sents[:]
                random.shuffle(random_sents)
                corrupted = " ".join(sent.text for sent in random_sents)

            # Example 2: numeric corruption with naive regex
            numeric_matches = re.findall(r"\b\d+\b", corrupted)
            for match in numeric_matches:
                if random.random() < 0.5:
                    new_num = str(random.randint(1, 9999))
                    corrupted = re.sub(rf"\b{match}\b", new_num, corrupted, count=1)

            # Example 3: insert contradictory phrase
            if random.random() < 0.3:
                contradiction_phrases = [
                    " Actually, the opposite is true,",
                    " In fact, that part is entirely reversed,",
                    " Contrarily, the entire scenario is false,",
                ]
                splitted = corrupted.split()
                if splitted:
                    insertion_idx = random.randint(0, len(splitted) - 1)
                    splitted.insert(insertion_idx, random.choice(contradiction_phrases))
                    corrupted = " ".join(splitted)

            if corrupted != text:
                results.append({
                    "summary": corrupted,
                    "post": source,
                    "original_idx": global_idx,
                    "augmented": True
                })
        return results
