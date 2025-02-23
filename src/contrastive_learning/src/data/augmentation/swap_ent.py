import spacy
from typing import List, Tuple, Set, Dict
import random
import re
from collections import defaultdict

class EntitySwapper:
    """Swap named entities while maintaining coherence without neuralcoref."""
    
    def __init__(self):
        # Initialize spaCy with transformer pipeline for better NER
        self.nlp = spacy.load("en_core_web_trf")
        
        self.entity_types = {
            "PERSON", "ORG", "GPE", "LOC", "PRODUCT",
            "WORK_OF_ART", "EVENT", "DATE", "TIME", "PERCENT", 
            "MONEY", "QUANTITY", "CARDINAL"
        }
        
        # Common pronouns for different entity types
        self.entity_pronouns = {
            "PERSON": {"he", "him", "his", "she", "her", "hers", "they", "them", "their"},
            "ORG": {"it", "its", "they", "them", "their"},
            "GPE": {"it", "its", "they", "them", "their"},
            "PRODUCT": {"it", "its"},
        }

    def _get_entity_context(
        self,
        doc: spacy.tokens.Doc,
        entity: spacy.tokens.Span
    ) -> Dict[str, Set[str]]:
        """Get contextual information about an entity."""
        context = {
            "verbs": set(),
            "deps": set(),
            "head_pos": set(),
            "following_tokens": set()
        }
        
        # Get surrounding context
        start_idx = max(0, entity.start - 3)
        end_idx = min(len(doc), entity.end + 3)
        
        # Collect verbs and dependencies
        for token in doc[start_idx:end_idx]:
            if token.pos_ == "VERB":
                context["verbs"].add(token.lemma_)
            if token.dep_ not in {"punct", "det"}:
                context["deps"].add(token.dep_)
        
        # Get head token information
        for token in entity:
            if token.head.pos_ not in {"PUNCT", "DET"}:
                context["head_pos"].add(token.head.pos_)
            
        # Get following token (for possessives, etc.)
        if entity.end < len(doc):
            context["following_tokens"].add(doc[entity.end].text)
            
        return context

    def _find_compatible_entities(
        self,
        ent: spacy.tokens.Span,
        candidates: List[spacy.tokens.Span],
        source_doc: spacy.tokens.Doc,
        target_doc: spacy.tokens.Doc
    ) -> List[spacy.tokens.Span]:
        """Find entities that can be swapped while maintaining coherence."""
        ent_context = self._get_entity_context(target_doc, ent)
        compatible = []
        
        for candidate in candidates:
            cand_context = self._get_entity_context(source_doc, candidate)
            
            # Check context compatibility
            context_match = (
                len(ent_context["deps"] & cand_context["deps"]) > 0 and
                len(ent_context["head_pos"] & cand_context["head_pos"]) > 0
            )
            
            # Check number agreement (singular/plural)
            number_match = (
                (ent.root.morph.get("Number") == candidate.root.morph.get("Number")) or
                (not ent.root.morph.get("Number") and not candidate.root.morph.get("Number"))
            )
            
            if context_match and number_match:
                compatible.append(candidate)
                
        return compatible

    def _find_swappable_entities(
        self, 
        doc: spacy.tokens.Doc,
        source_doc: spacy.tokens.Doc
    ) -> List[Tuple[spacy.tokens.Span, List[spacy.tokens.Span]]]:
        """Find entities that can be swapped."""
        swappable = []
        
        # Group entities by type
        target_entities = defaultdict(list)
        source_entities = defaultdict(list)
        
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                target_entities[ent.label_].append(ent)
                
        for ent in source_doc.ents:
            if ent.label_ in self.entity_types:
                source_entities[ent.label_].append(ent)
        
        # Find swappable entities
        for ent in doc.ents:
            if ent.label_ not in self.entity_types:
                continue
            
            # Get potential candidates of the same type
            candidates = [
                src_ent for src_ent in source_entities[ent.label_]
                if src_ent.text.lower() != ent.text.lower()
            ]
            
            # Filter candidates based on compatibility
            compatible_candidates = self._find_compatible_entities(
                ent, candidates, source_doc, doc
            )
            
            if compatible_candidates:
                swappable.append((ent, compatible_candidates))
                
        return swappable

    def _adjust_entity_case(
        self,
        original_text: str,
        replacement_text: str,
        start_idx: int
    ) -> str:
        """Adjust entity case based on position and original text."""
        if start_idx >= len(original_text):
            return replacement_text
        if start_idx == 0 or original_text[start_idx - 1] in {'.', '!', '?', '\n'}:
            return replacement_text.capitalize()
        elif original_text[start_idx].isupper():
            return replacement_text.upper()
        return replacement_text

    def _swap_entity(
        self,
        text: str,
        entity: spacy.tokens.Span,
        replacement: spacy.tokens.Span
    ) -> str:
        """Swap a single entity in the text."""
        start = entity.start_char
        end = entity.end_char
        
        replacement_text = replacement.text
        
        # Adjust case
        replacement_text = self._adjust_entity_case(text, replacement_text, start)
            
        # Handle possessives
        if text[end-2:end] == "'s":
            if not replacement_text.endswith('s'):
                replacement_text += "'s"
            else:
                replacement_text += "'"
        elif text[end-1:end] == "s'" and replacement_text.endswith('s'):
            replacement_text += "'"
            
        # Maintain spacing
        if end < len(text) and text[end].isspace():
            replacement_text += text[end]
            
        return text[:start] + replacement_text + text[end:]

    def generate(
       self,
       text: str,
       source: str,
       global_idx: int,
       max_variations: int = 3
   ) -> List[Dict]:
        """Generate samples by swapping entities with rank tracking."""
        doc = self.nlp(text)
        source_doc = self.nlp(source)
        
        swappable = self._find_swappable_entities(doc, source_doc)
        if not swappable:
            return [{
                    "original_idx": global_idx,
                    "augmented": False,
                }]
            
        results = []
        # Generate variations
        for _ in range(min(max_variations, len(swappable))):
            current_text = text
            
            # Swap 1-2 entities
            num_swaps = random.randint(1, min(2, len(swappable)))
            chosen = random.sample(swappable, num_swaps)
            
            for ent, candidates in chosen:
                replacement = random.choice(candidates)
                current_text = self._swap_entity(current_text, ent, replacement)
                
            if current_text != text:  # Only add if actually changed
                results.append({
                    "summary": current_text,
                    "post": source,
                    "original_idx": global_idx,
                    "augmented": True,
                })
        
        # Return unique results based on summary text
        seen = set()
        unique_results = []
        for res in results:
            if res["summary"] not in seen:
                seen.add(res["summary"])
                unique_results.append(res)

        if unique_results == []:
           return [{
                "original_idx": global_idx,
                "augmented": False,
            }]
       
        return unique_results