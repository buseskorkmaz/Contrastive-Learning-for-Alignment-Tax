import glob
import json
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import pandas as pd

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
import tqdm
import json
import random
import sys
import numpy as np
import traceback
import jsonlines
from torch import nn
from nltk import sent_tokenize
import itertools
import spacy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FactualityDetector:
    def __init__(self, modelname, max_seq_length=512):
        self.model = AutoModelForSequenceClassification.from_pretrained(modelname, subfolder='factuality_detector')
        self.tokenizer = AutoTokenizer.from_pretrained(modelname, subfolder='factuality_detector')
        self.max_seq_length = max_seq_length

        self.model = self.model.to(device)
        self.model.eval()
        self.nlp = spacy.load("en_core_web_sm")

    def break_context(self, context, n_gram=1, n_skip=1, all_possible=False):
        sents = sent_tokenize(context)
        len_nums = len(sents)
        sent_fragments = []
        if all_possible:
            sent_fragments = [" ".join(x) for x in list(itertools.permutations(sents, n_gram))]
        else:
            for k in range(0, len_nums, n_skip):
                sent_fragments.append(" ".join(sents[k:k+n_gram]))
        return sent_fragments
    
    
    def compute_summac_score(self, raw_batch, pool_mode='avg', return_matrix=False, n_gram=2, n_skip=2, all_possible=False):
        batch_response = []
        batch_value_matrix = []
        batch_context_response = []
        for item in raw_batch:
            context, response = item
            context_sents = self.break_context(context, n_gram=n_gram, n_skip=n_skip, all_possible=all_possible)
            response_sents = self.break_context(response)
            M=len(context_sents)
            N=len(response_sents)
            value_matrix = np.zeros(shape=(M, N))
            batch_data = []
            for this_context in (context_sents):
                for this_response in (response_sents):
                    batch_data.append((this_context, this_response))
            # [(this_context, this_response)]
            # import ipdb; ipdb.set_trace()
            model_input = self.tokenizer(batch_data, padding=True, truncation=True, 
                                         max_length=self.max_seq_length, return_tensors="pt")
            model_input = model_input.to(device)
            with torch.no_grad(): 
                logits = self.model(**model_input).logits
            probabilities = nn.functional.softmax(logits, dim=-1)
            generated_rewards=probabilities[:,1].detach().cpu().numpy()
            value_matrix=np.reshape(generated_rewards, (M,N)) # generated_rewards.item()

            batch_value_matrix.append(value_matrix)
            batch_context_response.append((context_sents, response_sents))
            response_factuality = np.max(value_matrix, axis=0)
            response_factuality = np.mean(response_factuality) if pool_mode=='avg' else np.min(response_factuality)
            batch_response.append(response_factuality)
    
        if return_matrix:
            return batch_value_matrix, batch_context_response, batch_response
        else:
            return batch_response


    def is_single_fact_response(self, input_text):
        # Process the input text with spaCy
        doc = self.nlp(input_text)
        # Check for the presence of a subject and a verb
        is_single_fact = all([token.pos_ in ['PROPN', 'ADP', 'PROPN'] for token in doc])
        return is_single_fact

    def compute_direct_scores(self, raw_batch):
        model_input = self.tokenizer(raw_batch, padding=True, truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        model_input = model_input.to(device)
        with torch.no_grad(): 
            logits = self.model(**model_input).logits
        probabilities = nn.functional.softmax(logits, dim=-1)
        generated_rewards=probabilities[:,1].cpu()
        return generated_rewards

    def generate_score(self, contexts, responses, summac_style=False, n_gram=2, n_skip=2, 
                       all_possible=False, compare_with_direct_score=True, direct_compare_threshold=0.3,
                       add_question_if_not_sentence=False, questions=None):
        for k, r in enumerate(responses):
            if self.is_single_fact_response(r) and add_question_if_not_sentence:
                responses[k] = f"{questions[k]} {r}"
        # print(responses)
                
        raw_batch = [(c, r) for c, r in zip(contexts, responses)]
        if summac_style:
            value_matrix, context_response, factuality_score = self.compute_summac_score(raw_batch, return_matrix=True, 
                                                                                         n_gram=n_gram, n_skip=n_skip, 
                                                                                         all_possible=all_possible)
            # only use direct baseline if the summac style factuality score is not close to 0 or 1
            threshold_condition = (factuality_score[0]>direct_compare_threshold) \
                and (factuality_score[0]<(1-direct_compare_threshold))
            if compare_with_direct_score and threshold_condition:
                direct_score = self.compute_direct_scores(raw_batch).mean().item()
                factuality_score = max(factuality_score[0], direct_score)
            else:
                factuality_score = factuality_score[0]
            return value_matrix, context_response, factuality_score
        else:
            generated_rewards = self.compute_direct_scores(raw_batch)
            return generated_rewards
        

    def generate_score_auto_ngram(self, contexts, responses, summac_style=False, 
                                  n_gram_list=[2, 3, 4, 5, 6], n_skip=2, all_possible=False):
        raw_batch = [(c, r) for c, r in zip(contexts, responses)]
        assert summac_style, "SummaC needs to be true for auto ngram"
        best_score=0
        best_ngram=0
        best_returns=None
        for n_gram in n_gram_list:
            value_matrix, context_response, factuality_score = self.compute_summac_score(raw_batch, return_matrix=True, 
                                                                                         n_gram=n_gram, n_skip=n_skip, 
                                                                                         all_possible=all_possible,
                                                                                         compare_with_direct_score=False)
            if factuality_score[0]>best_score:
                best_score=factuality_score[0]
                best_ngram=n_gram
                best_returns=(value_matrix, context_response, factuality_score)

        print("Best ngram: ", best_ngram)
        return best_returns

# detector = FactualityDetector("/dccstor/nsllm/research/models/factuality/token_type_512_synthetic/model_mnli_snli")


# document='''The film about a princess's mythical journey in ancient Polynesia took an estimated $81.1m (£65.3m) on its debut. That makes it the second-highest Thanksgiving debut of all time, behind Disney's Frozen, which took $93.6m (£75.3m) on its release in 2013. Some observers have said that Moana and its merchandise are appropriating Pacific Island culture. Disney withdrew a children's costume promoting the film after activists branded it "brownface", or mocking of their culture by stereotyping. The costume, a full-body suit with brown skin, traditional tattoos, grass skirt and bone necklace, represented the character Maui, considered a demi-god and ancestor by many Polynesians. Disney said it regretted any offence. JK Rowling's Fantastic Beasts and Where to Find Them fell to second on the US chart, taking $65.8m (£53m). Gossip surrounding Brad Pitt's marriage break-up failed to spark a huge amount of interest in his World War Two romance Allied, which also stars Marion Cotillard. It took $18m (£14.4m) over the long weekend, having cost $85m (£68.5m) to make, landing in fourth spot behind Doctor Strange. Kyle Davies, Paramount's head of domestic distribution, said the film appealed to "older audiences" but noted those "don't storm the theatres [on] weekend one". "I think they're going to take their time," he added. Warren Beatty fared worse - his first film in 15 years, the 1950s Hollywood comedy Rules Don't Apply, took just $2.2m (£1.7m). The film is Beatty's first directed feature since 1998's Bulworth. Bad Santa 2, released 13 years after the original and again starring Billy Bob Thornton, did a little better, taking $9m (£7.3m). Follow us on Facebook, on Twitter @BBCNewsEnts, or on Instagram at bbcnewsents. If you have a story suggestion email entertainment.news@bbc.co.uk.'''

# response1="Disney's latest animation Moana dominated the Thanksgiving box office over the five-day US holiday weekend."
# # response1="South Sudan's civil war could become violent, the UN has warned"
# response2="Disney's latest animation Moana dominated the Thanksgiving box office."
# score1 = detector.generate_score([document], [response1])
# print("Score: ", score1)

# score2 = detector.generate_score([document], [response2])
# print("Score: ", score2)

# value_matrix, context_response, factuality_score = detector.generate_score([document], [response1], summac_style=True, n_gram=2, n_skip=1)
# print("Summac style Score: ", factuality_score)

# value_matrix, context_response, factuality_score = detector.generate_score([document], [response2], summac_style=True, n_gram=2, n_skip=1)
# print("Summac style Score: ", factuality_score)

