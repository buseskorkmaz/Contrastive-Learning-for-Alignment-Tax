import json
import os
import re
import string

#from absl import app
import nltk
from nltk import tokenize
import numpy as np
import rank_bm25
import spacy
from dotenv import dotenv_values, load_dotenv
from genai import Credentials, Client
from genai.text.generation import CreateExecutionOptions
from genai.schema import (
    DecodingMethod,
    TextGenerationParameters,
    TextGenerationReturnOptions,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    print("Please install tqdm to run this example.")
    raise

parameters = TextGenerationParameters(
    max_new_tokens=1000,
    min_new_tokens=1,
    decoding_method=DecodingMethod.GREEDY,
    return_options=TextGenerationReturnOptions(
        # if ordered is False, you can use return_options to retrieve the corresponding prompt
        input_text=True,
    ),
)

here = os.path.dirname(__file__)
if here != '':
    here = here+'/'

with open(here+'../data/prompt_templates.json','r') as f:
    prompt_templates = json.load(f)

nltk.download('punkt', quiet=True)

MONTHS = [
    m.lower()
    for m in [
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
        'July',
        'August',
        'September',
        'October',
        'November',
        'December',
    ]
]
SPACY_MODEL = spacy.load('en_core_web_sm')

ATOMIC_FACT_INSTRUCTION = """\
Instructions:
1. You are given a paragraph. Your task is to break the sentence down into \
a list of atomic statements without adding any new information.
2. An atomic statement is a sentence containing a singular piece of information \
directly extracted from the provided paragraph.
3. Atomic statements may contradict one another.
4. The paragraph may contain information that is factually incorrect. Even in such \
cases, you are not to alter any information contained in the paragraph and must \
produce atomic statements that are completely faithful to the information in the paragraph.
5. Each atomic statement in the outputted list should check a different piece of \
information found explicitly in the paragraph.
6. Each atomic statement is standalone in that any actual nouns or proper nouns \
should be used in place of pronouns or anaphors.
7. Each atomic statement must not include any information beyond what is explicitly \
stated in the provided paragraph.
8. Where possible, avoid paraphrasing and instead try to only use language used in the \
paragraph without introducing new words. 
9. Use the previous examples to learn how to do this.
10. You should only output the atomic statement as a list, with each item starting \
with "- ". Do not include other formatting.
11. Your task is to do this for the last paragraph that is given. 
"""
class AtomicFactGenerator(object):
    """Atomic fact generator."""

    def __init__(
        self,
        model = "meta-llama/llama-3-70b-instruct",
        ):
            self.nlp = SPACY_MODEL

            self.model = model
        
            load_dotenv(override=True)
            credentials = Credentials.from_env()
            self.client = Client(credentials=credentials)
            self.model = model

            # get the demos
            #with utils.open_file_wrapped(self.demon_path, mode='r') as f:
            demos = []
            # with open(here+'demos/demos.txt', 'r') as f:
            #     for line in f:
            #         demos.append(line)
            # self.demos = demos     

    def run(self, paragraphs: list):
        """Convert the generation into a set of atomic facts."""
        assert isinstance(paragraphs, list), 'generation must be a list'
        assert all(isinstance(gen, str) for gen in paragraphs), 'each item in generation must be a string'

        return self.get_atomic_facts_from_paragraphs(paragraphs)

    def get_atomic_facts_from_paragraphs(self, paragraphs):
        """Get the atomic facts from the paragraphs."""
        paragraphs4prompts = []

        for paragraph in paragraphs:

            paragraphs4prompts.append([])
            initials = detect_initials(paragraph)
            curr_sentences = tokenize.sent_tokenize(paragraph)
            curr_sentences_2 = tokenize.sent_tokenize(paragraph)
            curr_sentences = fix_sentence_splitter(curr_sentences, initials)
            curr_sentences_2 = fix_sentence_splitter(curr_sentences_2, initials)
            # ensure the credability of the sentence splitter fixing algorithm
            assert curr_sentences == curr_sentences_2, (
                paragraph,
                curr_sentences,
                curr_sentences_2,
            )
            paragraphs4prompts[-1] += curr_sentences

        prompts = self.create_prompts(paragraphs4prompts)

        atoms = self.get_atomic_facts(prompts)

        return atoms

    def get_atomic_facts(self, prompts):
        
        results = []
        for idx, response in tqdm(
            enumerate(
                self.client.text.generation.create(
                    model_id=self.model,
                    inputs=prompts,
                    # set to ordered to True if you need results in the same order as prompts
                    execution_options=CreateExecutionOptions(ordered=True),
                    parameters=parameters,
                )
            ),
            total=len(prompts),
            desc="Progress",
            unit="input",
            ):
                results.append(response.results[0].generated_text)

        atoms = self.process_results(results)  

        return atoms

    def process_results(self, results):
        
        atoms = []
        for result in results:
            atoms.append([])
            sentences = result.split('\n')
            for sentence in sentences:
                if sentence[:2] != '- ':continue
                if len(sentence[2:])<5:continue #TODO: include better processing of atoms (e.g., check for verbs etc)
                atoms[-1].append(sentence[2:])

        return atoms

    def create_prompts(self, paragraphs):

        prompts = []
        for paragraph in paragraphs:
            #prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n'+ATOMIC_FACT_INSTRUCTION + '\n\n'
            prompt = ATOMIC_FACT_INSTRUCTION + '\n\n'
            # for line in self.demos:
            #     prompt+=line

            prompt+='\n\n'+'Please breakdown the following paragraph into independent facts:'
            
            for sentence in paragraph:
                prompt+=sentence+' '
            #prompt+='\n<|start_header_id|>assistant<|end_header_id|>'

            try:
                prompt = prompt_templates[self.model].format(prompt)
            except KeyError:
                None

            prompts.append(prompt)

        return prompts

def detect_initials(text):
    pattern = r'[A-Z]\. ?[A-Z]\.'
    match = re.findall(pattern, text)
    return [m for m in match]

def fix_sentence_splitter(curr_sentences, initials):
    """Fix sentence splitter issues."""
    for initial in initials:
        alpha1, alpha2 = None, None
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split('.') if t.strip()]

        if alpha1 and alpha2:
            for i, (sent1, sent2) in enumerate(
                zip(curr_sentences, curr_sentences[1:])
            ):
                if sent1.endswith(alpha1 + '.') and sent2.startswith(alpha2 + '.'):
                    # merge sentence i and i+1
                    curr_sentences = (
                        curr_sentences[:i]
                        + [curr_sentences[i] + ' ' + curr_sentences[i + 1]]
                        + curr_sentences[i + 2 :]
                    )
                break

    sentences, combine_with_previous = [], None

    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split()) <= 1 and sent_idx == 0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split()) <= 1:
            assert sent_idx > 0
            sentences[-1] += ' ' + sent
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += ' ' + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += ' ' + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)

    return sentences

if __name__ == '__main__':
    generator = AtomicFactGenerator(
        model='meta-llama/llama-3-70b-instruct' # 'meta-llama/llama-3-70b-instruct' problematic (though not when using prompt lab)
    )
    # atomic_facts = generator.run(
    #     ['Santa Clara, CA, Full-time, Linux kernel - Virtualization engineer at MIT.' 
    #      ' We are looking for talented embedded system software engineers with a focus on virtualization to '
    #      ' help us architect next generation hypervisor software for the Linux kernel. This '
    #      ' is a position in Santa Clara, CA. Some of the skills we look for: Technical expertise on the ARM '
    #      ' architecture, embedded virtualization, divisionot designs, Linux kernel, device drivers '
    #      ' and embedded software in general. Practical understanding and implementation of microkots .. '
    #     ]
    # )
    # print(atomic_facts)
    
    # None
