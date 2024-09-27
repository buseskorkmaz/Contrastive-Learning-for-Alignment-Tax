import argparse
import json
import os

import transformers

from seat import SEATRunner
import models
from util import generate_experiment_id

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs SEAT benchmark.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default="/gpfs/home/bsk18/bias-bench/",
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--tests",
    action="store",
    nargs="*",
    help="List of SEAT tests to run. Test files should be in `data_dir` and have "
    "corresponding names with extension .jsonl.",
)
parser.add_argument(
    "--n_samples",
    action="store",
    type=int,
    default=100000,
    help="Number of permutation test samples used when estimating p-values "
    "(exact test is used if there are fewer than this many permutations).",
)
parser.add_argument(
    "--parametric",
    action="store_true",
    help="Use parametric test (normal assumption) to compute p-values.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    # choices=["bert-base-uncased", "albert-base-v2", "roberta-base", "gpt2"],
    choices=["bert-base-uncased", "albert-base-v2", "roberta-base", "gpt2", "meta-llama/Llama-2-7b-hf", "microsoft/phi-2"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="BertModel",
    # choices=["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"],
    choices=["BertModel", "AlbertModel", "RobertaModel", "GPT2Model", "LlamaForCausalLM", "PhiForCausalLM"],
    help="Model to evalute (e.g., BertModel). Typically, these correspond to a HuggingFace "
    "class.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="seat", model=args.model, model_name_or_path=args.model_name_or_path
    )

    print("Running SEAT benchmark:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - tests: {args.tests}")
    print(f" - n_samples: {args.n_samples}")
    print(f" - parametric: {args.parametric}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")

    # Load model and tokenizer.
    model = getattr(models, args.model)(args.model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    runner = SEATRunner(
        experiment_id=experiment_id,
        tests=args.tests,
        data_dir=f"{args.persistent_dir}/data/seat",
        n_samples=args.n_samples,
        parametric=args.parametric,
        model=model,
        tokenizer=tokenizer,
    )
    results = runner()
    print(results)

    # Modified
    # Remove any slash from file experiment_id
    experiment_id = experiment_id.replace("/", "_")

    os.makedirs("/gpfs/home/bsk18/factual-bias-mitigation/scripts/seat/results/seat", exist_ok=True)
    with open(f"/gpfs/home/bsk18/factual-bias-mitigation/scripts/seat/results/seat/{experiment_id}.json", "w") as f:
        json.dump(results, f)