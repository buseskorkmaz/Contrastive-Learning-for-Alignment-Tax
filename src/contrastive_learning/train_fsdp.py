# train_fsdp.py is a script to train a contrastive model with FSDP (Fully Sharded Data Parallelism) using the Accelerate library.
import os
import sys
import torch
import logging
import warnings
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timedelta

import transformers
from transformers import (
    HfArgumentParser,
    AutoConfig,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from accelerate import Accelerator
from accelerate.utils import set_seed

from peft import LoraConfig, TaskType

from src.data.improved_dataset import prepare_filtered_data_for_training
from src.training.trainer import ContrastiveTrainer
from accelerate.logging import get_logger
from src.models.model_utils import create_contrastive_model, ContrastiveConfig
from accelerate import InitProcessGroupKwargs

logger = get_logger(__name__)

# Will error if the minimal version of transformers is not installed
check_min_version("4.34.0")
require_version("datasets>=2.14.0", "To fix: pip install -r requirements.txt")

@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(
        default="llama",
        metadata={"help": "Model type (llama, gpt2, phi)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    
    # LoRA parameters
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    
    # Quantization parameters
    use_4bit: bool = field(
        default=False,
        metadata={"help": "Load model in 4bit precision"}
    )
    use_nested_quant: bool = field(
        default=False,
        metadata={"help": "Use nested quantization for 4bit models"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit models"}
    )

@dataclass
class DataArguments:
    """Arguments for data processing"""
    data_path: str = field(
        metadata={"help": "Base directory containing processed data"}
    )
    pos_data_dir: str = field(
        metadata={"help": "Directory containing positive examples"}
    )
    neg_data_dir: str = field(
        metadata={"help": "Directory containing negative examples"}
    )
    cache_data_dir: str = field(
        default="cache/training",
        metadata={"help": "Directory for caching processed data"}
    )
    neg_types: List[str] = field(
        default_factory=lambda: ["swap_ent"],
        metadata={"help": "Types of negative examples to use"}
    )
    max_length: int = field(
        default=325,
        metadata={"help": "Maximum sequence length"}
    )
    max_neg_samples: int = field(
        default=5,
        metadata={"help": "Maximum number of negative samples per positive"}
    )
    validation_split: str = field(
        default="validation",
        metadata={"help": "Split to use for validation (train/validation/test)"}
    )

@dataclass
class TrainingArguments:
    """Arguments for training configuration with optimization parameters"""
    # Existing parameters
    output_dir: str = field(
        metadata={"help": "Directory for saving outputs"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Training batch size per device"}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Evaluation batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of steps for gradient accumulation"}
    )
    
    # Adding optimization-related parameters
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Initial learning rate"}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay coefficient for AdamW optimizer"}
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Epsilon for AdamW optimizer"}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for clipping"}
    )
    
    # Learning rate schedule parameters
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of warmup steps for learning rate scheduler"}
    )
    
    # Logging and saving parameters
    logging_steps: int = field(
        default=1,
        metadata={"help": "Number of steps between logging"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Number of steps between evaluations"}
    )
    save_steps: int = field(
        default=1000,
        metadata={"help": "Number of steps between saving checkpoints"}
    )
    
    # Other parameters
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    report_to: List[str] = field(
        default_factory=lambda: ["wandb"],
        metadata={"help": "List of platforms to report results to"}
    )
    log_with: str = field(
        default="wandb",
        metadata={"help": "Logging platform to use"}
    )
    
    # Mixed precision parameters
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 precision"}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bf16 precision"}
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the run for logging"}
    )

@dataclass
class FSDPArguments:
    """Arguments for FSDP configuration"""
    mixed_precision: bool = field(
        default=True,
        metadata={"help": "Whether to use mixed precision training"}
    )
    use_bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bfloat16 (alternative is float16)"}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Whether to offload optimizer state to CPU"}
    )
    activation_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use activation checkpointing"}
    )
    sharding_strategy: str = field(
        default="FULL_SHARD",
        metadata={"help": "FSDP sharding strategy"}
    )
    backward_prefetch: str = field(
        default="BACKWARD_POST",
        metadata={"help": "FSDP backward prefetch policy"}
    )
    min_num_params: int = field(
        default=1000,
        metadata={"help": "Minimum number of parameters for auto wrapping"}
    )

def setup_wandb(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    """Setup Weights & Biases logging"""
    import wandb
    from datetime import datetime
    run_name = f"contrastive_{model_args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run = wandb.init(
        project="contrastive-lm",
        name=run_name,
        config={
            "model_name": model_args.model_name_or_path,
            "model_type": model_args.model_type,
            "data_path": data_args.data_path,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "epochs": training_args.num_train_epochs,
            **vars(model_args),
            **vars(data_args),
            **vars(training_args)
        },
        group=f"FSDP-{datetime.now()}"
    )
    return run

def setup_4bit_model(model_args):
    """Setup 4-bit model loading configuration"""
    from transformers import BitsAndBytesConfig
    
    compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=model_args.use_nested_quant,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )
    
    return bnb_config

def setup_lora_config(model_args):
    """Setup LoRA configuration"""
    return LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

def setup_logger(output_dir: str) -> None:
    """
    Set up logging configuration for the training script.
    
    Args:
        output_dir (str): Directory where log files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file path
    log_file = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure logging format
    logging_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        format=logging_format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),  # Print to console
            logging.FileHandler(log_file)       # Save to file
        ]
    )
    
    # Suppress specific warnings that might clutter the logs
    warnings.filterwarnings(
        "ignore",
        message="FP16 is not supported on CPU; using FP32 instead"
    )
    warnings.filterwarnings(
        "ignore",
        message="torch.distributed.reduce_op is deprecated"
    )
    
    # Log initial setup information
    logger.info(f"Logging configured - saving logs to: {log_file}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")

def main():
    # Update dataset epochs during training
    class EpochCallback:
        def on_epoch_begin(self, args, state, control, **kwargs):
            if train_dataset is not None:
                train_dataset.set_epoch(state.epoch)
            if eval_dataset is not None:
                eval_dataset.set_epoch(state.epoch)

   # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, FSDPArguments))
    model_args, data_args, training_args, fsdp_args = parser.parse_args_into_dataclasses()
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=180))
    accelerator = Accelerator(kwargs_handlers=[init_process_group_kwargs], gradient_accumulation_steps=training_args.gradient_accumulation_steps)
    # accelerator = Accelerator()
    
    # Setup logging
    setup_logger(training_args.output_dir)
    logger = get_logger(__name__)
    
    # Setup wandb if enabled
    run_name = setup_wandb(model_args, data_args, training_args)
    training_args.run_name = run_name
    # print(type(run_name))
    # print(type(training_args.run_name))
    
    logger.info(f"Using accelerator config: {accelerator.state}")
    logger.info(f"Starting training with config:\n{training_args}")
    
    # Load base model config first to get hidden size
    base_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=data_args.cache_data_dir,
    )
    
    # Create ContrastiveConfig with proper hidden size
    contrastive_config = ContrastiveConfig(
        temperature=0.07,
        hidden_size=base_config.hidden_size,  # Set from base model config
        pooling="mean",
        projection_size=base_config.hidden_size,  # Usually same as hidden_size
        use_lora=model_args.use_lora,
        lora_config={
            "r": model_args.lora_r,
            "alpha": model_args.lora_alpha,
            "dropout": model_args.lora_dropout,
        } if model_args.use_lora else None
    )

    # Create model with proper config
    model, tokenizer = create_contrastive_model(
        accelerator=accelerator,
        model_name=model_args.model_type,
        logger=logger,
        pretrained_path=model_args.model_name_or_path,
        contrastive_config=contrastive_config,
        # torch_dtype=torch.bfloat16 if fsdp_args.use_bf16 else torch.float16,
        torch_dtype=torch.float16 if training_args.fp16 else torch.float32,
        cache_dir=data_args.cache_data_dir,
    )
    
    # Ensure model knows about max sequence length
    # model.base_model.config.max_position_embeddings = data_args.max_length

    # Make sure model knows about max sequence length
    if hasattr(model.config, "max_position_embeddings"):
        actual_max_length = min(data_args.max_length, model.config.max_position_embeddings)
        logger.info(f"Using max sequence length of {actual_max_length}")
        data_args.max_length = actual_max_length
    
    # Prepare datasets and collator with consistent max_length
    # train_dataset, eval_dataset, data_collator = prepare_data_for_training(
    #     data_args,
    #     model_args.model_name_or_path,
    #     tokenizer,
    #     accelerator,
    #     logger
    # )

    train_dataset, eval_dataset, data_collator = prepare_filtered_data_for_training(
        data_args=data_args,
        training_args=training_args,
        model_name_or_path= model_args.model_name_or_path,
        tokenizer=tokenizer,
        accelerator=accelerator,
        logger=logger,
        max_eval_samples=32,
        seed=training_args.seed
    )

    # Initialize trainer with the contrastive model
    trainer = ContrastiveTrainer(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        config=training_args,
        max_length=data_args.max_length,
        callbacks=[EpochCallback()],
        wandb_run=run_name
    )
    
    # Train
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise e
    finally:
        if accelerator.is_main_process:
            logger.info("*** Saving final model due to an error***")
            # Save the contrastive model properly
            unwrapped_model = accelerator.unwrap_model(trainer.model)
            # Save the base model
            unwrapped_model.base_model.save_pretrained(
                training_args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            # Save the classification heads and config
            torch.save(
                {
                    'classification_heads': unwrapped_model.classification_heads.state_dict(),
                    'contrastive_config': contrastive_config
                },
                os.path.join(training_args.output_dir, "contrastive_model.bin")
            )
            tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()