# training/trainer.py 
from typing import Optional, Dict, List, Union, Any
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, ShardedStateDictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from dataclasses import dataclass
from .loss import ContrastiveLoss, ContrastiveOutput
from tqdm import tqdm
from src.data.processor import DataProcessor, ProcessingConfig
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import wandb
import os
from torch.nn.utils.rnn import pad_sequence  # For padding variable-length reference sequences
from src.models.model_utils import add_gradient_monitoring, check_loss_scale, log_gradient_flow
from .evaluate import evaluate_test_metrics
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_state_dict

logger = get_logger(__name__)

@dataclass
class FSDPConfig:
    enabled: bool = True
    mixed_precision: bool = True
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    backward_prefetch: str = "BACKWARD_POST"  # BACKWARD_PRE, BACKWARD_POST
    cpu_offload: bool = False
    min_num_params: int = 1e6  # Min number of parameters for auto wrapping
    transformer_layer_cls_to_wrap: Optional[List[str]] = None  # e.g. ["GPTJBlock", "LlamaDecoderLayer"]

@dataclass
class TrainerConfig:
    seed: int
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    max_length: int
    warmup_steps: int
    learning_rate: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    fp16: bool = False
    bf16: bool = False
    report_to: List[str] = None
    log_with: str = "wandb"
    fsdp: Optional[FSDPConfig] = None

class ContrastiveTrainer:
    def __init__(
        self,
        accelerator: Accelerator,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        data_collator: Optional[Any] = None,
        config: Optional[TrainerConfig] = None,
        max_length: Optional[int] = None,
        callbacks: Optional[List[Any]] = None,
        wandb_run: Optional[Any] = None,
    ):
        self.config = config or TrainerConfig()

        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.callbacks = callbacks or []
        self.max_length = max_length
        self.wandb_run = wandb_run
        
        # Setup loss function
        self.criterion = ContrastiveLoss(
            accelerator,
            temperature=1.0,  # You can adjust these parameters as needed
            alpha=16.0,
            label_smoothing=0.0,
            ignore_prefix_size=0,
            # padding_idx=tokenizer.pad_token_id  # Pass the tokenizer's padding index
        )
        
        # Create dataloaders
        self.train_dataloader = self.get_train_dataloader()
        self.eval_dataloader = self.get_eval_dataloader() if eval_dataset else None
        
        # Setup optimizer and scheduler

        self.model = accelerator.prepare(self.model)
        self.optimizer, self.lr_scheduler = self.create_optimizer_and_scheduler()
        

        # Prepare everything with accelerator
        (
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler
        )
        # def check_fp16_hook(module, input, output):
        #     for name, param in module.named_parameters():
        #         if param.grad is not None:
        #             print(f"{name} param dtype: {param.dtype}, grad dtype: {param.grad.dtype}")

        # model.register_forward_hook(check_fp16_hook)

        model_dtype = torch.float32 if not self.config.fp16 else torch.float16
        print(f"Model dtype in trainer init: {model_dtype}", flush=True)
        # Now that the model has been wrapped by FSDP/Accelerate, log its trainable parameters:
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                # logger.info(f"Param {name} -> shape: {tuple(param.shape)}")

        logger.info(f"Post-prepare total trainable params: {sum(p.numel() for p in trainable_params):,}")
                
        self.state = {"epoch": 0, "step": 0}

        # Setup wandb gradient tracking if we're on the main process
        if self.accelerator.is_main_process:
            wandb.watch(
                self.model,
                log="all",  # or "gradients" to only track gradients
                log_freq=1,  # Log every 100 steps
                log_graph=True  # Log model graph
            )
        
        # Initialize data processor for evaluation
        if self.accelerator.is_main_process:
            print("Initializing DataProcessor for evaluation...", flush=True)
            self.processing_config = ProcessingConfig(
                cache_dir='',  # Empty since we don't need caching for evaluation
                toxicity_threshold=0.5,
                faithfulness_threshold=0.7,
                max_length=self.max_length,
                batch_size=self.config.per_device_eval_batch_size
            )
            self.data_processor = DataProcessor(
                config=self.processing_config, 
                augmenter=False,
                device='cpu'
            )
            print("DataProcessor initialized.", flush=True)
        else:
            self.data_processor = None
        
        """New code"""
        # Add gradient monitoring
        # add_gradient_monitoring(self.model, logger)
        
        # Initialize scale factors for loss components
        self.ce_scale = 1.0
        self.contrast_scale = 1.0
        
        # Add moving averages for loss components
        self.avg_ce_loss = 0
        self.avg_contrast_loss = 0
        self.beta = 0.9  # for moving average
        """end of new code"""

        self.accelerator.wait_for_everyone()
    
    def check_model_dtypes(self):
        """Check if all model parameters have consistent dtypes."""
        param_dtypes = set()
        param_info = []
        
        for name, param in self.model.named_parameters():
            param_dtypes.add(param.dtype)
            if param.grad is not None:
                param_info.append({
                    'name': name,
                    'param_dtype': param.dtype,
                    'grad_dtype': param.grad.dtype,
                    'requires_grad': param.requires_grad
                })

        consistent = len(param_dtypes) == 1
        dtype_str = str(next(iter(param_dtypes))) if consistent else f"Mixed: {param_dtypes}"
        
        logger.info(f"Model dtypes consistent: {consistent}, dtype: {dtype_str}")
        if not consistent:
            for info in param_info:
                logger.warning(
                    f"Parameter {info['name']}: dtype={info['param_dtype']}, "
                    f"grad_dtype={info['grad_dtype']}, requires_grad={info['requires_grad']}"
                )
            raise ValueError("Model parameters have inconsistent dtypes!")
        return consistent, dtype_str


    def save_checkpoint(self, step: int, epoch: int, metrics: Optional[Dict[str, float]] = None):
        """Save FSDP model checkpoint with sharded state dict."""
        import torch.distributed as dist
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            ShardedStateDictConfig,
        )

        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-step-{step}")
        rank = dist.get_rank()

        if rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
        dist.barrier()

        model = self.model
        while hasattr(model, "module"):
            model = model.module

        save_config = ShardedStateDictConfig(offload_to_cpu=True)

        # Save model state dict
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, save_config):
            model_state = model.state_dict()
            model_path = os.path.join(checkpoint_dir, f"pytorch_model-{step}-{rank}.bin")
            torch.save(model_state, model_path)
        del model_state
        dist.barrier()

        # Save optimizer state (sharded)
        optim_path = os.path.join(checkpoint_dir, f"optimizer-{step}-{rank}.bin")
        torch.save(self.optimizer.state_dict(), optim_path)
        
        # Save training metadata on rank 0
        if rank == 0:
            metadata = {
                "step": step,
                "epoch": epoch,
                "scheduler_state": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                "metrics": metrics or {},
                "world_size": dist.get_world_size()
            }
            torch.save(metadata, os.path.join(checkpoint_dir, "metadata.pt"))
            logger.info(f"Saved sharded checkpoint at step {step}")

        dist.barrier()

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """Create the training DataLoader with proper distributed handling."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        # Don't specify sampler - let accelerate handle it
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,  # Will be handled by accelerate if distributed
            collate_fn=self.data_collator,
            drop_last=True,  # Important for consistent batch sizes
            num_workers=0,  # Use 0 for distributed training
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=False  # Avoid memory issues
        )

    def get_eval_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        """Create the evaluation DataLoader with proper distributed handling."""
        if self.eval_dataset is None:
            return None
        
        # Don't specify sampler - let accelerate handle it
        return torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.config.per_device_eval_batch_size,
            shuffle=False,  # No shuffling needed for eval
            collate_fn=self.data_collator,
            drop_last=False,  # Keep all evaluation samples
            num_workers=0,  # Use 0 for distributed training
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=False  # Avoid memory issues
        )

    
    def create_optimizer_and_scheduler(self):
        """Create optimizer and learning rate scheduler with proper parameters"""
        # First verify we have trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found in the model!")
        
        # Separate parameters into those with and without weight decay
        decay_params = []
        no_decay_params = []
        
        # Standard LayerNorm and bias parameters shouldn't use weight decay
        for param_name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'layernorm' in param_name.lower() or 'bias' in param_name.lower():
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                # "weight_decay": self.config.weight_decay,
                "weight_decay": 0.0,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            }
        ]
        
        logger.info(f"Number of trainable parameters: {len(trainable_params)}")
        logger.info(f"Parameters with weight decay: {len(decay_params)}")
        logger.info(f"Parameters without weight decay: {len(no_decay_params)}")

        # Verify optimizer parameters exist in config
        required_attrs = ['learning_rate', 'adam_beta1', 'adam_beta2', 'adam_epsilon']
        for attr in required_attrs:
            if not hasattr(self.config, attr):
                raise ValueError(f"Missing required config attribute: {attr}")

        # Create AdamW optimizer
        try:
            # optimizer = torch.optim.AdamW(
            #     optimizer_grouped_parameters,
            #     lr=self.config.learning_rate,
            #     betas=(self.config.adam_beta1, self.config.adam_beta2),
            #     eps=self.config.adam_epsilon
            # )
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                # betas=(self.config.adam_beta1, self.config.adam_beta2),
                # eps=self.config.adam_epsilon
            )
        except Exception as e:
            logger.error(f"Failed to create optimizer: {str(e)}")
            raise
            
        # Calculate training steps
        try:
            num_update_steps_per_epoch = max(len(self.train_dataloader) // self.config.gradient_accumulation_steps, 1)
            num_training_steps = num_update_steps_per_epoch * self.config.num_train_epochs
            
            # Log training steps info
            logger.info(f"Number of update steps per epoch: {num_update_steps_per_epoch}")
            logger.info(f"Total number of training steps: {num_training_steps}")
            logger.info(f"Warmup steps: {self.config.warmup_steps}")
            
            if num_training_steps < self.config.warmup_steps:
                logger.warning(
                    f"Warmup steps ({self.config.warmup_steps}) is greater than "
                    f"total training steps ({num_training_steps})"
                )
        except Exception as e:
            logger.error(f"Failed to calculate training steps: {str(e)}")
            raise
        
        # Create learning rate scheduler
        try:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=num_training_steps
            )
            # lr_scheduler = get_constant_schedule_with_warmup(
            #     optimizer=optimizer,
            #     num_warmup_steps=self.config.warmup_steps
            # )   
        except Exception as e:
            logger.error(f"Failed to create learning rate scheduler: {str(e)}")
            raise
        
        logger.info("Successfully created optimizer and scheduler")
        return optimizer, lr_scheduler

    def evaluate(self):
        # Put model in eval mode
        self.model.eval()

        # Track scalar losses
        total_loss = 0.0
        total_contrast_loss = 0.0
        total_ce_loss = 0.0
        total_ntokens = 0

        print("Evaluating on the validation dataset...\n", flush=True)

        # 1) Evaluate losses
        for batch in tqdm(self.eval_dataloader, disable=not self.accelerator.is_local_main_process):
            if batch is None or "net_input" not in batch:
                logger.warning("Skipping invalid batch (missing 'net_input')")
                continue

            with torch.no_grad():
                # with torch.autocast('cuda', enabled=False): only for phi-2
                lm_logits, contrast_logits = self.model(
                    input_ids=batch["net_input"]["input_ids"],
                    attention_mask=batch["net_input"]["attention_mask"],
                    classification_head_name="contrast"
                )
            
                loss_output = self.criterion(
                    lm_logits=lm_logits,
                    contrast_logits=contrast_logits,
                    model=self.model,  # only needed if you rely on model.get_normalized_probs
                    sample=batch,
                    logger=logger,
                    reduce=True
                )
               
                total_loss += loss_output.loss.item() * batch["ntokens"]
                if loss_output.contrast_loss is not None:
                    total_contrast_loss += loss_output.contrast_loss.item() * batch["ntokens"]
                if loss_output.ce_loss is not None:
                    total_ce_loss += loss_output.ce_loss.item() * batch["ntokens"]
                total_ntokens += batch["ntokens"]

        # Compute average losses
        avg_loss = total_loss / total_ntokens
        avg_contrast_loss = total_contrast_loss / total_ntokens
        avg_ce_loss = total_ce_loss / total_ntokens

        metrics = {
            "eval_loss": avg_loss,
            "eval_contrast_loss": avg_contrast_loss,
            "eval_ce_loss": avg_ce_loss
        }

        # 2) Generate full outputs on each rank
        print("max_length", self.max_length, flush=True)
        print("Generating outputs for evaluation...\n", flush=True)
        local_source_list = []
        local_ref_list = []
        local_generated_list = []

        for batch in tqdm(self.eval_dataloader, disable=not self.accelerator.is_local_main_process):
            if batch is None or "net_input" not in batch:
                logger.warning("Skipping invalid batch (missing 'net_input')")
                continue
            with torch.no_grad():
                input_ids = batch["net_input"]["input_ids"]  # shape [b, src_seq_len]
                attention_mask = batch["net_input"]["attention_mask"]
                print(f"Rank {self.accelerator.process_index}: generating outputs for batch\n", flush=True)

                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=64,
                    # num_beams=1,
                    # do_sample=False,
                    repetition_penalty=1.5,
                )
                prompt_len = input_ids.size(1)
                generated_ids = generated_ids[:, prompt_len:]
                print(f"Rank {self.accelerator.process_index}: generated\n", flush=True)

                # Add source IDs
                local_source_list.append(input_ids)

                # Build references by removing negative IDs, then pad within each batch
                filtered_tensors = []
                for seq in batch["target"]:
                    filtered_seq = [x for x in seq.tolist() if x >= 0]
                    filtered_tensors.append(
                        torch.tensor(filtered_seq, dtype=torch.long, device=input_ids.device)
                    )
                if filtered_tensors:
                    ref_padded = pad_sequence(
                        filtered_tensors,
                        batch_first=True,
                        padding_value=self.tokenizer.pad_token_id
                    )
                else:
                    ref_padded = torch.empty((0, 1), dtype=torch.long, device=input_ids.device)
                local_ref_list.append(ref_padded)

                # Add generated IDs
                local_generated_list.append(generated_ids)

        print("Finished gathering local results for each batch, now padding to a consistent seq length...", flush=True)

        # 3) Pad each list element to a uniform sequence length before concatenation
        # Source IDs
        if local_source_list:
            max_seq_len_source = max(tensor.size(1) for tensor in local_source_list)
            for i, tensor in enumerate(local_source_list):
                pad_amt = max_seq_len_source - tensor.size(1)
                if pad_amt > 0:
                    local_source_list[i] = F.pad(tensor, (0, pad_amt), value=self.tokenizer.pad_token_id)
            all_source_ids_local = torch.cat(local_source_list, dim=0)
        else:
            all_source_ids_local = torch.empty((0, 1), dtype=torch.long, device=self.accelerator.device)

        # Reference IDs
        if local_ref_list:
            max_seq_len_ref = max(tensor.size(1) for tensor in local_ref_list)
            for i, tensor in enumerate(local_ref_list):
                pad_amt = max_seq_len_ref - tensor.size(1)
                if pad_amt > 0:
                    local_ref_list[i] = F.pad(tensor, (0, pad_amt), value=self.tokenizer.pad_token_id)
            all_ref_ids_local = torch.cat(local_ref_list, dim=0)
        else:
            all_ref_ids_local = torch.empty((0, 1), dtype=torch.long, device=self.accelerator.device)

        # Generated IDs
        if local_generated_list:
            max_seq_len_gen = max(tensor.size(1) for tensor in local_generated_list)
            for i, tensor in enumerate(local_generated_list):
                pad_amt = max_seq_len_gen - tensor.size(1)
                if pad_amt > 0:
                    local_generated_list[i] = F.pad(tensor, (0, pad_amt), value=self.tokenizer.pad_token_id)
            all_generated_ids_local = torch.cat(local_generated_list, dim=0)
        else:
            all_generated_ids_local = torch.empty((0, 1), dtype=torch.long, device=self.accelerator.device)

        # 4) Make sure every rank has the same local batch size (N_local).
        local_batch_size = all_source_ids_local.size(0)
        local_bs_t = torch.tensor([local_batch_size], device=self.accelerator.device)
        self.accelerator.wait_for_everyone()
        gathered_bs = self.accelerator.gather(local_bs_t)
        global_max_batch = gathered_bs.max().item()

        # Pad the batch dimension if needed (to match global_max_batch)
        def pad_batch_dim(tensor, final_size):
            curr_size = tensor.size(0)
            if curr_size < final_size:
                pad_amt = final_size - curr_size
                if len(tensor.shape) == 2:
                    seq_len = tensor.size(1)
                    new_rows = torch.full(
                        (pad_amt, seq_len),
                        self.tokenizer.pad_token_id,
                        dtype=tensor.dtype,
                        device=tensor.device
                    )
                    return torch.cat([tensor, new_rows], dim=0)
                else:
                    shape_rest = tensor.shape[1:]
                    new_rows = torch.full(
                        (pad_amt,) + shape_rest,
                        self.tokenizer.pad_token_id,
                        dtype=tensor.dtype,
                        device=tensor.device
                    )
                    return torch.cat([tensor, new_rows], dim=0)
            else:
                return tensor

        all_source_ids_local = pad_batch_dim(all_source_ids_local, global_max_batch)
        all_ref_ids_local = pad_batch_dim(all_ref_ids_local, global_max_batch)
        all_generated_ids_local = pad_batch_dim(all_generated_ids_local, global_max_batch)

        # 5) Now handle dimension 1 as well if they differ across ranks
        def local_max_seq_len(tensor):
            if tensor.size(0) == 0:
                return 0
            return tensor.size(1)

        local_max_src = local_max_seq_len(all_source_ids_local)
        local_max_ref = local_max_seq_len(all_ref_ids_local)
        local_max_gen = local_max_seq_len(all_generated_ids_local)

        local_max_src_t = torch.tensor([local_max_src], device=self.accelerator.device)
        local_max_ref_t = torch.tensor([local_max_ref], device=self.accelerator.device)
        local_max_gen_t = torch.tensor([local_max_gen], device=self.accelerator.device)
        self.accelerator.wait_for_everyone()
        gathered_max_src = self.accelerator.gather(local_max_src_t)
        gathered_max_ref = self.accelerator.gather(local_max_ref_t)
        gathered_max_gen = self.accelerator.gather(local_max_gen_t)
        final_max_src = gathered_max_src.max().item()
        final_max_ref = gathered_max_ref.max().item()
        final_max_gen = gathered_max_gen.max().item()

        print(f"Rank {self.accelerator.process_index} tokenizer.pad_token_id:", self.tokenizer.pad_token_id, flush=True)

        def pad_seq_len(tensor, final_len):
            if tensor.size(0) == 0:
                return tensor
            curr_len = tensor.size(1)
            if curr_len < final_len:
                pad_amt = final_len - curr_len
                return F.pad(tensor, (0, pad_amt), value=self.tokenizer.pad_token_id)
            else:
                return tensor

        all_source_ids_local = pad_seq_len(all_source_ids_local, final_max_src)
        all_ref_ids_local = pad_seq_len(all_ref_ids_local, final_max_ref)
        all_generated_ids_local = pad_seq_len(all_generated_ids_local, final_max_gen)

        print(f"Rank {self.accelerator.process_index} Gathering global sequences...\n", flush=True)
        self.accelerator.wait_for_everyone()

        # 6) Gather final [global_max_batch, final_max_seq] from every rank
        all_source_ids_global = self.accelerator.gather(all_source_ids_local)
        all_ref_ids_global = self.accelerator.gather(all_ref_ids_local)
        all_generated_ids_global = self.accelerator.gather(all_generated_ids_local)

        print(f"Rank {self.accelerator.process_index} Gathered global sequences...\n", flush=True)

        # 7) Decode only on rank=0
        all_source_texts = []
        all_references = []
        all_generated_outputs = []
        self.accelerator.wait_for_everyone()
        print(f"Rank {self.accelerator.process_index} Waiting all processes before batch decoding...\n", flush=True)
        self.accelerator.wait_for_everyone()
        print(f"Rank {self.accelerator.process_index} Batch decoding sources...\n", flush=True)

        all_source_ids_global = torch.clamp(all_source_ids_global, min=0, max=self.tokenizer.vocab_size - 1)
        all_ref_ids_global = torch.clamp(all_ref_ids_global, min=0, max=self.tokenizer.vocab_size - 1)
        all_generated_ids_global = torch.clamp(all_generated_ids_global, min=0, max=self.tokenizer.vocab_size - 1)

        if self.accelerator.is_main_process:
            print("Main process starting decoding...", flush=True)
            source_ids = all_source_ids_global
            ref_ids = all_ref_ids_global
            generated_ids = all_generated_ids_global
            del all_source_ids_global
            del all_ref_ids_global
            del all_generated_ids_global

            batch_size = 32
            for i in range(0, len(source_ids), batch_size):
                decoded = self.tokenizer.batch_decode(source_ids[i : i + batch_size], skip_special_tokens=True)
                all_source_texts.extend(decoded)
            for i in range(0, len(ref_ids), batch_size):
                decoded = self.tokenizer.batch_decode(ref_ids[i : i + batch_size], skip_special_tokens=True)
                all_references.extend(decoded)
            for i in range(0, len(generated_ids), batch_size):
                decoded = self.tokenizer.batch_decode(generated_ids[i : i + batch_size], skip_special_tokens=True)
                all_generated_outputs.extend(decoded)

            print("Main process completed decoding", flush=True)
            print("All source texts:\n", all_source_texts, flush=True)
            print("All generated outputs:\n", all_generated_outputs, flush=True)

            # Save outputs
            log_filename = f"generation_outputs_epoch_{self.state['epoch']}.txt"
            with open(log_filename, 'w', encoding='utf-8') as f:
                for source, generated in zip(all_source_texts, all_generated_outputs):
                    f.write(f"SOURCE: {source}\n")
                    f.write(f"GENERATED: {generated}\n")
                    f.write("-" * 80 + "\n")

            print("Main process starting scoring on GPU...", flush=True)
            examples = {"post": all_source_texts, "summary": all_generated_outputs}
            print(f"Processing {len(all_source_texts)} examples in batches...", flush=True)
            with self.accelerator.no_sync(self.model):
                toxicity_scores, faithfulness_scores = self.data_processor.score_samples(examples)

            metrics.update({
                "eval_avg_toxicity": sum(toxicity_scores) / len(toxicity_scores) if toxicity_scores else 0.0,
                "eval_avg_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
            })
            print("Main process completed GPU scoring", flush=True)
            self.accelerator.log(metrics, step=self.state["step"])
            if 'wandb' in self.config.report_to:
                self.wandb_run.log(metrics, step=self.state["step"])
                # for i in range(min(5, len(all_generated_outputs))):
                #     sample = {
                #         "source": all_source_texts[i],
                #         "reference": all_references[i],
                #         "generated": all_generated_outputs[i],
                #         "toxicity_score": toxicity_scores[i],
                #         "faithfulness_score": faithfulness_scores[i]
                #     }
                #     # self.wandb_run.log({f"eval_sample_{i}": sample}, step=self.state["step"])
        else:
            metrics = {}

        print(f"Rank {self.accelerator.process_index} completed evaluation, waiting for main process", flush=True)
        # Run test metrics evaluation
        print("Running test metrics evaluation...\n", flush=True)
        # test_metrics = evaluate_test_metrics(
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     accelerator=self.accelerator,
        #     logger=logger,
        #     state=self.state
        # )
        
        # Combine metrics
        # metrics.update(test_metrics)

        # Log all metrics to wandb if enabled
        if self.accelerator.is_main_process:
            if self.wandb_run is not None:
                # Log metrics to wandb
                self.wandb_run.log(metrics, step=self.state["step"])
                
                # If you want to also log example outputs as a table in wandb
                # if "test_generation_outputs_epoch_{self.state['epoch']}.txt" in os.listdir():
                #     with open(f"test_generation_outputs_epoch_{self.state['epoch']}.txt", 'r') as f:
                #         example_text = f.read()
                #     self.wandb_run.log({
                #         "test_examples": wandb.Table(
                #             columns=["epoch", "step", "examples"],
                #             data=[[self.state['epoch'], self.state['step'], example_text]]
                #         )
                #     }, step=self.state["step"])
            
            # Print metrics for logging
            print("\nEvaluation Results:")
            print(f"Validation Loss: {metrics['eval_loss']:.4f}")
            print(f"Validation Contrast Loss: {metrics['eval_contrast_loss']:.4f}")
            print(f"Validation CE Loss: {metrics['eval_ce_loss']:.4f}")
            # if 'test/toxicity' in metrics:
            #     print(f"Test Toxicity: {metrics['test/toxicity']:.4f}")
            # if 'test/faithfulness' in metrics:
            #     print(f"Test Faithfulness: {metrics['test/faithfulness']:.4f}")
                
        self.accelerator.wait_for_everyone()
        self.model.train()
        return metrics
    
    
    def train(self):
        """Main training loop that properly handles contrastive batch structure"""
        # Ensure model is in training mode and gradients are enabled
        self.model.train()
        
        # Debug parameter status before training
        trainable_params = sum(p.requires_grad for p in self.model.parameters())
        logger.info(f"Number of trainable parameters before training: {trainable_params}")
        
        total_steps = len(self.train_dataloader) * self.config.num_train_epochs
        progress_bar = tqdm(
            total=total_steps,
            disable=not self.accelerator.is_local_main_process
        )
        
        for epoch in range(self.config.num_train_epochs):
            self.model.train()
            
            if hasattr(self.train_dataset, "set_epoch"):
                self.train_dataset.set_epoch(epoch + 1)
                
            if hasattr(self.eval_dataset, "set_epoch"):
                self.eval_dataset.set_epoch(epoch + 1)
            
            for step, batch in enumerate(self.train_dataloader):
                # self.accelerator.wait_for_everyone()   
                # if batch.get("dummy", False):
                #     # Skip this batch since it contains no real data.
                #     continue
                # if batch is None or "net_input" not in batch:
                #     # logger.warning("Skipping invalid batch (missing 'net_input')")
                #     continue
                # Synchronize all ranks at the beginning of the iteration.
                dummy_flag = 1 if (batch is None or batch.get("dummy", False) or "net_input" not in batch) else 0
                flag_tensor = torch.tensor([dummy_flag], device=self.accelerator.device)

                # If distributed is initialized, call torch.distributed.all_reduce to sum the flag across ranks.
                if dist.is_initialized():
                    dist.all_reduce(flag_tensor, op=dist.ReduceOp.SUM)
                aggregated_flag = flag_tensor

                # If any rank had an invalid/dummy batch, aggregated_flag.item() will be > 0.
                if aggregated_flag.item() > 0:
                    continue

                # Clear any stored gradients
                # self.optimizer.zero_grad()
                total_param_bytes = 0
                total_grad_bytes = 0

                for name, param in self.model.named_parameters():
                    if param.data is not None:
                        total_param_bytes += param.numel() * param.element_size()
                    if param.grad is not None:
                        total_grad_bytes += param.grad.numel() * param.grad.element_size()

                logger.info(f"Total param memory: {total_param_bytes / 1e6:.2f} MB")
                logger.info(f"Total grad memory: {total_grad_bytes / 1e6:.2f} MB")
                # logger.info(torch.cuda.memory_summary())

                # max_id = batch["net_input"]["input_ids"].max()
                # if max_id >= self.model.base_model.config.vocab_size:
                #     logger.error("ERROR: Found out-of-range token ID:", max_id.item(),
                #         "which exceeds vocab_size:", self.model.base_model.config.vocab_size)
                #     raise

                with self.accelerator.accumulate(self.model):
                    # Debug input requires_grad
                    if any(t.requires_grad for t in batch["net_input"].values()):
                        logger.info("Input tensors require grad - this might cause issues")
                    
                    # Get model outputs with gradient tracking explicitly enabled
                    with torch.set_grad_enabled(True):
                        # with torch.autocast('cuda', enabled=False): only for phi-2
                        lm_logits,  contrast_logits = self.model(
                            input_ids=batch["net_input"]["input_ids"],
                            attention_mask=batch["net_input"]["attention_mask"],
                            classification_head_name="contrast"
                        )
                        
                        # Debug model output gradients
                        logger.info(f"Step {step}, Accumulation step: {step % self.accelerator.gradient_accumulation_steps}")
                        logger.info(f"CE output requires grad: {lm_logits.requires_grad}")
                        logger.info(f"Contrast output requires grad: {contrast_logits.requires_grad}")
                                                            
                        # Compute loss
                        loss_output = self.criterion(
                            lm_logits=lm_logits,
                            contrast_logits=contrast_logits,
                            model=self.model,  # only needed if you rely on model.get_normalized_probs
                            sample=batch,
                            logger=logger,
                            reduce=True
                        )
                        
                        # Extract losses if using ContrastiveOutput
                        if isinstance(loss_output, ContrastiveOutput):
                            loss = loss_output.loss
                            contrast_loss = loss_output.contrast_loss
                            ce_loss = loss_output.ce_loss
                        else:
                            loss = loss_output
                            contrast_loss = None
                            ce_loss = None

                        logger.info(f"Loss value before backward: {loss.item():.4f}")
                        logger.info(f"Loss requires grad: {loss.requires_grad}")
                        
                        # # Backward pass.
                        # self.accelerator.backward(loss)

                        # # Dynamic loss scaling
                        # if step > 0:  # Skip first step
                        #     if self.avg_ce_loss > 0:
                        #         self.ce_scale = min(1.0, 0.1 / self.avg_ce_loss)
                        #     if self.avg_contrast_loss > 0:
                        #         self.contrast_scale = min(1.0, 0.1 / self.avg_contrast_loss)

                        # Scale losses
                        # ce_loss = loss_output.ce_loss * self.ce_scale
                        # contrast_loss = loss_output.contrast_loss * self.contrast_scale
                        # loss = ce_loss + contrast_loss
                        
                        # Update moving averages
                        # self.avg_ce_loss = self.beta * self.avg_ce_loss + (1 - self.beta) * loss_output.ce_loss.item()
                        # self.avg_contrast_loss = self.beta * self.avg_contrast_loss + (1 - self.beta) * loss_output.contrast_loss.item()
                        
                        # Check loss scale
                        # check_loss_scale(loss, logger)

                        # Extract individual losses
                        if isinstance(loss_output, ContrastiveOutput):
                            loss = loss_output.loss
                            contrast_loss = loss_output.contrast_loss
                            ce_loss = loss_output.ce_loss
                        else:
                            loss = loss_output
                            contrast_loss = None
                            ce_loss = None
                            
                        logger.info(f"Loss value before backward: {loss.item():.4f}")
                        logger.info(f"Loss requires grad: {loss.requires_grad}")
                                            
                        # Backward pass with retained graph
                        self.accelerator.backward(loss)

                        if self.accelerator.sync_gradients:
                            # grad_norm = self.accelerator.clip_grad_norm_(
                            #     self.model.parameters(), 1
                            # )
                            # print(f"Rank {self.accelerator.process_index} Grad norm: {grad_norm}", flush=True)
                            self.check_model_dtypes()
                            # if self.accelerator.is_local_main_process:
                            #     self.wandb_run.log({"grad_norm": grad_norm})

                            # Log gradient flow before clipping
                            avg_grad_norm = log_gradient_flow(self.model.named_parameters(), logger)
                            if self.accelerator.is_main_process:
                                self.wandb_run.log({"average_grad_norm": avg_grad_norm}, step=self.state["step"])

                            norm = 0
                            for p in self.model.parameters():
                                norm += p.data.norm().item()
                            print("Param norm in evaluate:", norm)
                            
                            # Clip gradients
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.model.parameters(), 
                                self.config.max_grad_norm
                            )
                            
                            # Log post-clipping gradient norm
                            logger.info(f"Gradient norm before clipping: {grad_norm}")
                            parameters = [p for p in self.model.parameters() if p.grad is not None]
                            if parameters:  
                                post_clip_norm = torch.norm(
                                    torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2
                                ).item()
                            else:
                                post_clip_norm = 0.0

                            logger.info(f"Post-clip gradient norm: {post_clip_norm:.2e}")
                            # Additional gradient checks after clipping
                            if post_clip_norm < 1e-6:
                                logger.warning(f"Very small gradient norm after clipping: {grad_norm:.2e}")
                            elif post_clip_norm > self.config.max_grad_norm:
                                logger.warning(f"Gradient norm {grad_norm:.2e} exceeds max_grad_norm {self.config.max_grad_norm}")
                    
                            # Optimization step
                            self.optimizer.step()
                            logger.info("Optimizer step taken")
                            
                            self.lr_scheduler.step()
                    
                            # Zero gradients after step
                            self.optimizer.zero_grad()
                            logger.info("Gradients zeroed")
                        
                # Log learning rate
                current_lr = self.lr_scheduler.get_last_lr()[0]
                logger.info(f"Current learning rate: {current_lr}")
                    
                progress_bar.update(1)
                self.state["step"] += 1
                
                # Rest of your logging and evaluation code remains the same
                if self.state["step"] % self.config.logging_steps == 0:
                    log_dict = {
                        "train_loss": loss.item(),
                        "learning_rate": current_lr,
                        "epoch": epoch + 1,
                        "step": self.state["step"]
                    }
               
                    logger.info(f"CE Loss: {ce_loss.item():.4f} (scale: {self.ce_scale:.4f})")
                    logger.info(f"Contrast Loss: {contrast_loss.item():.4f} (scale: {self.contrast_scale:.4f})")
                    logger.info(f"Total Loss: {loss.item():.4f}")

                    if contrast_loss is not None:
                        log_dict["train_contrast_loss"] = contrast_loss.item()
                    if ce_loss is not None:
                        log_dict["train_ce_loss"] = ce_loss.item()
                    
                    self.accelerator.log(log_dict, step=self.state["step"])
                    
                    if self.accelerator.is_main_process:
                        self.wandb_run.log(log_dict, step=self.state["step"])
                    
                if self.state["step"] % self.config.save_steps == 0:
                    self.accelerator.wait_for_everyone()
                    logger.info("*** Saving model ***")
                    # Save the contrastive model properly
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                #     # if self.accelerator.is_main_process:
                        
                #     # Save the base model
                #     save_dir = os.path.join(self.config.output_dir, f"checkpoint-step-{self.state['step']}")
                #     print(save_dir)
                #     os.makedirs(save_dir, exist_ok=True)
                #     unwrapped_model.base_model.save_pretrained(
                #         save_dir,
                #         is_main_process=self.accelerator.is_main_process,
                #         save_function=self.accelerator.save,
                #     )
                #     # Save the classification heads and config
                #     torch.save(
                #         {
                #             'classification_heads': unwrapped_model.classification_heads.state_dict(),
                #             'contrastive_config': self.config
                #         },
                #         os.path.join(save_dir, "contrastive_model.bin")
                #     )
                #     self.tokenizer.save_pretrained(save_dir)

                #     print(f"Rank {self.accelerator.process_index} waits for saving", flush=True)
                #     # Wait again so that all ranks remain in sync after the main process finishes saving
                #     self.accelerator.wait_for_everyone()
                #     print(f"Done waiting", flush=True)

                    # Only on the main process:
                    with FSDP.summon_full_params(
                            unwrapped_model, 
                            rank0_only=True,
                            offload_to_cpu=True,
                            writeback=False,
                        ):
                        if self.accelerator.is_main_process:
                            save_dir = os.path.join(self.config.output_dir, f"checkpoint-step-{self.state['step']}")
                            os.makedirs(save_dir, exist_ok=True)

                            state_dict = unwrapped_model.state_dict()  # This is already on CPU
                            torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))

                            torch.save(
                                {
                                    'classification_heads': unwrapped_model.classification_heads.state_dict(),
                                    'contrastive_config': self.config
                                },
                                os.path.join(save_dir, "contrastive_model.bin")
                            )
                            self.tokenizer.save_pretrained(save_dir)

                            print(f"Rank {self.accelerator.process_index} finished saving.", flush=True)      

                if self.eval_dataloader and self.state["step"] % self.config.eval_steps == 0:
                    metrics = self.evaluate()
                    self.accelerator.log(metrics, step=self.state["step"])
                    if self.accelerator.is_main_process:
                        self.wandb_run.log(metrics, step=self.state["step"])
    
            self.accelerator.wait_for_everyone()
            self.state["epoch"] = epoch + 1
            self.accelerator.wait_for_everyone()
        
        self.accelerator.wait_for_everyone()
        logger.info("*** Saving final model at the end of training***")
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        with FSDP.summon_full_params(
            unwrapped_model,
            rank0_only=True,
            offload_to_cpu=True,
            writeback=False
        ):
            if self.accelerator.is_main_process:
                save_dir = os.path.join(self.config.output_dir, "alpha-16-final_checkpoint")
                os.makedirs(save_dir, exist_ok=True)

                torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
                torch.save(
                    {
                        'classification_heads': unwrapped_model.classification_heads.state_dict(),
                        'contrastive_config': self.config
                    },
                    os.path.join(save_dir, "contrastive_model.bin")
                )
                self.tokenizer.save_pretrained(save_dir)

        return self.state
