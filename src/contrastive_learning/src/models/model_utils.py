# src/models/model_utils.py contains utility functions for creating and managing contrastive models
from typing import Optional, Union, Dict, Any
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ContrastiveConfig:
    temperature: float = 0.07
    hidden_size: int = None
    pooling: str = "mean"  # ['mean', 'cls', 'last']
    projection_size: int = None
    use_lora: bool = False
    lora_config: Optional[Dict[str, Any]] = None


from typing import Optional, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ContrastiveOutput:
    loss: torch.Tensor
    contrast_loss: Optional[torch.Tensor] = None
    ce_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    net_output: Optional[Dict] = None
    contrast_net_output: Optional[Dict] = None

class ContrastiveHead(nn.Module):
    def __init__(self, logger, hidden_size: int, projection_size: int):
        super().__init__()
        # self.layer_norm = nn.LayerNorm(hidden_size)  # Add this
        self.dense = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.dense.weight, std=0.01)  # Small but non-zero initialization
        nn.init.constant_(self.dense.bias, 0)  # Zero initialization for bias

        # self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, projection_size)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1.0 / hidden_size)  # Scaled initialization
        nn.init.constant_(self.out_proj.bias, 0)  # Zero initialization for bias

        self.logger = logger
        
        # Explicitly enable gradients
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x, **kwargs):
        # Verify input requires gradients and dtype
        if self.training and not x.requires_grad:
            self.logger.warning("ContrastiveHead input doesn't require gradients!")
            
        # Log dtype information
        self.logger.info(f"ContrastiveHead input dtype: {x.dtype}")
        self.logger.info(f"Dense layer weight dtype: {self.dense.weight.dtype}")
        
        # Ensure input has same dtype as layer weights
        if x.dtype != self.dense.weight.dtype:
            x = x.to(self.dense.weight.dtype)
            
        # x = self.dropout(x)
        x = self.dense(x)
        # x = self.layer_norm(x)  # add this line if you want to use it
        x = F.gelu(x) 
        # x = F.relu(x)
        # x = self.dropout(x)
        x = self.out_proj(x)
        # Add L2 normalization to the output
        x = F.normalize(x, p=2, dim=-1)  # This ensures unit norm along the last dimension

        return x
    
class ContrastiveModel(nn.Module):
    def __init__(
        self,
        accelerator, 
        logger,
        base_model: PreTrainedModel,
        config: ContrastiveConfig
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.logger = logger
        self.accelerator = accelerator

        # Explicitly enable base model gradients
        for param in self.base_model.parameters():
            param.requires_grad = True
        
        # Register classification heads with explicit gradient enabling
        self.classification_heads = nn.ModuleDict()
        self.register_classification_head(
            "contrast",
            num_classes=config.projection_size,
            requires_grad=True  # New parameter
        )
        
        # Handle LoRA if requested
        if config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                **config.lora_config or {}
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            
            # Ensure LoRA parameters require gradients
            for param in self.base_model.parameters():
                if 'lora' in param.name:
                    param.requires_grad = True
                
        self._tied_weights_keys = getattr(self.base_model, "_tied_weights_keys", set())

    def pool_hidden_states(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool hidden states according to config"""
        # First ensure hidden_states has expected shape [batch, seq_len, hidden]
        if len(hidden_states.shape) != 3:
            raise ValueError(f"Expected hidden_states to have 3 dimensions, got shape {hidden_states.shape}")
                
        if self.config.pooling == "mean":
            # Mean pooling with attention mask
            if attention_mask is not None:
                # Ensure proper broadcasting
                mask = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
                # Sum and divide by mask sum
                masked_sum = (hidden_states * mask).sum(dim=1)
                mask_sum = mask.sum(dim=1).clamp(min=1e-9)
                pooled = masked_sum / mask_sum
            else:
                pooled = hidden_states.mean(dim=1)
        elif self.config.pooling == "cls":
            pooled = hidden_states[:, 0]
        else:  # last
            if attention_mask is not None:
                sequence_lengths = attention_mask.sum(1) - 1
                batch_size = hidden_states.size(0)
                pooled = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
            else:
                pooled = hidden_states[:, -1]
                    
        # Ensure gradients are maintained
        if self.training:
            pooled.requires_grad_(True)
                
        return pooled  # Shape: [batch, hidden]
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        classification_head_name: Optional[str] = None,
        src_select_index: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass with shape handling for FSDP compatibility"""
        # Verify gradients are enabled during training
        if self.training:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            self.accelerator.print(f"[Rank {self.accelerator.process_index}] Forward pass trainable parameters: {trainable}")

        # Get dimensions from input_ids
        batch_size, seq_length = input_ids.size()
        
        # Log initial shapes
        # print(f"Rank {self.accelerator.process_index} Input shapes - ids: {input_ids.shape}, mask: {attention_mask.shape}")
        
        # Ensure inputs maintain their shape
        input_ids = input_ids.view(batch_size, seq_length)
        attention_mask = attention_mask.view(batch_size, seq_length)
        
        # Add debugging before base model call
        # print(f"Rank {self.accelerator.process_index} Shapes before base model - ids: {input_ids.shape}, mask: {attention_mask.shape}")

        # Debug kwargs
        # self.logger.info("kwargs being passed to base model:")
        # for k, v in kwargs.items():
        #     if isinstance(v, torch.Tensor):
        #         self.logger.info(f"{k}: shape {v.shape}, dtype {v.dtype}")
        #     else:
        #         self.logger.info(f"{k}: {v}")

        # Forward through base model 
        # print(len(input_ids))
        # print(input_ids.shape)
        # print(len(attention_mask))
        # print(attention_mask.shape)
        transformer_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        # Get hidden states
        hidden_states = transformer_outputs.hidden_states[-1]  # [B, T, hidden]
        lm_logits = transformer_outputs.logits                 # [B, T, vocab_size]
        # print(f"Rank {self.accelerator.process_index} Hidden states shape: {hidden_states.shape}")

        # Get the dimensions again in case they changed
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Verify logits shape
        expected_shape = (batch_size, seq_length, self.base_model.config.vocab_size)
        assert lm_logits.shape == expected_shape, f"Expected logits shape {expected_shape}, got {lm_logits.shape}"
        
        self.logger.info(f"LM logits shape: {lm_logits.shape}")

        contrast_output = None
        if classification_head_name == "contrast":
            # 1) Don’t pool hidden states to [B, hidden]. Keep [B, T, hidden].
            # 2) Project each token from hidden_size -> projection_size.
            B, T, H = hidden_states.shape

            # Flatten so we can apply a single linear layer over all tokens
            hidden_2d = hidden_states.reshape(B * T, H)  
            # Pass through the contrast head => shape [B*T, projection_size]
            projected_2d = self.classification_heads["contrast"](hidden_2d)
            # Reshape back => [B, T, projection_size]
            proj_size = projected_2d.shape[-1]
            contrast_output = projected_2d.view(B, T, proj_size)
            self.logger.info(f"Contrast output shape: {contrast_output.shape}")

        # Add final shape debugging
        self.logger.info("Final output shapes:")
        self.logger.info(f"lm_logits: {lm_logits.shape}")
        if contrast_output is not None:
            self.logger.info(f"contrast_output: {contrast_output.shape}")

        return lm_logits, contrast_output               
                            
    def tie_weights(self):
        """Tie weights between base model and classification heads if needed"""
        if hasattr(self.base_model, "tie_weights"):
            self.base_model.tie_weights()
    
    @property
    def device(self):
        """Get device from base model"""
        return next(self.parameters()).device
    
    def get_output_embeddings(self):
        """Get output embeddings from base model"""
        return self.base_model.get_output_embeddings()
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings for base model"""
        self.base_model.set_output_embeddings(new_embeddings)
    
    def register_classification_head(
        self,
        name: str,
        num_classes: int = None,
        inner_dim: int = None,
        requires_grad: bool = True,
        **kwargs
    ):
        """Register a classification head following BART"""
        if name in self.classification_heads:
            raise ValueError(f"Head already exists: {name}")
            
        inner_dim = inner_dim or self.config.hidden_size
        
        head = ContrastiveHead(
            logger=self.logger, 
            hidden_size=inner_dim, 
            projection_size=num_classes
        )
        
        # Explicitly enable gradients for head parameters
        for param in head.parameters():
            param.requires_grad = requires_grad
            
        self.classification_heads[name] = head

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Prepare inputs for text generation"""
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    def can_generate(self) -> bool:
        """Check if model can generate text"""
        return hasattr(self.base_model, "can_generate") and self.base_model.can_generate

    def __getattr__(self, name: str):
        """Forward any unknown attributes to base model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
    
    def get_normalized_probs(self, net_output, log_probs: bool = False):
        """
        Get normalized probabilities (or log probs) from a net's output.
        This implementation works with the output format of LlamaForCausalLM and similar models.
        """
        # If net_output is a tensor, use it directly
        logits = net_output if isinstance(net_output, torch.Tensor) else net_output.logits
        
        # Apply softmax to get probabilities
        if log_probs:
            # For numerical stability, we use log_softmax instead of softmax + log
            return torch.nn.functional.log_softmax(logits, dim=-1)
        else:
            return torch.nn.functional.softmax(logits, dim=-1)

def verify_gradients(model, logger, accelerator):
    """Helper to verify gradient requirements"""
    total = frozen = 0
    for name, param in model.named_parameters():
        total += 1
        if not param.requires_grad:
            frozen += 1
            logger.warning(f"Parameter {name} is frozen!")
            
    accelerator.print(f"[Rank {accelerator.process_index}] Total parameters: {total}, Frozen: {frozen}")
    if frozen > 0:
        logger.warning(f"{frozen}/{total} parameters are frozen!")

def create_contrastive_model(
    model_name: str,
    accelerator,
    logger,
    pretrained_path: str,
    contrastive_config: Optional[ContrastiveConfig] = None,
    **kwargs
) -> tuple[ContrastiveModel, PreTrainedTokenizer]:
    # Load base model and tokenizer
    dtype = kwargs.get('torch_dtype', None)
    print("WTF DTYPE", dtype)
    if dtype is None:
        if kwargs.get('fp16', False):
            dtype = torch.float16
        else:
            dtype = torch.float32
            
    logger.info(f"Using dtype: {dtype}")
    print("kwargs:", kwargs)
    # Load base model with explicit dtype
    config = AutoConfig.from_pretrained(
        pretrained_path,
        trust_remote_code=True,
    )

    # Set max position embeddings to match max_length or model's default
    if hasattr(config, "max_position_embeddings"):
        logger.info(f"Original max_position_embeddings: {config.max_position_embeddings}")
        logger.info(f"Setting max_position_embeddings to ensure proper handling")
    
    # Now load the model with this config
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        config=config,
        trust_remote_code=True,
        device_map=None,  # Important for FSDP
        **kwargs
    )
    # base_model.config.gradient_checkpointing = False 

    # Log model configuration
    logger.info(f"Model config:\n{base_model.config}")
    
    # print(base_model.config)

    base_model.train()
    # Enable gradients for all parameters explicitly
    for param in base_model.parameters():
        param.requires_grad = True
        
    # Log base model parameters before modification
    base_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    base_total = sum(p.numel() for p in base_model.parameters())
    accelerator.print(f"[Rank {accelerator.process_index}] Base model - Total parameters: {base_total:,}")
    accelerator.print(f"[Rank {accelerator.process_index}] Base model - Trainable parameters: {base_trainable:,}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_path,
        trust_remote_code=True,
        padding_side='left'
    )
    
    # Ensure padding token exists
    if tokenizer.pad_token is None:
        print("TOKENIZER PAD TOKEN IS NONE")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        base_model.generation_config.pad_token_id = base_model.generation_config.eos_token_id
        tokenizer.padding_side = 'left'
    else:
        print("TOKENIZER PAD TOKEN IS NOT NONE")

    # Get the hidden size from the base model's config
    model_hidden_size = base_model.config.hidden_size
    accelerator.print(f"[Rank {accelerator.process_index}] Model hidden size: {model_hidden_size}")
    
    # Create or update contrastive config with proper hidden size
    if contrastive_config is None:
        contrastive_config = ContrastiveConfig(
            hidden_size=model_hidden_size,
            projection_size=model_hidden_size,  # Usually same as hidden_size
            temperature=1.0,  # Add explicit temperature
            pooling="mean"     # Add explicit pooling strategy
        )
    else:
        # Update the config with the model's hidden size if not already set
        if contrastive_config.hidden_size is None:
            contrastive_config.hidden_size = model_hidden_size
        if contrastive_config.projection_size is None:
            contrastive_config.projection_size = model_hidden_size
    
   # Create the model
    model = ContrastiveModel(
        accelerator=accelerator,
        base_model=base_model,
        logger=logger,
        config=contrastive_config
    )
    
    # Convert model and classification heads to correct dtype
    model = model.to(dtype)
    for name in model.classification_heads.keys():
        model.classification_heads[name] = model.classification_heads[name].to(dtype)
        # Double-check head parameters
        for param in model.classification_heads[name].parameters():
            if param.dtype != dtype:
                param.data = param.data.to(dtype)
    
    # Verify dtype consistency
    def check_dtype(module, prefix=''):
        for name, param in module.named_parameters():
            full_name = f"{prefix}.{name}" if prefix else name
            if param.dtype != dtype:
                logger.warning(f"Parameter {full_name} has dtype {param.dtype}, expected {dtype}")
                param.data = param.data.to(dtype)
    
    check_dtype(model.base_model, 'base_model')
    for name, head in model.classification_heads.items():
        check_dtype(head, f'classification_heads.{name}')
        
    logger.info("Model dtype verification completed")
    
    # Explicitly enable gradients for classification heads
    for head in model.classification_heads.values():
        for param in head.parameters():
            param.requires_grad = True
    
    # Verify and log parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Log detailed parameter information
    logger.info("=== Model Parameter Statistics ===")
    accelerator.print(f"[Rank {accelerator.process_index}] Total parameters: {total_params:,}")
    accelerator.print(f"[Rank {accelerator.process_index}] Trainable parameters: {trainable_params:,}")
    accelerator.print(f"[Rank {accelerator.process_index}] Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # Log parameter counts by component
    logger.info("\n=== Parameter Counts by Component ===")
    logger.info("Base Model:")
    base_params = sum(p.numel() for p in model.base_model.parameters())
    base_trainable = sum(p.numel() for p in model.base_model.parameters() if p.requires_grad)
    accelerator.print(f"[Rank {accelerator.process_index}]   Total: {base_params:,}")
    accelerator.print(f"[Rank {accelerator.process_index}]   Trainable: {base_trainable:,}")
    
    logger.info("Classification Heads:")
    for name, head in model.classification_heads.items():
        head_params = sum(p.numel() for p in head.parameters())
        head_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
        accelerator.print(f"[Rank {accelerator.process_index}]   {name}:")
        accelerator.print(f"[Rank {accelerator.process_index}]     Total: {head_params:,}")
        accelerator.print(f"[Rank {accelerator.process_index}]     Trainable: {head_trainable:,}")
    
    # Verify model configuration
    logger.info("\n=== Model Configuration ===")
    accelerator.print(f"[Rank {accelerator.process_index}] Hidden size: {model_hidden_size}")
    accelerator.print(f"[Rank {accelerator.process_index}] Temperature: {contrastive_config.temperature}")
    accelerator.print(f"[Rank {accelerator.process_index}] Pooling strategy: {contrastive_config.pooling}")
    
    # Verify all components require gradients
    has_frozen = any(not p.requires_grad for p in model.parameters())
    if has_frozen:
        logger.warning("Warning: Some parameters are frozen!")

    verify_gradients(model, logger, accelerator)

    try:
        logger.info("Attempting test forward pass...")
        # Create sample input
        batch_size = 1
        seq_length = 10
        sample_input = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
        sample_mask = torch.ones_like(sample_input)
        
        # Make sure input tensors are properly shaped
        logger.info(f"Sample input shape before prepare: {sample_input.shape}")
        
        # Let accelerator handle device placement
        sample_input, sample_mask = accelerator.prepare(sample_input, sample_mask)
        logger.info(f"Sample input shape after prepare: {sample_input.shape}")
        
        # Test forward pass with no_grad
        with torch.no_grad():
            # Explicitly handle shapes
            base_kwargs = {
                'input_ids': sample_input,
                'attention_mask': sample_mask,
                'output_hidden_states': True,
                'return_dict': True
            }
            
            # Get model outputs
            outputs = model.base_model(**base_kwargs)
            
            # Log hidden states shape
            if outputs.hidden_states:
                last_hidden = outputs.hidden_states[-1]
                logger.info(f"Last hidden shape: {last_hidden.shape}")
                
            # Get logits
            ce_logits = outputs.logits
            logger.info(f"CE logits shape: {ce_logits.shape}")
            
            # Get contrast output
            contrast_output = model.classification_heads["contrast"](
                model.pool_hidden_states(last_hidden, sample_mask)
            )
            logger.info(f"Contrast output shape: {contrast_output.shape}")
            
        logger.info("Test forward pass successful!")
    except Exception as e:
        logger.error(f"Error in test forward pass: {str(e)}")
        raise

    return model, tokenizer

def log_gradient_flow(named_parameters, logger):
    # Add layer-wise gradient stats
    layer_stats = defaultdict(list)
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layer_type = n.split('.')[0]  # Get base layer type
            grad_norm = p.grad.norm().item()
            layer_stats[layer_type].append(grad_norm)
            
    for layer, norms in layer_stats.items():
        avg = sum(norms)/len(norms)
        logger.info(f"{layer} grad norms - mean: {avg:.2e}, min: {min(norms):.2e}, max: {max(norms):.2e}")
    
    return avg

def add_gradient_monitoring(model, logger):
    """Add hooks to monitor gradient flow during training"""
    def hook_fn(grad):
        if grad is None:
            return
        if torch.isnan(grad).any():
            logger.error("NaN gradient detected!")
        elif torch.isinf(grad).any():
            logger.error("Inf gradient detected!")
        
        grad_norm = grad.norm()
        if grad_norm < 1e-6:
            logger.warning(f"Very small gradient norm: {grad_norm:.2e}")
        elif grad_norm > 1:
            logger.warning(f"Very large gradient norm: {grad_norm:.2e}")
        
        return grad
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(lambda grad, name=name: hook_fn(grad))

def check_loss_scale(loss, logger):
    """Check if loss value is in a reasonable range"""
    loss_value = loss.item()
    if loss_value < 1e-6:
        logger.warning(f"Very small loss value: {loss_value:.2e}")
    elif loss_value > 100:
        logger.warning(f"Very large loss value: {loss_value:.2e}")
    return loss_value