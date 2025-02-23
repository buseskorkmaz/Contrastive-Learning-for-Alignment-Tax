# from dataclasses import dataclass
# from typing import Dict, Any, Optional, List
# import torch
# from torch.distributed.fsdp import (
#     ShardingStrategy,
#     BackwardPrefetch,
#     MixedPrecision,
#     CPUOffload,
# )
# from torch.distributed.fsdp.wrap import (
#     transformer_auto_wrap_policy,
#     size_based_auto_wrap_policy,
# )

# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# @dataclass
# class ModelFSDPConfig:
#     """Model-specific FSDP configuration"""
#     transformer_layer_cls: List[str]  # Names of transformer layer classes to wrap
#     activation_checkpointing_policy: Dict[str, Any]  # Layer types for activation checkpointing
#     min_num_params: int = 1e6
#     mixed_precision_policy: Optional[MixedPrecision] = None
#     sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
#     backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_POST
#     cpu_offload: Optional[CPUOffload] = None
#     device_id: Optional[int] = None

# class FSDPConfigs:
#     """FSDP configurations for different model architectures"""
    
#     @staticmethod
#     def get_llama_config(
#         mixed_precision: bool = True,
#         use_bf16: bool = True,
#         cpu_offload: bool = False,
#         device_id: Optional[int] = None,
#     ) -> ModelFSDPConfig:
#         """Get FSDP configuration for LLaMA model"""
        
#         # Mixed precision settings optimized for LLaMA
#         if mixed_precision:
#             dtype = torch.bfloat16 if use_bf16 else torch.float16
#             mixed_precision_policy = MixedPrecision(
#                 param_dtype=dtype,
#                 reduce_dtype=dtype,
#                 buffer_dtype=dtype,
#             )
#         else:
#             mixed_precision_policy = None

#         return ModelFSDPConfig(
#             transformer_layer_cls=["LlamaDecoderLayer"],
#             activation_checkpointing_policy={
#                 "block_cls": ["LlamaDecoderLayer"],
#                 "kwargs": {
#                     "preserve_rng_state": False,
#                 }
#             },
#             min_num_params=1e8,  # Higher threshold for LLaMA due to larger layer sizes
#             mixed_precision_policy=mixed_precision_policy,
#             sharding_strategy=ShardingStrategy.FULL_SHARD,
#             backward_prefetch=BackwardPrefetch.BACKWARD_POST,
#             cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
#             device_id=device_id,
#         )

#     @staticmethod
#     def get_gpt2_config(
#         mixed_precision: bool = False,
#         use_bf16: bool = False,  # GPT-2 works better with fp16
#         cpu_offload: bool = False,
#         device_id: Optional[int] = None,
#     ) -> ModelFSDPConfig:
#         """Get FSDP configuration for GPT-2 model"""
        
#         if mixed_precision:
#             dtype = torch.bfloat16 if use_bf16 else torch.float16
#             mixed_precision_policy = MixedPrecision(
#                 param_dtype=dtype,
#                 reduce_dtype=dtype,
#                 buffer_dtype=dtype,
#             )
#         else:
#             mixed_precision_policy = None

#         return ModelFSDPConfig(
#             transformer_layer_cls=["GPT2Block"],
#             activation_checkpointing_policy={
#                 "block_cls": ["GPT2Block"],
#                 "kwargs": {
#                     "preserve_rng_state": False,
#                 }
#             },
#             min_num_params=5e7,  # Lower threshold for GPT-2
#             mixed_precision_policy=mixed_precision_policy,
#             sharding_strategy=ShardingStrategy.FULL_SHARD,
#             backward_prefetch=BackwardPrefetch.BACKWARD_POST,
#             cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
#             device_id=device_id,
#         )

#     @staticmethod
#     def get_phi2_config(
#         mixed_precision: bool = True,
#         use_bf16: bool = True,
#         cpu_offload: bool = False,
#         device_id: Optional[int] = None,
#     ) -> ModelFSDPConfig:
#         """Get FSDP configuration for Phi-2 model"""
        
#         if mixed_precision:
#             dtype = torch.bfloat16 if use_bf16 else torch.float16
#             mixed_precision_policy = MixedPrecision(
#                 param_dtype=dtype,
#                 reduce_dtype=dtype,
#                 buffer_dtype=dtype,
#             )
#         else:
#             mixed_precision_policy = None

#         return ModelFSDPConfig(
#             transformer_layer_cls=["PhiDecoderLayer"],
#             activation_checkpointing_policy={
#                 "block_cls": ["PhiDecoderLayer"],
#                 "kwargs": {
#                     "preserve_rng_state": False,
#                 }
#             },
#             min_num_params=8e7,
#             mixed_precision_policy=mixed_precision_policy,
#             sharding_strategy=ShardingStrategy.FULL_SHARD,
#             backward_prefetch=BackwardPrefetch.BACKWARD_POST,
#             cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
#             device_id=device_id,
#         )

# def apply_fsdp_wrapper(model, config: ModelFSDPConfig):
#     """Apply FSDP wrapping to model with activation checkpointing"""
#     from torch.distributed.fsdp.wrap import wrap, size_based_auto_wrap_policy
#     from torch.utils.checkpoint import checkpoint
#     import inspect
#     import functools
    
#     # Helper function to find layer class in module hierarchy
#     def find_layer_class(model, class_name):
#         # Check in model's module
#         if hasattr(model, 'module'):
#             module = model.module
#         else:
#             module = model
            
#         # Look through all attributes of the model
#         for name, obj in inspect.getmembers(module):
#             # If we find a matching class name
#             if name == class_name:
#                 return obj
#             # Check if it's a module that might contain our class
#             elif inspect.ismodule(obj):
#                 if hasattr(obj, class_name):
#                     return getattr(obj, class_name)
        
#         # Look in the model's base module
#         base_module = inspect.getmodule(model.__class__)
#         if base_module and hasattr(base_module, class_name):
#             return getattr(base_module, class_name)
            
#         # For LLaMA specifically
#         if 'LlamaDecoderLayer' in class_name:
#             try:
#                 from transformers.models.llama.modeling_llama import LlamaDecoderLayer
#                 return LlamaDecoderLayer
#             except ImportError:
#                 pass
            
#         raise AttributeError(f"Could not find class {class_name} in model or its modules")

#     # Get transformer layer classes for wrapping
#     block_types = tuple(
#         find_layer_class(model, layer_name)
#         for layer_name in config.transformer_layer_cls
#     )
    
#     # Combine transformer layer policy with size-based policy
#     def combined_auto_wrap_policy(module, recurse, unwrapped_params: int, **kwargs):
#         # First check if it's a transformer layer
#         is_transformer = any(isinstance(module, block_type) for block_type in block_types)
#         if is_transformer:
#             return True
            
#         # If not a transformer layer, fall back to size-based policy
#         return size_based_auto_wrap_policy(
#             module=module,
#             recurse=recurse,
#             min_num_params=config.min_num_params,
#             **kwargs
#         )
    
#     # Apply activation checkpointing before FSDP wrapping
#     def check_fn(submodule):
#         return any(
#             isinstance(submodule, find_layer_class(model, block_name))
#             for block_name in config.activation_checkpointing_policy["block_cls"]
#         )
    
#     if config.activation_checkpointing_policy:
#         for module in model.modules():
#             if check_fn(module):
#                 # Create a wrapped forward method that handles the checkpoint properly
#                 original_forward = module.forward
                
#                 @functools.wraps(original_forward)
#                 def checkpointed_forward(*args, **kwargs):
#                     def custom_forward(*custom_args):
#                         return original_forward(*custom_args, **kwargs)
                    
#                     if len(args) == 0:
#                         raise ValueError("No arguments provided to forward method")
                        
#                     return checkpoint(
#                         custom_forward,
#                         *args,
#                         **config.activation_checkpointing_policy["kwargs"]
#                     )
                
#                 module.forward = checkpointed_forward
    
#     # Initialize FSDP wrapped model
#     fsdp_model = FSDP(
#         model,
#         auto_wrap_policy=combined_auto_wrap_policy,
#         mixed_precision=config.mixed_precision_policy,
#         sharding_strategy=config.sharding_strategy,
#         backward_prefetch=config.backward_prefetch,
#         cpu_offload=config.cpu_offload,
#         device_id=config.device_id,
#     )
    
#     return fsdp_model