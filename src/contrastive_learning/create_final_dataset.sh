#!/bin/bash

# python src/utils/improved_create_final_splits.py \
#     --data_dir constrastove_training/data/processed_filled \
#     --output_dir constrastove_training/data/processed_v6_debug \
#     --tokenizer_names gpt2 \
#     --neg_aug_types swap_ent mask_ent mask_rel regen_ent regen_rel toxic\
#     --splits_ratio 0.9 0.05 0.05 \
#     --spacy_model en_core_web_sm \
#     --pos_aug_types original back_translation 

python src/utils/improved_create_final_splits.py \
    --data_dir constrastove_training/data/processed_filled \
    --output_dir constrastove_training/data/processed_v8_debug \
    --tokenizer_names meta-llama/Llama-2-7b-hf \
    --neg_aug_types swap_ent mask_ent mask_rel regen_ent regen_rel toxic sys_lowcon\
    --splits_ratio 0.9 0.05 0.05 \
    --spacy_model en_core_web_sm \
    --pos_aug_types original back_translation 