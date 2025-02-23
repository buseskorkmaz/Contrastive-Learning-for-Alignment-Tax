# echo "Starting training script..." > training_phi2.log
# echo "Process ID (PID): $$" >> training_phi2.log
# mask_rel regen_ent regen_rel swap_ent sys_lowcon toxic 
# accelerate launch --debug train_fsdp.py \
#     --model_name_or_path "microsoft/phi-2" \
#     --model_type "microsoft/phi-2" \
#     --data_path   "constrastove_training/data/processed_v7_debug" \
#     --pos_data_dir "constrastove_training/data/processed_v7_debug/pos" \
#     --neg_data_dir "constrastove_training/data/processed_v7_debug/neg" \
#     --neg_types mask_ent mask_rel regen_ent regen_rel swap_ent sys_lowcon toxic \
#     --output_dir "constrastove_training/output/training/phi2" \
#     --max_length 340 \
#     --max_neg_samples 4 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 3 \
#     --per_device_eval_batch_size 1 \
#     --warmup_steps 5 \
#     --logging_steps 1 \
#     --eval_steps 92 \
#     --save_steps 92 \
#     --weight_decay 0.00 \
#     --max_grad_norm 3.0 \
#     --report_to "wandb" \
#     --fp16 True \
#     --bf16 False \
#     --adam_epsilon 1e-6 \
#     --adam_beta1 0.9 \
#     --adam_beta2 0.999 \
#     --seed 42 

# # Get and display the PID
# PID=$!
# echo "Process started with PID: $PID"
# echo "Process ID (PID): $PID" >> training_phi2.log
# echo "To kill this process later, use: kill $PID"

echo "Starting training script..." > training_gpt2.log
echo "Process ID (PID): $$" >> training_gpt2.log

nohup accelerate launch --debug train_fsdp.py \
    --model_name_or_path "gpt2" \
    --model_type "gpt2" \
    --data_path   "constrastove_training/data/processed_v6_debug" \
    --pos_data_dir "constrastove_training/data/processed_v6_debug/pos" \
    --neg_data_dir "constrastove_training/data/processed_v6_debug/neg" \
    --neg_types mask_ent mask_rel regen_ent regen_rel swap_ent sys_lowcon toxic \
    --output_dir "constrastove_training/output/training/gpt2" \
    --max_length 512 \
    --max_neg_samples 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --num_train_epochs 4 \
    --per_device_eval_batch_size 1 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --eval_steps 3000 \
    --save_steps 3000 \
    --weight_decay 0.00 \
    --max_grad_norm 1.0 \
    --report_to "wandb" \
    --fp16 False \
    --bf16 False \
    --adam_epsilon 1e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --seed 42 >> training_gpt2.log 2>&1 &

# Get and display the PID
PID=$!
echo "Process started with PID: $PID"
echo "Process ID (PID): $PID" >> training_gpt2.log
echo "To kill this process later, use: kill $PID"


# For Phi-2
# python scripts/train_fsdp.py \
#     --model_name_or_path "microsoft/phi-2" \
#     --model_type "phi" \
#     --train_file "data/train.json" \
#     --mixed_precision true \
#     --use_bf16 true