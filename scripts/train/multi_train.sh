#!/bin/bash

# # Base directory for logs
# LOG_DIR="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/train/logs/inlp"

# # Python script to run
# PYTHON_SCRIPT="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/train/train.py"

# # Data options
# POS_DATA_OPTIONS=("all")
# NEG_DATA_OPTIONS=("all")

# # Function to generate combinations
# generate_combinations() {
#     local -n options=$1
#     local result=()
#     for opt in "${options[@]}"; do
#         result+=("$opt")
#         if [ "$opt" != "all" ] && [ "$opt" != "original" ]; then
#             result+=("original $opt")
#         fi
#     done
#     echo "${result[@]}"
# }

# # Generate combinations
# POS_COMBINATIONS=($(generate_combinations POS_DATA_OPTIONS))
# NEG_COMBINATIONS=($(generate_combinations NEG_DATA_OPTIONS))

# # Loop through all combinations
# for pos_opt in "${POS_COMBINATIONS[@]}"; do
#     for neg_opt in "${NEG_COMBINATIONS[@]}"; do
#         # Create a unique log directory for this combination
#         pos_name=${pos_opt// /_}
#         neg_name=${neg_opt// /_}
#         COMBO_LOG_DIR="${LOG_DIR}/${pos_name}_${neg_name}"
#         mkdir -p "$COMBO_LOG_DIR"

#         # Construct the job submission command
#         JOB_CMD="jbsub -queue x86_1h -mem 80g -require a100_80gb -cores 4+1 \
#                  -e ${COMBO_LOG_DIR}/error.log \
#                  -o ${COMBO_LOG_DIR}/output.log \
#                  python ${PYTHON_SCRIPT}"

#         # Add positive data options
#         for opt in $pos_opt; do
#             JOB_CMD+=" --pos_data $opt"
#         done

#         # Add negative data options
#         for opt in $neg_opt; do
#             JOB_CMD+=" --neg_data $opt"
#         done

#         # Submit the job
#         echo "Submitting job: $JOB_CMD"
#         eval $JOB_CMD

#         # Optional: add a small delay between job submissions
#         sleep 2
#     done
# done

# echo "All jobs submitted."

# MODEL_NAME="instructive_debiasing-phi2"

# COMBO_LOG_DIR="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/TruthfulQA/output/logs/${MODEL_NAME}"
# mkdir -p "${COMBO_LOG_DIR}"
# mkdir -p "/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/TruthfulQA/output/${MODEL_NAME}"

# PYTHON_SCRIPT="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/TruthfulQA/truthfulqa/evaluate.py"

# JOB_CMD="jbsub -queue x86_6h -mem 80g -require a100_80gb -cores 4+1 \
#                  -e ${COMBO_LOG_DIR}/error.log \
#                  -o ${COMBO_LOG_DIR}/output.log \
#                  python ${PYTHON_SCRIPT} \
#                  --models ${MODEL_NAME} \
#                  --metrics mc bleu rouge judge info \
#                  --device 0 \
#                  --input_path /dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/TruthfulQA/data/v0/TruthfulQA.csv \
#                  --output_path /dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/TruthfulQA/output/${MODEL_NAME}/answers.csv"

# echo "Submitting job: $JOB_CMD"
# eval $JOB_CMD

# COMBO_LOG_DIR="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/summarization/output/logs/analysis"

# PYTHON_SCRIPT="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/summarization/analysis.py"

# JOB_CMD="jbsub -queue x86_1h -mem 20g -cores 4 \
#                  -e ${COMBO_LOG_DIR}/error.log \
#                  -o ${COMBO_LOG_DIR}/output.log \
#                  python ${PYTHON_SCRIPT}"

# echo "Submitting job: $JOB_CMD"
# eval $JOB_CMD

MODEL_NAMES=(
    "gpt2" "phi2"
    "llama2-7b_cda_race" "llama2-7b_cda_gender" "llama2-7b_cda_religion" 
    "llama2-7b_dropout" "phi2_dropout" 
    "gpt2_cda_race" "gpt2_cda_religion" "gpt2_cda_gender"
    "gpt2_dropout" "instructive_debiasing-gpt2" "instructive_debiasing-llama2" "instructive_debiasing-phi2"
    "phi2_cda_race" "phi2_cda_gender" "phi2_cda_religion"
)

# MODEL_NAMES=(
#    "llama2-7b"
# )

PYTHON_SCRIPT="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/mmlu/evaluate_vllm.py"
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    COMBO_LOG_DIR="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/mmlu/logs/${MODEL_NAME}"
    mkdir -p ${COMBO_LOG_DIR}
    JOB_CMD="jbsub -queue x86_6h -mem 80g -cores 4+1 -require a100_80gb \
                    -e ${COMBO_LOG_DIR}/error.log \
                    -o ${COMBO_LOG_DIR}/output.log \
                    python ${PYTHON_SCRIPT} \
                    --model ${MODEL_NAME} \
                    --ntrain 2 \
                    --global_record_file eval_record_collection_${MODEL_NAME}.csv"
        
        echo "Submitting job: $JOB_CMD"
        eval $JOB_CMD
done
echo "All jobs submitted."

# List of all model names
# MODEL_NAMES=(
#     "self_debiasing-phi2" "self_debiasing-llama2" "instructive_debiasing-phi2"
#     "instructive_debiasing-llama2" 
#     "llama2-7b" "phi2"
#     "llama2-7b_cda_race" "llama2-7b_cda_gender" "llama2-7b_cda_religion"
#     "phi2_cda_race" "phi2_cda_gender" "phi2_cda_religion"
#     "llama2-7b_dropout" "phi2_dropout" 
# )

# MODEL_NAMES=("instructive_debiasing-phi2" "phi2")

# # Python script path
# PYTHON_SCRIPT="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/summarization/evaluate.py"

# # Loop through all model names and submit jobs
# for MODEL_NAME in "${MODEL_NAMES[@]}"; do
#     COMBO_LOG_DIR="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/summarization/output/logs/${MODEL_NAME}"
#     mkdir -p ${COMBO_LOG_DIR}

#     JOB_CMD="jbsub -queue x86_6h -mem 40g -cores 4+1 \
#                     -e ${COMBO_LOG_DIR}/error.log \
#                     -o ${COMBO_LOG_DIR}/output.log \
#                     python ${PYTHON_SCRIPT} \
#                     --model_name ${MODEL_NAME}"

#     echo "Submitting job for model: ${MODEL_NAME}"
#     echo "Command: $JOB_CMD"
#     eval $JOB_CMD
# done

# echo "All jobs submitted."

# MODEL_NAMES=(
#     "self_debiasing-phi2" "self_debiasing-llama2" "instructive_debiasing-phi2"
#     )
# PYTHON_SCRIPT="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/realtoxicityprompts/evaluate.py"

# for MODEL_NAME in "${MODEL_NAMES[@]}"; do
#     COMBO_LOG_DIR="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/realtoxicityprompts/logs/output/${MODEL_NAME}"
#     mkdir -p ${COMBO_LOG_DIR}
#     JOB_CMD="jbsub -queue x86_6h -mem 40g -cores 4+1 \
#                     -e ${COMBO_LOG_DIR}/error.log \
#                     -o ${COMBO_LOG_DIR}/output.log \
#                     python ${PYTHON_SCRIPT} \
#                     --models $MODEL_NAME"

#     echo "Submitting job: $JOB_CMD"
#     eval $JOB_CMD
# done

# echo "All jobs submitted."

# MODEL_NAME="meta-llama/Llama-2-7b-hf"
# MODEL="LlamaForCausalLM"
# COMBO_LOG_DIR="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/seat/logs/llama2"

# PYTHON_SCRIPT="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/seat/run_seat.py"

# JOB_CMD="jbsub -queue x86_6h -mem 40g -cores 4+1 \
#                  -e ${COMBO_LOG_DIR}/error.log \
#                  -o ${COMBO_LOG_DIR}/output.log \
#                  python ${PYTHON_SCRIPT} \
#                  --model ${MODEL} \
#                  --model_name_or_path ${MODEL_NAME}"

# echo "Submitting job: $JOB_CMD"
# eval $JOB_CMD

# PYTHON_SCRIPT=/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/data/cda/positive_bt.py
# COMBO_LOG_DIR="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/summarization/output/logs/train/pos_aug"
# mkdir -p ${COMBO_LOG_DIR}
# JOB_CMD="jbsub -queue x86_6h -mem 40g -cores 4+1 \
#                -e ${COMBO_LOG_DIR}/error.log \
#                -o ${COMBO_LOG_DIR}/output.log \
#                python ${PYTHON_SCRIPT} \
#                --input_dir /dccstor/autofair/busekorkmaz/factual-bias-mitigation/data/tldr/pos \
#                --output_dir /dccstor/autofair/busekorkmaz/factual-bias-mitigation/data/tldr/pos_bt"
   
# echo "Submitting job: $JOB_CMD"
# eval $JOB_CMD

# PYTHON_SCRIPT=/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/summarization/train.py
# NEG_OPTION=all
# COMBO_LOG_DIR="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/summarization/output/logs/train/pos_all_neg_${NEG_OPTION}"
# mkdir -p ${COMBO_LOG_DIR}
# JOB_CMD="jbsub -queue x86_6h -mem 80g -require a100_80gb -cores 4+1 \
#                -e ${COMBO_LOG_DIR}/error.log \
#                -o ${COMBO_LOG_DIR}/output.log \
#                python ${PYTHON_SCRIPT} \
#                --pos_data all \
#                --neg_data ${NEG_OPTION}"
   
# echo "Submitting job: $JOB_CMD"
# eval $JOB_CMD