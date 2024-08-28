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

MODEL_NAME="gpt2 inlp sentence_debiasing"

COMBO_LOG_DIR="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/TruthfulQA/output/logs/all_mc"

PYTHON_SCRIPT="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/TruthfulQA/truthfulqa/evaluate.py"

JOB_CMD="jbsub -queue x86_6h -mem 80g -require a100_80gb -cores 4+1 \
                 -e ${COMBO_LOG_DIR}/error.log \
                 -o ${COMBO_LOG_DIR}/output.log \
                 python ${PYTHON_SCRIPT} \
                 --models ${MODEL_NAME} \
                 --metrics mc bleu rouge bleurt judge info \
                 --device 0 \
                 --input_path /dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/TruthfulQA/data/v0/TruthfulQA.csv \
                 --output_path /dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/TruthfulQA/output/all_mc/answers.csv "

echo "Submitting job: $JOB_CMD"
eval $JOB_CMD