#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1

# Configuration
OUTPUT_DIR="output"
NUM_GPUS=8  # Total number of GPUs to use for each job
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Create a main log file for the entire process
MAIN_LOG="logs/main_process_${TIMESTAMP}.log"

# Augmentation types
declare -a AUG_TYPES=(
    # "pos/original"
    # "neg/swap_ent"
    # "neg/mask_ent"
    # "neg/mask_rel"
    # "pos/back_translation"
    # "neg/regen_ent"
    # "neg/regen_rel"
    neg/sys_lowcon
    # neg/toxic
)

# Function to launch a single augmentation job and wait for completion
launch_aug_job() {
    local aug_type=$1
    local timestamp=$2
    
    # Create job-specific log file
    local log_file="logs/processing_${aug_type//\//_}_${timestamp}.log"
    
    echo "Starting ${aug_type} job..." >> "$MAIN_LOG"
    
    # Launch job with nohup and store PID
    nohup torchrun \
        --nproc_per_node=8 \
        --master_port=29500 \
        process_tldr.py \
        "output.dir=${OUTPUT_DIR}" \
        "augmentation.type=${aug_type}" \
        > "$log_file" 2>&1 &
    
    local pid=$!
    echo $pid > "logs/process_${aug_type//\//_}_${timestamp}.pid"
    
    echo "Launched ${aug_type} job (PID: $pid)" >> "$MAIN_LOG"
    echo "Waiting for completion..." >> "$MAIN_LOG"
    
    # Wait for this specific PID to complete
    wait $pid
    local exit_status=$?
    
    if [ $exit_status -eq 0 ]; then
        echo "Completed ${aug_type} job successfully" >> "$MAIN_LOG"
    else
        echo "Error: ${aug_type} job failed with status ${exit_status}" >> "$MAIN_LOG"
        echo "Check log file: $log_file" >> "$MAIN_LOG"
        exit $exit_status
    fi
}

# Start the entire process with nohup
(
    echo "Starting augmentation processing at $(date)" >> "$MAIN_LOG"
    echo "----------------------------------------" >> "$MAIN_LOG"

    # Process each augmentation type sequentially
    for aug_type in "${AUG_TYPES[@]}"; do
        echo "----------------------------------------" >> "$MAIN_LOG"
        echo "Processing augmentation: ${aug_type}" >> "$MAIN_LOG"
        echo "----------------------------------------" >> "$MAIN_LOG"
        
        launch_aug_job "$aug_type" "$TIMESTAMP"
        
        echo "Finished processing ${aug_type}" >> "$MAIN_LOG"
        echo "" >> "$MAIN_LOG"
    done

    echo "All augmentation jobs completed!" >> "$MAIN_LOG"
    echo "Check individual log files in logs/ directory for details" >> "$MAIN_LOG"
) &

# Save the main process PID
echo $! > "logs/main_process_${TIMESTAMP}.pid"

echo "Main process started with PID $(cat logs/main_process_${TIMESTAMP}.pid)"
echo "Monitor main process: tail -f $MAIN_LOG"
echo "Monitor GPU usage with: watch nvidia-smi"
echo "Kill main process with: kill $(cat logs/main_process_${TIMESTAMP}.pid)"