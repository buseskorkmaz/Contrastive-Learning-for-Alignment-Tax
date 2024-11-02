#PBS -l walltime=1:30:00
#PBS -l select=1:ncpus=4:mem=80gb:ngpus=1:gpu_type=A100

echo "Host - $HOSTNAME"
# echo "Commit - $(git rev-parse HEAD)"
# nvidia-smi
# nvidia-smi nvlink -c

# module load python/3.7
module purge
module load git/2.41.0-GCCcore-12.3.0-nodocs
module load git-lfs/3.2.0 
# module load tools/dev
# module load tools/prod
# module load anaconda3/personal
eval "$(~/miniconda3/bin/conda shell.bash hook)"

conda activate factual_bias_mitigation
# pip install torch torchvision torchaudio
# pip install transformers


echo "Host - $HOSTNAME"
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
# MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
# MASTER_PORT=$(shuf -i 2000-65000 -n 1)
# echo MASTER_ADDR= $MASTER_ADDR
# echo MASTER_PORT= $MASTER_PORT

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH=$HOME/miniconda3/lib/
# export NCCL_DEBUG=INFO
# # export NCCL_BLOCKING_WAIT=1
# # # export NCCL_P2P_LEVEL=NVL
# export NCCL_TIMEOUT=5400000  # Setting a longer timeout
# export NCCL_DEBUG=INFO
# export HYDRA_FULL_ERROR=1
# export NCCL_P2P_DISABLE=1
# export NCCL_P2P_LEVEL=LOC
# export TOKENIZERS_PARALLELISM=false
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

# H=`hostname`
# THEID=`echo -e $HOSTNAMES  | python -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
# echo THEID=$THEID
# pip install nvidia-cuda-nvcc-cu12
# pip uninstall flash-attn -y
# cd $HOME/FMs-at-work/flash-attention
# python setup.py install
# pip uninstall nvidia-cudnn-cu12 -y
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
# pip install --use-pep517 flash-attn --no-build-isolation

# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install nvidia-cudnn-cu12==8.9.2.26
# pip install torch torchvision torchaudio --upgrade
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install dependencies.
cd $HOME/factual-bias-mitigation/
NEG_OPTION=original
COMBO_LOG_DIR="/gpfs/home/bsk18/factual-bias-mitigation/scripts/summarization/output/logs/train/pos_all_neg_${NEG_OPTION}"
mkdir -p ${COMBO_LOG_DIR}
# python /gpfs/home/bsk18/factual-bias-mitigation/scripts/summarization/train.py --pos_data all --neg_data original
# python /gpfs/home/bsk18/factual-bias-mitigation/scripts/summarization/summarize.py --pos_data all

# python /gpfs/home/bsk18/factual-bias-mitigation/scripts/summarization/evaluate.py --model_name gpt2-tldr

# python /gpfs/home/bsk18/factual-bias-mitigation/src/contrastive_learning/index_map.py \
#     --data_dir /gpfs/home/bsk18/factual-bias-mitigation/data/tldr/train \
#     --pos_data_options original bt  \
#     --neg_data_options original \
#     --splits train \
#     --output_dir /gpfs/home/bsk18/factual-bias-mitigation/data/tldr/indices/pos_all_neg_original

# python /gpfs/home/bsk18/factual-bias-mitigation/src/contrastive_learning/index_map.py \
#     --data_dir /gpfs/home/bsk18/factual-bias-mitigation/data/tldr/validation \
#     --pos_data_options original \
#     --neg_data_options original \
#     --splits validation \
#     --output_dir /gpfs/home/bsk18/factual-bias-mitigation/data/tldr/indices/pos_all_neg_original

# python /gpfs/home/bsk18/factual-bias-mitigation/src/contrastive_learning/index_map.py \
#     --data_dir /gpfs/home/bsk18/factual-bias-mitigation/data/tldr/test \
#     --pos_data_options original \
#     --neg_data_options original \
#     --splits test \
#     --output_dir /gpfs/home/bsk18/factual-bias-mitigation/data/tldr/indices/pos_all_neg_original

# python /gpfs/home/bsk18/factual-bias-mitigation/scripts/summarization/contrastive_train.py --pos_data all --neg_data original