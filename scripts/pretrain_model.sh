#!/bin/bash

QUEUE="default@pbs-m1.metacentrum.cz"
WALLTIME="24:00:00"
SELECT="1:ncpus=1:ngpus=1:mem=10gb:scratch_local=10gb:cl_grimbold=False:cl_konos=False"

CONDA_ENV="/storage/brno2/home/stepanb2/.conda/envs/cuda-new"

RUN_NAME="pretrain_1M"



declare -A COMMANDS=(
    # ["pretrain_gemma3.py"]="gemma3"
    ["pretrain_gpt2.py"]="gpt2"
    # ["pretrain_moep.py"]="moep"
    # ["pretrain_smollm2.py"]="smollm2"
)

mkdir -p scripts/pbs_outputs/$RUN_NAME


for script in "${!COMMANDS[@]}"; do

    model_name=${COMMANDS[$script]}

    mkdir -p scripts/pbs_outputs/$RUN_NAME/$model_name

    qsub <<EOF
#!/bin/bash
#PBS -q $QUEUE
#PBS -l walltime=$WALLTIME
#PBS -l select=$SELECT
#PBS -N ${RUN_NAME}_${model_name}
#PBS -o scripts/pbs_outputs/$RUN_NAME/${model_name}/stdout.txt
#PBS -e scripts/pbs_outputs/$RUN_NAME/${model_name}/stderr.txt

module add mambaforge
conda activate $CONDA_ENV

cd \$PBS_O_WORKDIR

echo "================================"
echo "Model : ${model_name}"
echo "Script: ${script}"
echo "Host  : \$(hostname)"
echo "GPU:"
nvidia-smi
echo "================================"

python experiments/${script} \
--dataset_limit_words 1_000_000 \
--run_id ${model_name}_$RUN_NAME 


clean_scratch

EOF

done