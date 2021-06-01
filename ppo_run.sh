#!/bin/bash
#SBATCH --account=def-mlefsrud         
#SBATCH --cpus-per-task=6                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=0:30:00                   # The job will run for 30 minutes
#SBATCH --output=joboutput.txt  # Write the log in $SCRATCH
#SBATCH --error=joberror.txt  # Write the log in $SCRATCH


# 1. Create your environement locally
module load python/3.7

virtualenv --no-download $SLURM_TMPDIR/plantrl

source $SLURM_TMPDIR/plantrl/bin/activate

pip install --no-index torch torchvision   # why is this not in virtualenv
pip install absl-py
pip install --user gym
pip install --no-index -r requirements.txt

# 2. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR

python main.py --env-name "GrowSpaceEnv-Images-v0" --custom-gym growspace --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 75 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01

# 5. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR/runtest1 $SCRATCH
