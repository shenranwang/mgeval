#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=8GB
#SBATCH --output=logs/evaluate_test_samples_%j.out

echo "Hello $USER! You are on node $HOSTNAME. The time is $(date)."

module purge
module load mamba
source activate mgeval

midi_path=$1
outfile=$2
echo 'Setting $midi_path to = '
echo $midi_path
echo 'Setting $outfile to = '
echo $outfile

python . --set1dir $midi_path --set2dir ../data/augmented/ComMU/test --outfile $outfile --num-bar 8
conda deactivate