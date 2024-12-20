#!/bin/bash


#$ -q *@@maginn              # Specify queue
#$ -pe smp 4                # Specify number of cores to use.
#$ -N multiple_seeds          # Specify job name


module load python

pip install scikit-multilearn
pip install tensorflow
pip install gpflow
pip install scikit-learn
pip install scipy
pip install matplotlib
pip install pandas


export OMP_NUM_THREADS=${NSLOTS}

python  multiple_seeds.py

cp "model_summary_Tc_146316.txt" "/scratch365/bagbodek/GCGP_clone_05_02_2024/GCGP_copy/results_regen_multiple_randomseeds/multiple_seeds_all_results/"
