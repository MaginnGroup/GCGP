#!/bin/bash


#$ -q *@@maginn              # Specify queue
#$ -pe smp 4                # Specify number of cores to use.
#$ -N kern_sweep          # Specify job name


module load python

pip install scikit-multilearn
pip install tensorflow
pip install gpflow
pip install scikit-learn
pip install scipy
pip install matplotlib
pip install pandas


export OMP_NUM_THREADS=${NSLOTS}

python  kern_sweep.py

cp "model_summary_FILE_INDEX.txt" "/scratch365/bagbodek/GCGP_clone_05_02_2024/GCGP_v01/kernel_sweep/kernel_sweep_all_results/"
