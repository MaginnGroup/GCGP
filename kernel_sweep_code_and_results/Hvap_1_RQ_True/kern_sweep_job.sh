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

/usr/bin/python3.9  kern_sweep.py

cp "model_summary_Hvap_1_RQ_True.txt" "/scratch365/bagbodek/GCGP_clone_05_02_2024/GCGP_v01/kernel_sweep/kernel_sweep_all_results/"
