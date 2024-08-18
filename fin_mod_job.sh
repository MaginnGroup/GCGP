#!/bin/bash


#$ -q *@@maginn              # Specify queue
#$ -pe smp 64                 # Specify number of cores to use.
#$ -N fin_mod_imp           # Specify job name


module load python

pip install scikit-multilearn
pip install tensorflow
pip install gpflow
pip install scikit-learn
pip install scipy
pip install matplotlib
pip install pandas



export OMP_NUM_THREADS=${NSLOTS}

python  final_model_imp.py



