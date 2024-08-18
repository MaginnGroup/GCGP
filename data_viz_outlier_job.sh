#!/bin/bash


#$ -q long              # Specify queue
#$ -pe smp 8                # Specify number of cores to use.
#$ -N data_viz          # Specify job name


module load python

pip install scikit-multilearn
pip install tensorflow
pip install gpflow
pip install scikit-learn
pip install scipy
pip install matplotlib
pip install pandas


export OMP_NUM_THREADS=${NSLOTS}

python  Data_Viz_and_Outlier_Check.py
