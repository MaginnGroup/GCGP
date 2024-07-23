#!/bin/bash
for k in $(cat labels_others_kernel_sweep.txt)
do

mkdir "$k"
cd    "$k"


cp ../*.csv .
cp ../kernel_sweep_crc.py  ../kernel_sweep_job.sh .



PROP_flag=$(echo "$k" | awk -F'[_]' '{print $1;}')
kern_flag=$(echo "$k" | awk -F'[_]' '{print $2;}')
START_flag=$(echo "$k" | awk -F'[_]' '{print $3;}')
STOP_flag=$(echo "$k" | awk -F'[_]' '{print $4;}')


sed -i "s/PROP_INDEX/""$PROP_flag""/g" kernel_sweep_crc.py
sed -i "s/kernelID/""$kern_flag""/g" kernel_sweep_crc.py
sed -i "s/START_INDEX/""$START_flag""/g" kernel_sweep_crc.py
sed -i "s/STOP_INDEX/""$STOP_flag""/g" kernel_sweep_crc.py



qsub kernel_sweep_job.sh

cd ..

done




