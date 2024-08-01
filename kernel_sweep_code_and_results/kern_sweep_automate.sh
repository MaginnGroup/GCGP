#!/bin/bash
for k in $(cat labels_kernel_sweep.txt)
do

mkdir "$k"
cd    "$k"


PROP_flag=$(echo "$k" | awk -F'[_]' '{print $1;}')
METHOD_flag=$(echo "$k" | awk -F'[_]' '{print $2;}')
KERNEL_flag=$(echo "$k" | awk -F'[_]' '{print $3;}')
ANISOTROPY_flag=$(echo "$k" | awk -F'[_]' '{print $4;}')



cp ../"$PROP_flag"_prediction_data.csv .
cp ../kern_sweep.py  ../kern_sweep_job.sh .


sed -i "s/prop_ID/""$PROP_flag""/g" kern_sweep.py
sed -i "s/method_ID/""$METHOD_flag""/g" kern_sweep.py
sed -i "s/kern_ID/""$KERNEL_flag""/g" kern_sweep.py
sed -i "s/anisotropy_ID/""$ANISOTROPY_flag""/g" kern_sweep.py

sed -i "s/FILE_INDEX/""$k""/g" kern_sweep_job.sh

qsub kern_sweep_job.sh

cd ..

done
