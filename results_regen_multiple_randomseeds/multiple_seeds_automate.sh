#!/bin/bash
for k in $(cat labels_multiple_seeds.txt)
do

mkdir "$k"
cd    "$k"



PROP_flag=$(echo "$k" | awk -F'[_]' '{print $1;}')
SEED_flag=$(echo "$k" | awk -F'[_]' '{print $2;}')




cp ../"$PROP_flag"_prediction_data.csv .
cp ../multiple_seeds.py  ../multiple_seeds_job.sh .


sed -i "s/prop_ID/""$PROP_flag""/g" multiple_seeds.py
sed -i "s/SEED_ID/""$SEED_flag""/g" multiple_seeds.py


sed -i "s/FILE_INDEX/""$k""/g" multiple_seeds_job.sh

qsub multiple_seeds_job.sh

cd ..

done
