#!/bin/bash

# Output CSV file
output_file=/scratch365/bagbodek/GCGP_clone_05_02_2024/GCGP_copy/results_regen_multiple_randomseeds/"multiple_seeds_results.csv"
echo "Property,Seed,LogMarginalLikelihood,TestMAPD,TrainMAPD,TestMAE,TrainMAE,TestRMSE,TrainRMSE,TestR2,TrainR2" > $output_file


for k in $(cat labels_multiple_seeds_Pc_Vc.txt)
do

cd    /scratch365/bagbodek/GCGP_clone_05_02_2024/GCGP_copy/results_regen_multiple_randomseeds/"$k"

file=model_summary_$k.txt

# Extract relevant lines and parse the float values
property=$(grep "Property:" "$file" | awk '{print $2}')
seed=$(grep "Seed:" "$file" | awk '{print $2}')
log_marginal_likelihood=$(grep "Log-marginal Likelihood:" "$file" | awk '{print $3}')
test_mapd=$(grep "Test MAPD:" "$file" | awk '{print $3}')
train_mapd=$(grep "Train MAPD:" "$file" | awk '{print $3}')
test_mae=$(grep "Test MAE:" "$file" | awk '{print $3}')
train_mae=$(grep "Train MAE:" "$file" | awk '{print $3}')
test_rmse=$(grep "Test RMSE:" "$file" | awk '{print $3}')
train_rmse=$(grep "Train RMSE:" "$file" | awk '{print $3}')
test_r2=$(grep "Test R2:" "$file" | awk '{print $3}')
train_r2=$(grep "Train R2:" "$file" | awk '{print $3}')

# Append the extracted values to the CSV file
echo "$property,$seed,$log_marginal_likelihood,$test_mapd,$train_mapd,$test_mae,$train_mae,$test_rmse,$train_rmse,$test_r2,$train_r2" >> $output_file
    
done
