#!/bin/bash

# Output CSV file
output_file=/scratch365/bagbodek/GCGP_clone_05_02_2024/GCGP_copy/kernel_sweep_code_and_results/"kernel_sweep_combined_results.csv"
echo "Property,Model,Kernel,Anisotropic?,LML,Test_MAPD,Train_MAPD,Test_MAE,Train_MAE,Test_R2,Train_R2"  > $output_file


for k in $(cat labels_kernel_sweep.txt)
do

Property_=$(echo "$k" | awk -F'[_]' '{print $1;}')
Model_=$(echo "$k" | awk -F'[_]' '{print $2;}')
Kernel_=$(echo "$k" | awk -F'[_]' '{print $3;}')
Anisotropy_=$(echo "$k" | awk -F'[_]' '{print $4;}')



file=/scratch365/bagbodek/GCGP_clone_05_02_2024/GCGP_copy/kernel_sweep_code_and_results/kernel_sweep_all_results/model_summary_$k.txt

# Extract relevant lines and parse the float values
lml_=$(grep "Log-marginal Likelihood:" "$file" | awk '{print $3}')
test_mapd=$(grep "Test MAPD:" "$file" | awk '{print $3}')
train_mapd=$(grep "Train MAPD:" "$file" | awk '{print $3}')
test_mae=$(grep "Test MAE:" "$file" | awk '{print $3}')
train_mae=$(grep "Train MAE:" "$file" | awk '{print $3}')
test_r2=$(grep "Test R2:" "$file" | awk '{print $3}')
train_r2=$(grep "Train R2:" "$file" | awk '{print $3}')

# Append the extracted values to the CSV file
echo "$Property_,$Model_,$Kernel_,$Anisotropy_,$lml_,$test_mapd,$train_mapd,$test_mae,$train_mae,$test_r2,$train_r2" >> $output_file
    
done


