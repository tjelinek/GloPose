#!/bin/bash

# Directory containing .pkl files
pkl_directory="/mnt/personal/jelint19/data/SyntheticObjectsWorkshopKubric/"

# Script to run with each .pkl file
script_to_run="generate_dataset_kubric.batch"

# Check if the script exists
if [ ! -f "$script_to_run" ]; then
    echo "Error: The script '$script_to_run' does not exist."
    exit 1
fi

# Loop through .pkl files and run the script with each file as an argument
for pkl_file in "$pkl_directory"*.pkl; do
    if [ -f "$pkl_file" ]; then
        base_filename=$(basename "$pkl_file")
        echo "Running $script_to_run with $base_filename"
        # Run the script with the .pkl file as an argument
        sbatch "$script_to_run" "$base_filename"
    fi
done
