#!/bin/bash

# Download URLs
segmentations_url="https://cloud.tugraz.at/index.php/s/z8SCsWCYM3YcQWX/download?path=%2F&files=HO3D_v3_segmentations_rendered.zip"
ho3d_url="https://cloud.tugraz.at/index.php/s/z8SCsWCYM3YcQWX/download?path=%2F&files=HO3D_v3.zip"

# Output directory
output_dir="/mnt/personal/jelint19/data/HO3D"

# Create the output directory if it does not exist
mkdir -p "$output_dir"

# Download the datasets
echo "Downloading HO3D_v3_segmentations_rendered.zip..."
wget -O "$output_dir/HO3D_v3_segmentations_rendered.zip" "$segmentations_url"

echo "Downloading HO3D_v3.zip..."
wget -O "$output_dir/HO3D_v3.zip" "$ho3d_url"

# Extract the datasets
echo "Extracting HO3D_v3_segmentations_rendered.zip..."
unzip "$output_dir/HO3D_v3_segmentations_rendered.zip" -d "$output_dir"

echo "Extracting HO3D_v3.zip..."
unzip "$output_dir/HO3D_v3.zip" -d "$output_dir"

echo "Datasets downloaded and extracted to $output_dir."
