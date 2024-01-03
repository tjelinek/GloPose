#!/bin/bash

# Function to download and extract the dataset
download_and_extract() {
    local url=$1
    local filename=$(basename "$url")

    echo "Downloading $filename..."
    wget "$url" --no-check-certificate

    echo "Extracting $filename..."
    tar -xzf "$filename"

    echo "Cleaning up..."
    rm "$filename"
}

# Directory to save the dataset
base_dir="/mnt/personal/jelint19/data/"

# Create a folder for the YCBInEOAT dataset
dataset_dir="$base_dir/YCBInEOAT/YCBInEOAT"
mkdir -p "$dataset_dir"

# Download and extract each file from the YCBInEOAT dataset
declare -a ycbineoat_files=(
    "bleach0.tar.gz"
    "bleach_hard_00_03_chaitanya.tar.gz"
    "cracker_box_reorient.tar.gz"
    "cracker_box_yalehand0.tar.gz"
    "mustard0.tar.gz"
    "mustard_easy_00_02.tar.gz"
    "sugar_box1.tar.gz"
    "sugar_box_yalehand0.tar.gz"
    "tomato_soup_can_yalehand0.tar.gz"
)

for file in "${ycbineoat_files[@]}"; do
    cd "$dataset_dir"
    download_and_extract "https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCBInEOAT/$file"
done

# Create a folder for the YCB_Video_data_organized dataset
dataset_dir="$base_dir/YCBInEOAT/YCB_Video_data_organized"
mkdir -p "$dataset_dir"

# Download and extract each file from the YCB_Video_data_organized dataset
declare -a ycb_video_data_files=(
    "0048.tar.gz"
    "0049.tar.gz"
    "0050.tar.gz"
    "0051.tar.gz"
    "0052.tar.gz"
    "0053.tar.gz"
    "0054.tar.gz"
    "0055.tar.gz"
    "0056.tar.gz"
    "0057.tar.gz"
    "0058.tar.gz"
    "0059.tar.gz"
)

for file in "${ycb_video_data_files[@]}"; do
    cd "$dataset_dir"
    download_and_extract "https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCB_Video_data_organized/$file"
done

echo "All files downloaded and extracted successfully."
