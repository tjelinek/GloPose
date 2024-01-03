#!/bin/bash

# Function to download and extract the dataset
download_and_extract() {
    local url=$1
    local filename=$(basename "$url")

    echo "Downloading $filename..."
    wget "$url"

    echo "Extracting $filename..."
    tar -xvf "$filename"

    echo "Cleaning up..."
    rm "$filename"
}

# Directory to save the dataset
base_dir="/mnt/personal/jelint19/data/behave"
mkdir -p "$base_dir"

# Download and extract scanned objects
cd "$base_dir"
download_and_extract "https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/objects.zip"

# Download and extract calibration files
cd "$base_dir"
download_and_extract "https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/calibs.zip"

# Download and extract train and test split
cd "$base_dir"
download_and_extract "https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/split.json"

# Download and extract sequences separated by dates
cd "$base_dir"
declare -a date_sequences=(
    "Date01.zip"
    "Date02.zip"
    "Date03.zip"
    "Date04.zip"
    "Date05.zip"
    "Date06.zip"
    "Date07.zip"
)

for file in "${date_sequences[@]}"; do
    download_and_extract "https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/$file"
done

# Extract all sequence files
cd "$base_dir"
unzip "Date*.zip" -d sequences

# Download and extract raw videos for each date
cd "$base_dir"
declare -a raw_videos=(
    "date01_color.tar"
    "date01_depth.tar"
    "date01_time.tar"
    "date02_color.tar"
    "date02_depth.tar"
    "date02_time.tar"
    "date03_color.tar"
    "date03_depth_sub03.tar"
    "date03_depth_sub04.tar"
    "date03_depth_sub05.tar"
    "date03_empty.tar"
    "date03_time.tar"
    "date04_color.tar"
    "date04_depth.tar"
    "date04_time.tar"
    "date05_color.tar"
    "date05_depth.tar"
    "date05_time.tar"
    "date06_color.tar"
    "date06_depth.tar"
    "date06_time.tar"
    "date07_color.tar"
    "date07_depth_sub4-5.tar"
    "date07_depth_sub8.tar"
    "date07_empty.tar"
    "date07_time.tar"
)

for file in "${raw_videos[@]}"; do
    download_and_extract "https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/video/$file"
done

# Download and extract annotations at 30fps
cd "$base_dir"
download_and_extract "https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/behave-30fps-params-v1.tar"

# Download and extract human and object segmentation masks
cd "$base_dir"
download_and_extract "https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/masks/masks-date01-02.tar"
download_and_extract "https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/masks/masks-date03.tar"
download_and_extract "https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/masks/masks-date04-06.tar"
download_and_extract "https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/masks/masks-date07.tar"

echo "All files downloaded and extracted successfully."
