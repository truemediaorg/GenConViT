#!/bin/bash

# Define the S3 bucket
BUCKET="s3://truemedia-media/"

# Define the local directory where you want to copy files
LOCAL_DIR="../all_data"

# List of video file extensions to copy
declare -a EXTENSIONS=("mp4" "mov" "avi" "mkv" "flv" "webm")

# Loop over each extension and copy the files
for EXT in "${EXTENSIONS[@]}"
do
    echo "Copying *.$EXT files from $BUCKET to $LOCAL_DIR"
    aws s3 cp $BUCKET $LOCAL_DIR --recursive --exclude "*" --include "*.$EXT"
done

echo "Copy operation completed."
