#!/bin/bash

# Define the download URL and output filename
URL="https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth"
OUTPUT_FILE="rf-detr-base.pth"

# Download the file
echo "Downloading RF-DETR model..."
wget -O ${OUTPUT_FILE} "${URL}"

# Confirm download
if [ -f ${OUTPUT_FILE} ]; then
    echo "Download complete: ${OUTPUT_FILE}"
else
    echo "Download failed."
fi