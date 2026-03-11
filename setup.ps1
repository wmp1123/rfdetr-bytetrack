$URL = "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth"
$OUTPUT_FILE = "rf-detr-base.pth"

Write-Host "Downloading RF-DETR model..."
Invoke-WebRequest -Uri $URL -OutFile $OUTPUT_FILE

if (Test-Path $OUTPUT_FILE) {
    Write-Host "Download complete: $OUTPUT_FILE"
} else {
    Write-Host "Download failed."
}
