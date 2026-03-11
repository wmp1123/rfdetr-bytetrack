# traffic analysis

This project performs vehicle detection and tracking using RF-DETR and ByteTrack.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wmp1123/rfdetr-bytetrack.git
cd rfdetr-bytetrack
```

2. Create a virtual environment
```bash
python3.10 -m venv rfdetr-bytetrack
source rfdetr-bytetrack/bin/activate 

conda create -n rfdetr-bytetrack python=3.10 -y
conda activate rfdetr-bytetrack
```

3. Install dependencies
```bash
pip install -r requirements.txt
```
If the libraries in requirements.txt do not work correctly, install the following versions instead:
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2 peft==0.12.0
pip install --upgrade "protobuf<=3.20.3"
```

4. Download pre-trained model
Linux / macOS:
```bash
./setup.sh
```
Windows:
```bash
./setup.ps1
```

## Usage
5. Run the script
```bash
python traffic_analysis.py
```
### Configure In/Out Zones:
```bash
ZONE_OUT_POLYGONS
```
to define the vehicle entry and exit zones.

## Demo

| Before | After (RF-DETR + ByteTrack) |
|-------|------------------------------|
| <img src="assets/before.gif" width="400"/> | <img src="assets/after.gif" width="400"/> |

## Run Evaluation

6. After processing the videos and generating tracking results, you can evaluate the performance using the TrackEval toolkit:
For detailed information, please refer https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md

```bash
cd TrackEval
python scripts/run_mot_challenge.py \
  --BENCHMARK my_challenge \
  --SPLIT_TO_EVAL train \
  --TRACKERS_TO_EVAL RFDETR_ByteTrack \
  --METRICS HOTA CLEAR Identity \ 
  --USE_PARALLEL False \
  --NUM_PARALLEL_CORES 1 \
  --DO_PREPROC False
```
## Project Structure
```bash
data/
├── raw_video/        # Original input videos used for processing
├── images/           # Extracted video frames used for detection and tracking
├── gt/               # Ground truth and tracking result files (MOT format)
└── processed_video/  # Output videos with detection and tracking visualization
```