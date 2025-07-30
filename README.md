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
4. Download pre-trained model
```bash
./setup.sh
```

5. Run the script
```bash
python traffic_analysis.py
```

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

data/
├── raw_video/        # Original input videos used for processing
├── images/           # Extracted video frames used for detection and tracking
├── gt/               # Ground truth and tracking result files (e.g., MOT format)
└── processed_video/  # Output videos with detection and tracking visualized