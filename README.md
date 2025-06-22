# traffic analysis

This project performs vehicle detection and tracking using RF-DETR and ByteTrack.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wmp1123/rfdetr-bytetrack.git
cd traffic_analysis
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
./setup.sh

5. Run the script
```bash
python traffic_analysis.py
```