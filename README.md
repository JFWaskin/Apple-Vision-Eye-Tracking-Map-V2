# Apple Vision Eye Tracking Map

A specialized tool for mapping eye-tracking data to text using Apple's Vision Framework for OCR (Optical Character Recognition).

## Overview

This project provides a complete pipeline for processing eye-tracking data and mapping gaze coordinates to text elements on a screen. It uses Apple's Vision Framework to perform OCR on screen recordings and then maps the eye-tracking coordinates to the recognized text.

## Key Features

- Chinese and English text recognition support
- Efficient parallel processing with Metal GPU acceleration
- Time period-based analysis for focused reading sessions
- CSV output format for easy analysis and visualization
- Comprehensive visualization tools for mapped gaze data

## Complete Dataset Included

This repository includes a comprehensive dataset with real eye-tracking data from multiple participants:

### Participants (Total: ~1.8GB)
- **曾祥泰 (Zeng Xiangtai)**: 2 sessions (~285MB)
- **戴萌萌 (Dai Mengmeng)**: 4 sessions (~219MB) 
- **沈若枢 (Shen Ruoshu)**: 1 session (~181MB)
- **张璇 (Zhang Xuan)**: Multiple sessions (~249MB)
- **江宇静 (Jiang Yujing)**: Multiple sessions (~820MB)

Each participant directory includes:
- Raw eye-tracking CSV data
- Video recordings of reading sessions (via Git LFS)
- Processed data files
- Session metadata

*See `input/README.md` for detailed information about each participant and usage examples.*

## Requirements

- macOS with Apple Silicon (for Vision Framework and Metal acceleration)
- Python 3.8+
- OpenCV
- Pandas
- Matplotlib
- Jieba (for Chinese word segmentation)
- Git LFS (for video file handling)

See `requirements.txt` for a complete list of dependencies.

## Repository Structure

- `*.py` - Core Python modules
- `input/` - Complete participant dataset with videos and CSV files
- `processed_output/` - Output directory for processed results
- `requirements.txt` - Python dependencies
- `summary.md` - Summary of project findings

## Quick Start

1. Clone the repository (with LFS for video files):
```bash
git clone https://github.com/JFWaskin/Apple-Vision-Eye-Tracking-Map-V2.git
cd Apple-Vision-Eye-Tracking-Map-V2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run with sample data:
```bash
python simple_ocr_map.py \
  --csv input/沈若枢/User1_241227135925/User1_241227135925_raw.csv \
  --video input/沈若枢/User1_241227135925/User1_241227135925.mp4 \
  --output test_results.csv
```

## Usage

```bash
python simple_ocr_map.py --video VIDEO_PATH --csv EYETRACKING_CSV --interval 5 --confidence 70 --distance 50 --output OUTPUT_CSV
```

See `python simple_ocr_map.py --help` for a complete list of options.

## License

See the LICENSE file for details.
