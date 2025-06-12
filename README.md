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

## Requirements

- macOS with Apple Silicon (for Vision Framework and Metal acceleration)
- Python 3.8+
- OpenCV
- Pandas
- Matplotlib
- Jieba (for Chinese word segmentation)

See `requirements.txt` for a complete list of dependencies.

## Repository Structure

- `*.py` - Core Python modules
- `input/` - Directory for eye-tracking data (not included due to size)
- `processed_output/` - Output directory for processed results (not included due to size)
- `requirements.txt` - Python dependencies
- `summary.md` - Summary of project findings

## Usage

```bash
python simple_ocr_map.py --video VIDEO_PATH --csv EYETRACKING_CSV --interval 5 --confidence 70 --distance 50 --output OUTPUT_CSV
```

See `python simple_ocr_map.py --help` for a complete list of options.

## License

See the LICENSE file for details.
