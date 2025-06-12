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

## Sample Data Included

This repository includes real eye-tracking data for testing:

- `input/User1_241227135925/` - Contains real participant data:
  - Full raw eye-tracking data (~33MB)
  - Sample data file (first 10,000 lines) for quick testing
  - See the README in that directory for usage details

*Note: Video files are not included due to size constraints.*

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
- `input/` - Directory for eye-tracking data (includes sample data)
- `processed_output/` - Output directory for processed results
- `requirements.txt` - Python dependencies
- `summary.md` - Summary of project findings

## Usage

```bash
python simple_ocr_map.py --video VIDEO_PATH --csv EYETRACKING_CSV --interval 5 --confidence 70 --distance 50 --output OUTPUT_CSV
```

For quick testing with the included sample data:

```bash
python simple_ocr_map.py --csv input/User1_241227135925/User1_241227135925_raw_sample.csv --video PATH_TO_VIDEO --output test_output.csv
```

See `python simple_ocr_map.py --help` for a complete list of options.

## License

See the LICENSE file for details.
