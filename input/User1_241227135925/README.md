# User1 Eye Tracking Data

This directory contains the eye-tracking data for User1 from session 241227135925.

## Files

- `User1_241227135925_raw.csv`: Raw eye-tracking data with gaze coordinates
  - Contains timestamps and gaze coordinates for both eyes
  - Used as input for the eye-tracking to text mapping process
  - File size: ~33MB with complete data

- `User1_241227135925_raw_sample.csv`: Smaller sample of the raw data
  - Contains the first 10,000 lines of the raw data
  - Useful for quick testing and development
  - File size: Much smaller, faster to process

## Usage

This data can be used directly with the simple_ocr_map.py script:

```bash
# Using full data:
python simple_ocr_map.py --csv input/User1_241227135925/User1_241227135925_raw.csv --video PATH_TO_VIDEO --output output_file.csv

# For quick testing with sample data:
python simple_ocr_map.py --csv input/User1_241227135925/User1_241227135925_raw_sample.csv --video PATH_TO_VIDEO --output output_file.csv
```

*Note: The video file is not included due to size constraints. You will need to provide your own video recording of the reading session.* 