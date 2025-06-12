# Participant Eye-Tracking Data

This directory contains real eye-tracking data from multiple participants for testing and development of the eye-tracking to text mapping system.

## Participants Included

### 曾祥泰 (Zeng Xiangtai)
- **Sessions**: 2 sessions (User1_241227091458, User1_241227091534)
- **Total Size**: ~285MB
- **Description**: Complete eye-tracking data with CSV files and video recordings

### 戴萌萌 (Dai Mengmeng)  
- **Sessions**: 4 sessions (User1_241225095134, User1_241225095158, User1_241225095222, User1_241225095301)
- **Total Size**: ~219MB
- **Description**: Multiple sessions providing variety in reading patterns

### 沈若枢 (Shen Ruoshu)
- **Sessions**: 1 session (User1_241227135925)
- **Total Size**: ~181MB
- **Description**: Single comprehensive session with full data

### 张璇 (Zhang Xuan)
- **Sessions**: Multiple sessions
- **Total Size**: ~249MB
- **Description**: Additional participant data for testing

### 江宇静 (Jiang Yujing)
- **Sessions**: Multiple sessions
- **Total Size**: ~820MB
- **Description**: Comprehensive dataset with multiple recording sessions

## File Structure

Each participant directory contains subdirectories for individual sessions, with each session including:

- `*_raw.csv`: Raw eye-tracking data with timestamps and gaze coordinates
- `*_basic.csv`: Basic processed eye-tracking data
- `*_filter.csv`: Filtered eye-tracking data
- `*_velocity.csv`: Velocity calculations
- `*_index.csv`: Index data
- `*.mp4`: Video recording of the reading session
- Additional annotation and analysis files

## Usage

Use any of these datasets with the main processing script:

```bash
# Example with 曾祥泰 data:
python simple_ocr_map.py \
  --csv input/曾祥泰/User1_241227091458/User1_241227091458_raw.csv \
  --video input/曾祥泰/User1_241227091458/User1_241227091458.mp4 \
  --output results_zengxiangtai.csv

# Example with 戴萌萌 data:
python simple_ocr_map.py \
  --csv input/戴萌萌/User1_241225095134/User1_241225095134_raw.csv \
  --video input/戴萌萌/User1_241225095134/User1_241225095134.mp4 \
  --output results_daimengmeng.csv
```

## Data Notes

- All video files are stored using Git LFS due to their large size
- CSV files contain the actual eye-tracking coordinates and timing data
- Each participant may have different numbers of sessions
- Data represents real reading/translation tasks with Chinese-English text pairs

## File Sizes

The total dataset is approximately 1.8GB, with video files making up the majority of the storage space. CSV files are typically much smaller (1-50MB each) while video files can be 100-500MB each. 