# 3-Frame Interval Speed Test Guide

## Overview

This guide explains how to run the speed-optimized eye-tracking recognition pipeline with a 3-frame interval and provides detailed optimization recommendations.

## Prerequisites

### System Requirements
- **Operating System**: macOS (required for Apple Vision Framework and Metal GPU acceleration)
- **Python**: 3.8+
- **Git LFS**: Required to download participant data files

### Installation

1. **Install Git LFS** (if not already installed):
```bash
brew install git-lfs
git lfs install
```

2. **Pull participant data files**:
```bash
# Pull all LFS files (may take time depending on data size)
git lfs pull

# Or pull specific participant data:
git lfs pull --include="input/沈若枢/**"
```

3. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

## Running the Speed Test

### Quick Start

Run the test on a specific participant session:

```bash
python test_3frame_speed.py --participant 沈若枢 --session User1_241227135925
```

### Available Participants

Based on the repository structure:
- 沈若枢 (Shen Ruoshu)
- 戴萌萌 (Dai Mengmeng)
- 曾祥泰 (Zeng Xiangtai)
- 张璇 (Zhang Xuan)
- 江宇静 (Jiang Yujing)

List available sessions:
```bash
ls input/*/User*
```

### Command-Line Options

```bash
python test_3frame_speed.py \
  --participant <name>      # Required: Participant name
  --session <session>       # Required: Session name (e.g., User1_241227135925)
  --output-dir <dir>        # Optional: Output directory (default: test_results)
```

### What Gets Tested

The test script is configured for **maximum speed** with the following optimizations:

| Parameter | Value | Impact |
|-----------|-------|--------|
| Frame Interval | **3** | Process every 3rd frame (3× faster than interval=1) |
| Fast Mode | **Enabled** | Use Vision Framework's fast recognition mode |
| Batch Size | **128** | Large batches for better GPU utilization |
| Chunk Size | **200** | Balance between memory and speed |
| In-Memory Processing | **Enabled** | Avoid disk I/O (requires CoreGraphics) |
| Threads | **12** | Maximum parallelism |
| Confidence Threshold | 60% | Balanced accuracy/coverage |
| Distance Threshold | 10px | Gaze-to-text matching tolerance |

## Understanding the Output

### Enhanced Logging

The enhanced logging system now provides:

```
🚀 Starting Eye-Tracking to Text Mapping Pipeline

⏱️  Gaze Data Loading: 0.25s (15432 points)
📹 Starting OCR processing (interval=3)...
⏱️  OCR Processing: 45.32s (850 frames, 12453 words)
💾 Memory: 2048.3 MB

🎯 Mapping gaze points to text (conf≥60%, dist≤10px)...
⏱️  Gaze Mapping: 2.15s (12308 mapped)
⏱️  Result Saving: 0.18s

================================================================================
📈 PERFORMANCE SUMMARY
================================================================================
⏱️  Total Pipeline Time: 47.90s

🔍 Stage Breakdown:
  Gaze Data Loading............    0.25s ( 0.5%)
  Frame Extraction.............   40.12s (83.8%)
  Preprocessing................    3.20s ( 6.7%)
  OCR Processing...............   45.32s (94.6%)
  Text Categorization..........    0.08s ( 0.2%)
  Gaze Mapping.................    2.15s ( 4.5%)
  Result Saving................    0.18s ( 0.4%)

📊 Processing Statistics:
  Total Frames Processed: 850
  Total OCR Calls: 107
  Words Recognized: 12453
  Gaze Points (Total): 15432
  Gaze Points (Mapped): 12308
  Mapping Success Rate: 79.8%

🚀 Performance Metrics:
  OCR Processing Speed: 18.76 fps
  Cache Hit Rate: 15.3% (164/1072)
  Final Batch Size: 128
  Final Processing Speed: 20.45 fps

💾 Memory Usage:
  Average: 1845.32 MB
  Peak: 2048.67 MB

🎮 GPU Statistics:
  GPU Time: 42.15s
  GPU Memory Peak: 1536.45 MB
================================================================================
```

### Key Metrics to Watch

1. **OCR Processing Speed (fps)**: Higher is better
   - Target: >15 fps
   - Indicates GPU utilization efficiency

2. **Cache Hit Rate**: Higher is better
   - Shows how many frames were deduplicated
   - High hit rate means video has static content

3. **Mapping Success Rate**: Balance with speed
   - Shows % of gaze points successfully mapped to text
   - Lower interval = higher accuracy, but slower

4. **Memory Usage**: Monitor for stability
   - Peak memory should stay under system limits
   - Adjust chunk_size if memory issues occur

## Output Files

The test generates:

1. **`test_results/<session>_3frame_mapped.csv`**
   - Main output: gaze points mapped to recognized text
   - Columns: timestamp, frame_num, gaze_x, gaze_y, text, confidence, distance, category

2. **`ocr_results.csv`**
   - All recognized text from video frames
   - Columns: frame, text, conf, x, y, w, h, category

## Troubleshooting

### "Apple Vision Framework is not available"

**Cause**: Running on non-macOS system or missing frameworks.

**Solution**: Must run on macOS with Xcode installed.

### "CSV/Video file appears to be a Git LFS pointer"

**Cause**: Git LFS files not downloaded.

**Solution**:
```bash
git lfs pull
```

### "No Metal GPU found"

**Cause**: Metal not available or not detected.

**Impact**: Will fall back to CPU (much slower).

**Solution**: Ensure running on Mac with Metal-capable GPU.

### High Memory Usage / Out of Memory

**Cause**: Processing too many frames at once.

**Solution**: Reduce chunk size:
```python
# In test_3frame_speed.py, modify config:
'chunk_size': 100,  # Reduce from 200
```

### Low Mapping Success Rate (<50%)

**Causes**:
- Frame interval too high (missing text)
- Confidence threshold too strict
- Distance threshold too small

**Solutions**:
- Reduce interval (e.g., 2 instead of 3)
- Lower confidence threshold (e.g., 50 instead of 60)
- Increase distance threshold (e.g., 15 instead of 10)

## Next Steps

After running the test:

1. **Analyze the performance summary** - Identify bottlenecks
2. **Check the mapping success rate** - Balance speed vs accuracy
3. **Review the optimization guide** - See OPTIMIZATION_GUIDE.md for detailed recommendations
4. **Experiment with parameters** - Find the optimal balance for your use case

## Manual Testing

To run with custom parameters:

```bash
python simple_ocr_map.py \
  --video input/沈若枢/User1_241227135925/User1_241227135925.mp4 \
  --csv input/沈若枢/User1_241227135925/User1_241227135925_raw.csv \
  --interval 3 \
  --fast-mode \
  --batch-size 128 \
  --chunk-size 200 \
  --confidence 60 \
  --distance 10 \
  --output test_output.csv
```

## Benchmarking Different Intervals

Compare performance across different intervals:

```bash
# Interval = 1 (highest accuracy, slowest)
python simple_ocr_map.py --video <video> --csv <csv> --interval 1 --output results_1.csv

# Interval = 3 (balanced)
python simple_ocr_map.py --video <video> --csv <csv> --interval 3 --output results_3.csv

# Interval = 5 (fastest, lower accuracy)
python simple_ocr_map.py --video <video> --csv <csv> --interval 5 --output results_5.csv
```

Compare mapping success rates and processing times to find the optimal interval.
