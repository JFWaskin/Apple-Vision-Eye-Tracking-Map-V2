#!/bin/bash
# Real Data Test Script for macOS (Apple Vision Framework required)
# This script runs the complete pipeline on real participant data

set -e

echo "================================================================================"
echo "🔬 REAL DATA PIPELINE TEST - 3 FRAME INTERVAL"
echo "================================================================================"
echo ""

# Check environment
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ ERROR: This script requires macOS with Apple Vision Framework"
    echo "   Current OS: $OSTYPE"
    echo ""
    echo "📝 To run this test:"
    echo "   1. Transfer this repository to a Mac"
    echo "   2. Run: git lfs pull (to download participant data)"
    echo "   3. Run: pip install -r requirements.txt"
    echo "   4. Run: bash run_real_test.sh"
    exit 1
fi

# Configuration
PARTICIPANT="${1:-沈若枢}"
SESSION="${2:-User1_241227135925}"
OUTPUT_DIR="real_test_results"
INTERVAL=3

echo "📋 Configuration:"
echo "   Participant: $PARTICIPANT"
echo "   Session: $SESSION"
echo "   Frame Interval: $INTERVAL"
echo "   Output Directory: $OUTPUT_DIR"
echo ""

# Check if data exists
CSV_FILE="input/$PARTICIPANT/$SESSION/${SESSION}_raw.csv"
VIDEO_FILE_MP4="input/$PARTICIPANT/$SESSION/${SESSION}.mp4"
VIDEO_FILE_MOV="input/$PARTICIPANT/$SESSION/${SESSION}.mov"

echo "🔍 Checking for participant data..."
if [ ! -f "$CSV_FILE" ]; then
    echo "❌ CSV file not found: $CSV_FILE"
    exit 1
fi

# Check if it's a Git LFS pointer
if head -1 "$CSV_FILE" | grep -q "version https://git-lfs"; then
    echo "⚠️  CSV file is a Git LFS pointer. Running git lfs pull..."
    git lfs pull --include="input/$PARTICIPANT/$SESSION/*.csv"
fi

# Find video file
VIDEO_FILE=""
if [ -f "$VIDEO_FILE_MP4" ]; then
    VIDEO_FILE="$VIDEO_FILE_MP4"
elif [ -f "$VIDEO_FILE_MOV" ]; then
    VIDEO_FILE="$VIDEO_FILE_MOV"
else
    echo "❌ Video file not found:"
    echo "   Tried: $VIDEO_FILE_MP4"
    echo "   Tried: $VIDEO_FILE_MOV"
    exit 1
fi

# Check if video is Git LFS pointer
if head -1 "$VIDEO_FILE" | grep -q "version https://git-lfs"; then
    echo "⚠️  Video file is a Git LFS pointer. Running git lfs pull..."
    git lfs pull --include="input/$PARTICIPANT/$SESSION/*.mp4,input/$PARTICIPANT/$SESSION/*.mov"
fi

echo "✅ Data files located:"
echo "   CSV: $CSV_FILE ($(wc -l < "$CSV_FILE") lines)"
echo "   Video: $VIDEO_FILE ($(du -h "$VIDEO_FILE" | cut -f1))"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "================================================================================"
echo "🚀 STEP 1: Running OCR Recognition Pipeline (Interval=$INTERVAL)"
echo "================================================================================"
echo ""

# Run the pipeline
python3 simple_ocr_map.py \
    --video "$VIDEO_FILE" \
    --csv "$CSV_FILE" \
    --interval $INTERVAL \
    --confidence 60 \
    --distance 10 \
    --output "$OUTPUT_DIR/mapped_gaze_interval${INTERVAL}.csv" \
    --fast-mode \
    --batch-size 128 \
    --chunk-size 200

echo ""
echo "✅ Pipeline completed!"
echo ""

# Check if ocr_results.csv was generated
if [ ! -f "ocr_results.csv" ]; then
    echo "⚠️  ocr_results.csv not found in current directory"
    OCR_CSV=""
else
    mv ocr_results.csv "$OUTPUT_DIR/ocr_results_interval${INTERVAL}.csv"
    OCR_CSV="$OUTPUT_DIR/ocr_results_interval${INTERVAL}.csv"
    echo "📝 OCR results: $OCR_CSV"
fi

MAPPED_CSV="$OUTPUT_DIR/mapped_gaze_interval${INTERVAL}.csv"
echo "🎯 Mapped results: $MAPPED_CSV"
echo ""

echo "================================================================================"
echo "🗺️  STEP 2: Generating Progress Maps"
echo "================================================================================"
echo ""

# Generate comparison figure
echo "📊 Generating downsampling comparison..."
python3 generate_progress_map.py \
    --video "$VIDEO_FILE" \
    --mapped-csv "$MAPPED_CSV" \
    --ocr-csv "$OCR_CSV" \
    --comparison \
    --output "$OUTPUT_DIR/progress_map_comparison.png"

echo ""
echo "📊 Generating adaptive downsampled map..."
python3 generate_progress_map.py \
    --video "$VIDEO_FILE" \
    --mapped-csv "$MAPPED_CSV" \
    --ocr-csv "$OCR_CSV" \
    --downsample adaptive \
    --target-points 500 \
    --show-trajectory \
    --show-text \
    --output "$OUTPUT_DIR/progress_map_adaptive.png"

echo ""
echo "📊 Generating spatial downsampled map..."
python3 generate_progress_map.py \
    --video "$VIDEO_FILE" \
    --mapped-csv "$MAPPED_CSV" \
    --ocr-csv "$OCR_CSV" \
    --downsample spatial \
    --show-trajectory \
    --show-text \
    --output "$OUTPUT_DIR/progress_map_spatial.png"

echo ""
echo "================================================================================"
echo "📈 STEP 3: Results Summary"
echo "================================================================================"
echo ""

# Analyze results
python3 << EOF
import pandas as pd
import os

mapped_csv = "$MAPPED_CSV"
ocr_csv = "$OCR_CSV" if "$OCR_CSV" else None

if os.path.exists(mapped_csv):
    df = pd.read_csv(mapped_csv)
    print(f"📊 Mapped Gaze Results:")
    print(f"   Total mapped points: {len(df):,}")
    print(f"   Unique frames: {df['frame_num'].nunique()}")
    if 'confidence' in df.columns:
        print(f"   Avg confidence: {df['confidence'].mean():.1f}%")
    if 'distance' in df.columns:
        print(f"   Avg distance: {df['distance'].mean():.2f} px")
    if 'timestamp' in df.columns:
        duration_s = (df['timestamp'].max() - df['timestamp'].min()) / 1_000_000
        print(f"   Duration: {duration_s:.1f}s")
    print()

if ocr_csv and os.path.exists(ocr_csv):
    ocr_df = pd.read_csv(ocr_csv)
    print(f"📝 OCR Results:")
    print(f"   Total text entries: {len(ocr_df):,}")
    if 'text' in ocr_df.columns:
        print(f"   Unique words: {ocr_df['text'].nunique()}")
    if 'conf' in ocr_df.columns:
        print(f"   Avg confidence: {ocr_df['conf'].mean():.1f}%")
    print()

print(f"🗺️  Generated Progress Maps:")
output_dir = "$OUTPUT_DIR"
for fname in os.listdir(output_dir):
    if fname.endswith('.png'):
        fpath = os.path.join(output_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"   ✅ {fname} ({size_kb:.1f} KB)")
print()
EOF

echo "================================================================================"
echo "✅ REAL DATA TEST COMPLETE"
echo "================================================================================"
echo ""
echo "📁 Results saved to: $OUTPUT_DIR/"
echo ""
echo "🎯 Next steps:"
echo "   1. Review performance summary above"
echo "   2. Open progress maps in $OUTPUT_DIR/"
echo "   3. Compare downsampling methods"
echo "   4. Adjust parameters if needed and re-run"
echo ""
