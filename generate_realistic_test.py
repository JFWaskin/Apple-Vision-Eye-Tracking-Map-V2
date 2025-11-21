#!/usr/bin/env python3
"""
Generate realistic test data based on actual participant data structure.
This simulates what the pipeline would produce with real data on macOS.
"""

import argparse
import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path


def load_real_gaze_structure(sample_csv="input/User1_241227135925/User1_241227135925_raw.csv"):
    """Load structure from real CSV file (even if it's an LFS pointer)."""
    # Try to read the file to get column names
    try:
        df = pd.read_csv(sample_csv, nrows=5)
        return list(df.columns)
    except:
        # Fallback to known structure
        return [
            'timestampUs', 'leftEyeX', 'leftEyeY', 'rightEyeX', 'rightEyeY',
            'leftEyeConfidence', 'rightEyeConfidence'
        ]


def generate_realistic_gaze_data(num_points=10000, duration_s=180):
    """
    Generate realistic gaze data matching actual participant data format.

    Simulates a reading session with:
    - Two-column bilingual text (Chinese-English)
    - Natural reading patterns (fixations, saccades)
    - Realistic timestamps and confidence values
    """
    print("\n📊 Generating realistic gaze data...")

    # Video dimensions (typical setup)
    video_width = 1920
    video_height = 1080

    # Text layout (bilingual reading material)
    # Left column: Source text (Chinese), Right column: Target text (English)
    left_col_x = (200, 800)  # Source text area
    right_col_x = (1100, 1700)  # Target text area
    text_y = (150, 950)  # Vertical range

    # Reading pattern parameters
    num_lines = 20
    line_height = (text_y[1] - text_y[0]) / num_lines

    gaze_points = []
    timestamp_us = 0
    time_increment_us = (duration_s * 1_000_000) // num_points

    for i in range(num_points):
        # Simulate reading pattern:
        # - 60% source text (left)
        # - 35% target text (right)
        # - 5% elsewhere (distractions, returns)

        rand = np.random.random()
        if rand < 0.60:  # Source text
            x = np.random.uniform(left_col_x[0], left_col_x[1])
            line_idx = np.random.randint(0, num_lines)
            y = text_y[0] + line_idx * line_height + np.random.normal(0, 8)
        elif rand < 0.95:  # Target text
            x = np.random.uniform(right_col_x[0], right_col_x[1])
            line_idx = np.random.randint(0, num_lines)
            y = text_y[0] + line_idx * line_height + np.random.normal(0, 8)
        else:  # Elsewhere
            x = np.random.uniform(0, video_width)
            y = np.random.uniform(0, video_height)

        # Add natural jitter between left and right eye
        left_x = x + np.random.normal(0, 2)
        left_y = y + np.random.normal(0, 2)
        right_x = x + np.random.normal(0, 2)
        right_y = y + np.random.normal(0, 2)

        # Realistic confidence values
        left_conf = np.random.uniform(0.85, 0.99)
        right_conf = np.random.uniform(0.85, 0.99)

        gaze_points.append({
            'timestampUs': timestamp_us,
            'leftEyeX': left_x,
            'leftEyeY': left_y,
            'rightEyeX': right_x,
            'rightEyeY': right_y,
            'leftEyeConfidence': left_conf,
            'rightEyeConfidence': right_conf
        })

        timestamp_us += time_increment_us

    df = pd.DataFrame(gaze_points)
    print(f"   ✅ Generated {len(df):,} gaze points")
    print(f"   Duration: {duration_s}s")
    print(f"   Sampling rate: ~{num_points/duration_s:.1f} Hz")

    return df


def generate_realistic_ocr_data(num_frames=1800, interval=3):
    """
    Generate realistic OCR results matching Vision Framework output.

    Simulates bilingual text recognition across video frames.
    """
    print("\n📝 Generating realistic OCR data...")

    # Sample bilingual vocabulary
    chinese_words = [
        "眼睛", "追踪", "技术", "研究", "系统", "数据", "分析", "方法",
        "实验", "结果", "阅读", "理解", "翻译", "语言", "文本", "测试",
        "用户", "界面", "设计", "开发", "应用", "性能", "优化", "算法"
    ]

    english_words = [
        "eye", "tracking", "technology", "research", "system", "data", "analysis", "method",
        "experiment", "results", "reading", "comprehension", "translation", "language", "text", "test",
        "user", "interface", "design", "development", "application", "performance", "optimization", "algorithm"
    ]

    ocr_entries = []

    # Text layout
    left_col_x = 200  # Source (Chinese)
    right_col_x = 1100  # Target (English)
    text_y_start = 150
    line_height = 40
    word_width_cn = 80
    word_width_en = 120

    # Generate OCR entries for sampled frames
    processed_frames = list(range(0, num_frames, interval))

    for frame_idx, frame_num in enumerate(processed_frames):
        # Each frame has multiple text lines
        num_lines = 10
        words_per_line = 5

        for line_idx in range(num_lines):
            y = text_y_start + line_idx * line_height

            # Left column (Chinese)
            for word_idx in range(words_per_line):
                x = left_col_x + word_idx * (word_width_cn + 10)
                word = chinese_words[(frame_idx * num_lines * words_per_line + line_idx * words_per_line + word_idx) % len(chinese_words)]

                ocr_entries.append({
                    'frame': frame_num,
                    'text': word,
                    'conf': np.random.uniform(85, 98),
                    'x': x,
                    'y': y,
                    'w': word_width_cn,
                    'h': 35,
                    'category': 'source'
                })

            # Right column (English)
            for word_idx in range(words_per_line):
                x = right_col_x + word_idx * (word_width_en + 15)
                word = english_words[(frame_idx * num_lines * words_per_line + line_idx * words_per_line + word_idx) % len(english_words)]

                ocr_entries.append({
                    'frame': frame_num,
                    'text': word,
                    'conf': np.random.uniform(85, 98),
                    'x': x,
                    'y': y,
                    'w': word_width_en,
                    'h': 35,
                    'category': 'target'
                })

    df = pd.DataFrame(ocr_entries)
    print(f"   ✅ Generated {len(df):,} OCR entries")
    print(f"   Frames processed: {len(processed_frames)} (interval={interval})")
    print(f"   Unique words: {df['text'].nunique()}")

    return df


def generate_mapped_data(gaze_df, ocr_df, fps=30):
    """
    Generate realistic mapped gaze-to-text data.

    Simulates the output of map_gaze_to_text_optimized().
    """
    print("\n🎯 Generating realistic mapped data...")

    # Convert timestamps to frame numbers
    gaze_df['frame_num'] = (gaze_df['timestampUs'] / 1_000_000 * fps).astype(int)

    # Calculate average gaze position
    gaze_df['gaze_x'] = (gaze_df['leftEyeX'] + gaze_df['rightEyeX']) / 2
    gaze_df['gaze_y'] = (gaze_df['leftEyeY'] + gaze_df['rightEyeY']) / 2

    mapped_data = []

    for _, gaze_row in gaze_df.iterrows():
        frame_num = gaze_row['frame_num']
        gaze_x = gaze_row['gaze_x']
        gaze_y = gaze_row['gaze_y']

        # Find OCR entries in this frame
        frame_ocr = ocr_df[ocr_df['frame'] == frame_num]

        if len(frame_ocr) > 0:
            # Calculate distances to all text boxes
            distances = np.sqrt(
                (frame_ocr['x'] + frame_ocr['w']/2 - gaze_x)**2 +
                (frame_ocr['y'] + frame_ocr['h']/2 - gaze_y)**2
            )

            # Find closest text within threshold (100px)
            min_dist = distances.min()
            if min_dist < 100:
                closest_idx = distances.idxmin()
                ocr_row = frame_ocr.loc[closest_idx]

                mapped_data.append({
                    'timestamp': gaze_row['timestampUs'],
                    'frame_num': frame_num,
                    'gaze_x': gaze_x,
                    'gaze_y': gaze_y,
                    'text': ocr_row['text'],
                    'confidence': ocr_row['conf'],
                    'distance': min_dist,
                    'category': ocr_row['category']
                })

    df = pd.DataFrame(mapped_data)

    if len(df) > 0:
        mapping_rate = len(df) / len(gaze_df) * 100
        print(f"   ✅ Generated {len(df):,} mapped points")
        print(f"   Mapping rate: {mapping_rate:.1f}%")
        print(f"   Avg confidence: {df['confidence'].mean():.1f}%")
        print(f"   Avg distance: {df['distance'].mean():.2f} px")
    else:
        print(f"   ⚠️  No mappings generated")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate realistic test data based on real participant structure")
    parser.add_argument("--output-dir", default="realistic_test_data", help="Output directory")
    parser.add_argument("--gaze-points", type=int, default=10000, help="Number of gaze points")
    parser.add_argument("--duration", type=int, default=180, help="Duration in seconds")
    parser.add_argument("--interval", type=int, default=3, help="Frame sampling interval")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("🔬 REALISTIC TEST DATA GENERATOR")
    print("="*80)
    print(f"\nBased on real participant data structure from input/")
    print(f"Output directory: {output_dir}")
    print(f"Gaze points: {args.gaze_points:,}")
    print(f"Duration: {args.duration}s")
    print(f"Frame interval: {args.interval}")
    print(f"FPS: {args.fps}")

    # Calculate total frames
    total_frames = args.duration * args.fps

    # Generate data
    gaze_df = generate_realistic_gaze_data(args.gaze_points, args.duration)
    ocr_df = generate_realistic_ocr_data(total_frames, args.interval)
    mapped_df = generate_mapped_data(gaze_df, ocr_df, args.fps)

    # Save files
    gaze_csv = output_dir / "raw_gaze.csv"
    ocr_csv = output_dir / "ocr_results.csv"
    mapped_csv = output_dir / "mapped_gaze.csv"

    gaze_df.to_csv(gaze_csv, index=False)
    ocr_df.to_csv(ocr_csv, index=False)
    mapped_df.to_csv(mapped_csv, index=False)

    print("\n" + "="*80)
    print("✅ REALISTIC TEST DATA GENERATION COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  📊 {gaze_csv} ({gaze_csv.stat().st_size / 1024:.1f} KB)")
    print(f"  📝 {ocr_csv} ({ocr_csv.stat().st_size / 1024:.1f} KB)")
    print(f"  🎯 {mapped_csv} ({mapped_csv.stat().st_size / 1024:.1f} KB)")

    print(f"\n📋 Data Summary:")
    print(f"  Gaze points: {len(gaze_df):,}")
    print(f"  OCR entries: {len(ocr_df):,}")
    print(f"  Mapped points: {len(mapped_df):,}")
    if len(mapped_df) > 0:
        print(f"  Mapping rate: {len(mapped_df) / len(gaze_df) * 100:.1f}%")
        print(f"  Source text fixations: {len(mapped_df[mapped_df['category'] == 'source'])}")
        print(f"  Target text fixations: {len(mapped_df[mapped_df['category'] == 'target'])}")

    print("\n🚀 Ready to test! Run:")
    print(f"  python3 generate_progress_map.py \\")
    print(f"    --video dummy.mp4 \\")
    print(f"    --mapped-csv {mapped_csv} \\")
    print(f"    --ocr-csv {ocr_csv} \\")
    print(f"    --comparison")
    print()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
