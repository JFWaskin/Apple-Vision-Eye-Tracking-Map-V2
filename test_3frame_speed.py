#!/usr/bin/env python3
"""
Speed-optimized test script for 3-frame interval recognition pipeline.
This script configures and runs the eye-tracking to text mapping with maximum speed.

Usage:
    python test_3frame_speed.py --participant <participant_name> --session <session_name>

Example:
    python test_3frame_speed.py --participant 沈若枢 --session User1_241227135925
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from simple_ocr_map import process_with_args

def main():
    parser = argparse.ArgumentParser(description="Test 3-frame interval recognition with speed optimization")
    parser.add_argument("--participant", required=True, help="Participant name (e.g., 沈若枢)")
    parser.add_argument("--session", required=True, help="Session name (e.g., User1_241227135925)")
    parser.add_argument("--output-dir", default="test_results", help="Output directory for results")
    args = parser.parse_args()

    # Construct paths
    participant_dir = Path("input") / args.participant / args.session
    if not participant_dir.exists():
        print(f"❌ Error: Participant directory not found: {participant_dir}")
        print("\nAvailable participants:")
        input_dir = Path("input")
        if input_dir.exists():
            for p in input_dir.iterdir():
                if p.is_dir() and not p.name.startswith('.'):
                    print(f"  - {p.name}")
                    for s in p.iterdir():
                        if s.is_dir() and s.name.startswith('User'):
                            print(f"    → {s.name}")
        return 1

    # Find CSV and video files
    csv_file = participant_dir / f"{args.session}_raw.csv"
    video_file = None
    for ext in ['.mp4', '.mov', '.MP4', '.MOV']:
        potential_video = participant_dir / f"{args.session}{ext}"
        if potential_video.exists():
            video_file = potential_video
            break

    if not csv_file.exists():
        print(f"❌ Error: CSV file not found: {csv_file}")
        return 1

    if not video_file or not video_file.exists():
        print(f"❌ Error: Video file not found in {participant_dir}")
        print(f"   Expected: {args.session}.mp4 or {args.session}.mov")
        return 1

    # Check file sizes (Git LFS check)
    csv_size = csv_file.stat().st_size
    video_size = video_file.stat().st_size

    if csv_size < 200:  # Likely a Git LFS pointer
        print(f"⚠️  Warning: CSV file appears to be a Git LFS pointer ({csv_size} bytes)")
        print("   Run 'git lfs pull' to download actual files")
        return 1

    if video_size < 1000:  # Likely a Git LFS pointer
        print(f"⚠️  Warning: Video file appears to be a Git LFS pointer ({video_size} bytes)")
        print("   Run 'git lfs pull' to download actual files")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{args.session}_3frame_mapped.csv"

    # Configure for maximum speed with 3-frame interval
    config = {
        'video': str(video_file),
        'csv': str(csv_file),
        'interval': 3,  # 🚀 3-frame interval for speed
        'confidence': 60,  # Balanced confidence threshold
        'distance': 10,  # Distance threshold in pixels
        'output': str(output_file),
        'fast_mode': True,  # 🚀 Use fast OCR mode
        'batch_size': 128,  # 🚀 Large batch size for GPU efficiency
        'chunk_size': 200,  # 🚀 Moderate chunk size for memory management
        'encoding': 'utf-8',
        'word_level': True,
        'in_memory': True,  # 🚀 Use in-memory processing (faster)
        'debug': False,
        'threads': 12,  # 🚀 Maximum parallelism
    }

    print("\n" + "="*80)
    print("🚀 3-FRAME INTERVAL SPEED TEST")
    print("="*80)
    print(f"\nParticipant: {args.participant}")
    print(f"Session: {args.session}")
    print(f"CSV: {csv_file} ({csv_size / 1024 / 1024:.1f} MB)")
    print(f"Video: {video_file} ({video_size / 1024 / 1024:.1f} MB)")
    print(f"Output: {output_file}")
    print("\n🎛️  Speed Optimizations:")
    print(f"  • Frame interval: {config['interval']} (process every {config['interval']}rd frame)")
    print(f"  • Fast OCR mode: {config['fast_mode']}")
    print(f"  • Batch size: {config['batch_size']}")
    print(f"  • Chunk size: {config['chunk_size']}")
    print(f"  • In-memory processing: {config['in_memory']}")
    print(f"  • Threads: {config['threads']}")
    print("="*80 + "\n")

    # Run the pipeline
    start_time = time.time()
    success = process_with_args(config)
    total_time = time.time() - start_time

    if success:
        print("\n" + "="*80)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"⏱️  Total time: {total_time:.2f}s")

        if output_file.exists():
            result_size = output_file.stat().st_size
            print(f"📊 Result file: {output_file} ({result_size / 1024:.1f} KB)")

            # Quick analysis
            import pandas as pd
            try:
                df = pd.read_csv(output_file)
                print(f"📈 Mapped gaze points: {len(df)}")
                if 'confidence' in df.columns:
                    print(f"📈 Average confidence: {df['confidence'].mean():.1f}%")
                if 'distance' in df.columns:
                    print(f"📈 Average distance: {df['distance'].mean():.2f} px")
            except Exception as e:
                print(f"⚠️  Could not analyze results: {e}")

        print("="*80 + "\n")
        return 0
    else:
        print("\n❌ TEST FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
