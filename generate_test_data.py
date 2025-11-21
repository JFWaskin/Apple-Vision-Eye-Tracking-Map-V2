#!/usr/bin/env python3
"""
Generate synthetic test data for validating the pipeline on Linux VM.

Creates realistic eye-tracking data and OCR results for testing.
"""

import argparse
import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path


class SyntheticDataGenerator:
    """Generate synthetic eye-tracking and OCR data."""

    def __init__(self, video_width=1920, video_height=1080, fps=30, duration_s=60):
        self.video_width = video_width
        self.video_height = video_height
        self.fps = fps
        self.duration_s = duration_s
        self.total_frames = int(fps * duration_s)

    def generate_reading_pattern(self, num_lines=10, words_per_line=8):
        """
        Generate realistic reading pattern (left-to-right, top-to-bottom).

        Args:
            num_lines: Number of text lines
            words_per_line: Words per line

        Returns:
            List of (x, y, timestamp_us) tuples
        """
        gaze_points = []

        # Text area parameters
        margin_x = 200
        margin_y = 150
        line_height = 80
        word_width = 180
        word_spacing = 20

        start_time = 0
        time_per_word_ms = 300  # 300ms per word (realistic reading speed)

        for line_idx in range(num_lines):
            y = margin_y + line_idx * line_height

            for word_idx in range(words_per_line):
                x = margin_x + word_idx * (word_width + word_spacing)

                # Add some natural jitter
                x_jitter = np.random.normal(0, 5)
                y_jitter = np.random.normal(0, 3)

                # Generate multiple fixations per word (realistic)
                num_fixations = np.random.randint(2, 5)
                for fix_idx in range(num_fixations):
                    timestamp_us = start_time * 1000 + fix_idx * 50000  # 50ms between fixations

                    gaze_points.append({
                        'x': x + x_jitter,
                        'y': y + y_jitter,
                        'timestamp_us': int(timestamp_us)
                    })

                start_time += time_per_word_ms

            # Line return (saccade back to left)
            start_time += 100  # 100ms for saccade

        return gaze_points

    def generate_gaze_csv(self, output_path, num_lines=10, words_per_line=8):
        """
        Generate synthetic eye-tracking CSV file.

        Mimics the format of real participant data.
        """
        print(f"\n📊 Generating synthetic gaze data...")

        # Generate reading pattern
        gaze_points = self.generate_reading_pattern(num_lines, words_per_line)

        # Add noise and invalid points (realistic)
        total_points = len(gaze_points) * 3  # Add some invalid points
        noise_points = total_points - len(gaze_points)

        for _ in range(noise_points):
            gaze_points.append({
                'x': np.random.uniform(0, self.video_width),
                'y': np.random.uniform(0, self.video_height),
                'timestamp_us': int(np.random.uniform(0, self.duration_s * 1e6))
            })

        # Create DataFrame
        df = pd.DataFrame(gaze_points)

        # Add required columns
        df['timestampUs'] = df['timestamp_us']
        df['leftEyeX'] = df['x'] + np.random.normal(0, 2, len(df))
        df['leftEyeY'] = df['y'] + np.random.normal(0, 2, len(df))
        df['rightEyeX'] = df['x'] + np.random.normal(0, 2, len(df))
        df['rightEyeY'] = df['y'] + np.random.normal(0, 2, len(df))

        # Sort by timestamp
        df = df.sort_values('timestampUs').reset_index(drop=True)

        # Save
        df.to_csv(output_path, index=False)
        print(f"   ✅ Saved {len(df)} gaze points to {output_path}")
        print(f"   Duration: {df['timestampUs'].max() / 1e6:.1f}s")

        return df

    def generate_ocr_csv(self, output_path, num_lines=10, words_per_line=8):
        """
        Generate synthetic OCR results CSV.

        Creates text boxes matching the reading pattern.
        """
        print(f"\n📝 Generating synthetic OCR data...")

        ocr_entries = []

        # Text area parameters (same as gaze generation)
        margin_x = 200
        margin_y = 150
        line_height = 80
        word_width = 180
        word_spacing = 20

        words = [
            "The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "Hello", "world", "this", "is", "test", "data", "for", "OCR",
            "Reading", "pattern", "analysis", "with", "eye", "tracking", "system", "works",
            "Python", "code", "generates", "synthetic", "realistic", "gaze", "points", "accurately"
        ]

        frame_num = 0
        for line_idx in range(num_lines):
            y = margin_y + line_idx * line_height

            for word_idx in range(words_per_line):
                x = margin_x + word_idx * (word_width + word_spacing)
                word_text = words[(line_idx * words_per_line + word_idx) % len(words)]

                ocr_entries.append({
                    'frame': frame_num,
                    'text': word_text,
                    'conf': np.random.uniform(80, 99),
                    'x': x,
                    'y': y,
                    'w': word_width,
                    'h': 60,
                    'category': 'source' if y < self.video_height / 2 else 'target'
                })

            frame_num += 30  # Change text every 30 frames

        # Create DataFrame
        df = pd.DataFrame(ocr_entries)
        df.to_csv(output_path, index=False)

        print(f"   ✅ Saved {len(df)} OCR entries to {output_path}")
        return df

    def generate_video(self, output_path, num_lines=10, words_per_line=8):
        """
        Generate synthetic video with text content.

        Args:
            output_path: Path to save video
            num_lines: Number of text lines
            words_per_line: Words per line
        """
        print(f"\n🎥 Generating synthetic video...")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.fps,
                                (self.video_width, self.video_height))

        # Text parameters
        margin_x = 200
        margin_y = 150
        line_height = 80
        word_width = 180
        word_spacing = 20

        words = [
            "The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "Hello", "world", "this", "is", "test", "data", "for", "OCR",
            "Reading", "pattern", "analysis", "with", "eye", "tracking", "system", "works",
            "Python", "code", "generates", "synthetic", "realistic", "gaze", "points", "accurately"
        ]

        for frame_idx in range(self.total_frames):
            # Create white background
            frame = np.ones((self.video_height, self.video_width, 3), dtype=np.uint8) * 255

            # Draw text
            for line_idx in range(num_lines):
                y = margin_y + line_idx * line_height

                for word_idx in range(words_per_line):
                    x = margin_x + word_idx * (word_width + word_spacing)
                    word_text = words[(line_idx * words_per_line + word_idx) % len(words)]

                    # Draw word
                    cv2.putText(frame, word_text, (x, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

                    # Draw bounding box (faint)
                    cv2.rectangle(frame, (x-5, y-50), (x+word_width, y+10),
                                (200, 200, 200), 1)

            # Add frame number
            cv2.putText(frame, f"Frame {frame_idx}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

            writer.write(frame)

            if (frame_idx + 1) % 100 == 0:
                print(f"   Progress: {frame_idx + 1}/{self.total_frames} frames", end='\r')

        writer.release()
        print(f"\n   ✅ Saved {self.total_frames} frames to {output_path}")

        # Get file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   File size: {size_mb:.1f} MB")

    def generate_mapped_csv(self, gaze_df, ocr_df, output_path):
        """
        Generate synthetic mapped gaze-to-text CSV.

        Simulates the output of the mapping pipeline.
        """
        print(f"\n🎯 Generating synthetic mapped data...")

        mapped_data = []

        for _, gaze_row in gaze_df.iterrows():
            gaze_x = gaze_row['x']
            gaze_y = gaze_row['y']

            # Find closest OCR entry
            distances = np.sqrt(
                (ocr_df['x'] + ocr_df['w']/2 - gaze_x)**2 +
                (ocr_df['y'] + ocr_df['h']/2 - gaze_y)**2
            )

            if len(distances) > 0 and distances.min() < 100:  # Within 100px
                closest_idx = distances.idxmin()
                ocr_row = ocr_df.iloc[closest_idx]

                mapped_data.append({
                    'timestamp': gaze_row['timestampUs'],
                    'frame_num': ocr_row['frame'],
                    'gaze_x': gaze_x,
                    'gaze_y': gaze_y,
                    'text': ocr_row['text'],
                    'confidence': ocr_row['conf'],
                    'distance': distances.min(),
                    'category': ocr_row['category']
                })

        df = pd.DataFrame(mapped_data)
        df.to_csv(output_path, index=False)

        print(f"   ✅ Saved {len(df)} mapped points to {output_path}")
        print(f"   Mapping rate: {len(df) / len(gaze_df) * 100:.1f}%")

        return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test data")
    parser.add_argument("--output-dir", default="test_data", help="Output directory")
    parser.add_argument("--duration", type=int, default=10, help="Video duration (seconds)")
    parser.add_argument("--lines", type=int, default=10, help="Number of text lines")
    parser.add_argument("--words-per-line", type=int, default=8, help="Words per line")
    parser.add_argument("--skip-video", action="store_true", help="Skip video generation (faster)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("🔧 SYNTHETIC DATA GENERATOR")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Duration: {args.duration}s")
    print(f"Text layout: {args.lines} lines × {args.words_per_line} words")

    # Create generator
    generator = SyntheticDataGenerator(
        video_width=1920,
        video_height=1080,
        fps=30,
        duration_s=args.duration
    )

    # Generate OCR data first (needed for mapping)
    ocr_csv = output_dir / "ocr_results.csv"
    ocr_df = generator.generate_ocr_csv(ocr_csv, args.lines, args.words_per_line)

    # Generate gaze data
    gaze_csv = output_dir / "raw_gaze.csv"
    gaze_df = generator.generate_gaze_csv(gaze_csv, args.lines, args.words_per_line)

    # Generate mapped data
    mapped_csv = output_dir / "mapped_gaze.csv"
    mapped_df = generator.generate_mapped_csv(gaze_df, ocr_df, mapped_csv)

    # Generate video (optional, slow)
    if not args.skip_video:
        video_path = output_dir / "test_video.mp4"
        generator.generate_video(video_path, args.lines, args.words_per_line)
    else:
        print("\n⏭️  Skipping video generation (use --skip-video=False to generate)")
        # Create dummy video file for path validation
        video_path = output_dir / "test_video.mp4"
        video_path.touch()

    print("\n" + "="*80)
    print("✅ SYNTHETIC DATA GENERATION COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  📊 Gaze data: {gaze_csv}")
    print(f"  📝 OCR data: {ocr_csv}")
    print(f"  🎯 Mapped data: {mapped_csv}")
    print(f"  🎥 Video: {video_path}")

    print(f"\n📋 Summary:")
    print(f"  Gaze points: {len(gaze_df)}")
    print(f"  OCR entries: {len(ocr_df)}")
    print(f"  Mapped points: {len(mapped_df)}")
    print(f"  Mapping rate: {len(mapped_df) / len(gaze_df) * 100:.1f}%")

    print("\n🚀 Ready to test! Run:")
    print(f"  python generate_progress_map.py \\")
    print(f"    --video {video_path} \\")
    print(f"    --mapped-csv {mapped_csv} \\")
    print(f"    --ocr-csv {ocr_csv} \\")
    print(f"    --comparison")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
