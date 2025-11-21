#!/usr/bin/env python3
"""
Generate progress map with intelligent downsampling to avoid overlapping.

Features:
- Spatial downsampling to prevent overlapping markers
- Temporal downsampling to show reading progression
- Color-coded by time for progress visualization
- Heatmap overlay option
- Text annotation overlay
- Interactive and static output modes
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import Normalize
import cv2


class ProgressMapGenerator:
    """Generate downsampled progress maps from gaze tracking data."""

    def __init__(self, video_path: str, mapped_csv: str, ocr_csv: Optional[str] = None):
        """
        Initialize progress map generator.

        Args:
            video_path: Path to source video
            mapped_csv: Path to mapped gaze data CSV
            ocr_csv: Optional path to OCR results CSV
        """
        self.video_path = video_path
        self.mapped_csv = mapped_csv
        self.ocr_csv = ocr_csv

        # Load data
        self.mapped_data = pd.read_csv(mapped_csv)
        if ocr_csv and os.path.exists(ocr_csv):
            self.ocr_data = pd.read_csv(ocr_csv)
        else:
            self.ocr_data = None

        # Get video dimensions
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            # Use defaults if video not available
            self.video_width = 1920
            self.video_height = 1080
            self.fps = 30
            self.total_frames = 1000

        print(f"📹 Video: {self.video_width}x{self.video_height} @ {self.fps} fps")
        print(f"📊 Mapped gaze points: {len(self.mapped_data)}")
        if self.ocr_data is not None:
            print(f"📝 OCR entries: {len(self.ocr_data)}")

    def spatial_downsample(self, data: pd.DataFrame, grid_size: int = 20) -> pd.DataFrame:
        """
        Apply spatial downsampling to prevent overlapping points.

        Creates a grid and selects one representative point per cell.

        Args:
            data: DataFrame with gaze_x, gaze_y columns
            grid_size: Size of grid cells in pixels

        Returns:
            Downsampled DataFrame
        """
        print(f"\n🔽 Applying spatial downsampling (grid_size={grid_size}px)...")

        # Create grid indices
        data = data.copy()
        data['grid_x'] = (data['gaze_x'] // grid_size).astype(int)
        data['grid_y'] = (data['gaze_y'] // grid_size).astype(int)

        # Group by grid cell and select representative point
        # Use the point with highest confidence in each cell
        if 'confidence' in data.columns:
            downsampled = data.loc[data.groupby(['grid_x', 'grid_y'])['confidence'].idxmax()]
        else:
            # Or just take the first point in each cell
            downsampled = data.groupby(['grid_x', 'grid_y']).first().reset_index()

        print(f"   Before: {len(data)} points")
        print(f"   After: {len(downsampled)} points ({len(downsampled)/len(data)*100:.1f}% retained)")
        print(f"   Reduction: {len(data) - len(downsampled)} points removed")

        return downsampled.drop(columns=['grid_x', 'grid_y'])

    def temporal_downsample(self, data: pd.DataFrame, interval: int = 10) -> pd.DataFrame:
        """
        Apply temporal downsampling to show progression clearly.

        Args:
            data: DataFrame with timestamp column
            interval: Keep every Nth point in time order

        Returns:
            Downsampled DataFrame
        """
        print(f"\n⏱️  Applying temporal downsampling (interval={interval})...")

        # Sort by timestamp
        data_sorted = data.sort_values('timestamp').reset_index(drop=True)

        # Select every Nth point
        downsampled = data_sorted.iloc[::interval].reset_index(drop=True)

        print(f"   Before: {len(data)} points")
        print(f"   After: {len(downsampled)} points ({len(downsampled)/len(data)*100:.1f}% retained)")

        return downsampled

    def adaptive_downsample(self, data: pd.DataFrame, target_points: int = 500) -> pd.DataFrame:
        """
        Adaptively downsample to achieve target number of points.

        Combines spatial and temporal downsampling intelligently.

        Args:
            data: DataFrame with gaze data
            target_points: Target number of points to display

        Returns:
            Downsampled DataFrame
        """
        print(f"\n🎯 Adaptive downsampling to ~{target_points} points...")

        current_size = len(data)

        if current_size <= target_points:
            print(f"   No downsampling needed ({current_size} ≤ {target_points})")
            return data

        # Calculate reduction factor needed
        reduction_factor = current_size / target_points

        # Split between spatial and temporal downsampling
        # Use more spatial downsampling for dense data
        spatial_factor = np.sqrt(reduction_factor)
        temporal_factor = reduction_factor / spatial_factor

        # Apply spatial downsampling
        grid_size = int(max(10, 20 * spatial_factor))
        data = self.spatial_downsample(data, grid_size=grid_size)

        # Apply temporal downsampling if still too many points
        if len(data) > target_points:
            interval = int(np.ceil(len(data) / target_points))
            data = self.temporal_downsample(data, interval=interval)

        print(f"   Final: {len(data)} points")
        return data

    def generate_progress_map(
        self,
        output_path: str,
        downsample_method: str = 'adaptive',
        target_points: int = 500,
        show_trajectory: bool = True,
        show_text: bool = True,
        show_heatmap: bool = False,
        background_frame: Optional[int] = None,
        figsize: Tuple[int, int] = (16, 10)
    ):
        """
        Generate progress map visualization.

        Args:
            output_path: Path to save output image
            downsample_method: 'adaptive', 'spatial', 'temporal', or 'none'
            target_points: Target number of points (for adaptive)
            show_trajectory: Show reading trajectory lines
            show_text: Show text annotations
            show_heatmap: Show heatmap overlay
            background_frame: Frame number to use as background (None = blank)
            figsize: Figure size in inches
        """
        print("\n" + "="*80)
        print("🗺️  GENERATING PROGRESS MAP")
        print("="*80)

        # Apply downsampling
        data = self.mapped_data.copy()

        if downsample_method == 'adaptive':
            data = self.adaptive_downsample(data, target_points=target_points)
        elif downsample_method == 'spatial':
            data = self.spatial_downsample(data, grid_size=20)
        elif downsample_method == 'temporal':
            data = self.temporal_downsample(data, interval=10)
        elif downsample_method == 'none':
            print("\n⚠️  No downsampling applied")
        else:
            raise ValueError(f"Unknown downsample method: {downsample_method}")

        # Sort by timestamp for trajectory
        data = data.sort_values('timestamp').reset_index(drop=True)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=100)

        # Set background
        if background_frame is not None and os.path.exists(self.video_path):
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, background_frame)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ax.imshow(frame_rgb, extent=[0, self.video_width, self.video_height, 0], alpha=0.3)
            cap.release()
        else:
            ax.set_facecolor('#f0f0f0')

        # Show text boxes from OCR
        if show_text and self.ocr_data is not None:
            print("\n📝 Drawing OCR text boxes...")
            # Get unique frames in the mapped data
            unique_frames = data['frame_num'].unique()
            ocr_subset = self.ocr_data[self.ocr_data['frame'].isin(unique_frames)]

            for _, row in ocr_subset.iterrows():
                if pd.notna(row['x']) and pd.notna(row['y']):
                    rect = patches.Rectangle(
                        (row['x'], row['y']),
                        row['w'], row['h'],
                        linewidth=0.5,
                        edgecolor='gray',
                        facecolor='none',
                        alpha=0.3
                    )
                    ax.add_patch(rect)

        # Create heatmap if requested
        if show_heatmap:
            print("\n🔥 Generating heatmap...")
            heatmap = self.create_heatmap(data)
            ax.imshow(heatmap, extent=[0, self.video_width, self.video_height, 0],
                     alpha=0.5, cmap='hot', interpolation='gaussian')

        # Normalize timestamps for color mapping
        if len(data) > 0:
            timestamps = data['timestamp'].values
            time_norm = Normalize(vmin=timestamps.min(), vmax=timestamps.max())
            colormap = cm.get_cmap('viridis')

            # Draw trajectory lines
            if show_trajectory and len(data) > 1:
                print("\n🎨 Drawing trajectory...")
                points = data[['gaze_x', 'gaze_y']].values
                segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)

                # Color segments by time
                colors = [colormap(time_norm(t)) for t in timestamps[:-1]]

                lc = LineCollection(segments, colors=colors, linewidths=1, alpha=0.5)
                ax.add_collection(lc)

            # Draw gaze points
            print(f"\n🎯 Drawing {len(data)} gaze points...")
            scatter = ax.scatter(
                data['gaze_x'],
                data['gaze_y'],
                c=timestamps,
                cmap='viridis',
                s=30,
                alpha=0.7,
                edgecolors='white',
                linewidths=0.5,
                norm=time_norm,
                zorder=5
            )

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, label='Time (ms)', pad=0.02)
            cbar.ax.tick_params(labelsize=8)

        # Configure axes
        ax.set_xlim(0, self.video_width)
        ax.set_ylim(self.video_height, 0)  # Invert Y axis
        ax.set_aspect('equal')
        ax.set_xlabel('X Position (pixels)', fontsize=10)
        ax.set_ylabel('Y Position (pixels)', fontsize=10)

        # Add title with statistics
        if len(data) > 0:
            duration_s = (data['timestamp'].max() - data['timestamp'].min()) / 1_000_000
            title = (f"Reading Progress Map\n"
                    f"{len(data)} points | Duration: {duration_s:.1f}s | "
                    f"Downsample: {downsample_method}")
        else:
            title = "Reading Progress Map (No Data)"

        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)

        # Add grid
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

        # Tight layout
        plt.tight_layout()

        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Progress map saved to: {output_path}")

        # Get file size
        file_size = os.path.getsize(output_path) / 1024
        print(f"   File size: {file_size:.1f} KB")

        plt.close()

    def create_heatmap(self, data: pd.DataFrame, sigma: float = 30) -> np.ndarray:
        """
        Create heatmap from gaze points.

        Args:
            data: DataFrame with gaze_x, gaze_y
            sigma: Gaussian blur sigma for smoothing

        Returns:
            Heatmap as 2D array
        """
        # Create empty heatmap
        heatmap = np.zeros((self.video_height, self.video_width), dtype=np.float32)

        # Add gaussian blobs at each gaze point
        for _, row in data.iterrows():
            x, y = int(row['gaze_x']), int(row['gaze_y'])
            if 0 <= x < self.video_width and 0 <= y < self.video_height:
                heatmap[y, x] += 1

        # Apply gaussian blur
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def generate_comparison_figure(self, output_path: str):
        """Generate figure comparing different downsampling methods."""
        print("\n" + "="*80)
        print("📊 GENERATING COMPARISON FIGURE")
        print("="*80)

        methods = [
            ('none', 'No Downsampling'),
            ('temporal', 'Temporal (every 10th)'),
            ('spatial', 'Spatial (20px grid)'),
            ('adaptive', 'Adaptive (~500 points)')
        ]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, (method, title) in enumerate(methods):
            ax = axes[idx]
            print(f"\n📈 Method {idx+1}/4: {title}")

            # Apply downsampling
            data = self.mapped_data.copy()
            if method == 'adaptive':
                data = self.adaptive_downsample(data, target_points=500)
            elif method == 'spatial':
                data = self.spatial_downsample(data, grid_size=20)
            elif method == 'temporal':
                data = self.temporal_downsample(data, interval=10)

            # Sort by timestamp
            data = data.sort_values('timestamp').reset_index(drop=True)

            if len(data) > 0:
                # Normalize timestamps
                timestamps = data['timestamp'].values
                time_norm = Normalize(vmin=timestamps.min(), vmax=timestamps.max())

                # Draw trajectory
                if len(data) > 1:
                    points = data[['gaze_x', 'gaze_y']].values
                    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
                    colormap = cm.get_cmap('viridis')
                    colors = [colormap(time_norm(t)) for t in timestamps[:-1]]
                    lc = LineCollection(segments, colors=colors, linewidths=1, alpha=0.4)
                    ax.add_collection(lc)

                # Draw points
                scatter = ax.scatter(
                    data['gaze_x'],
                    data['gaze_y'],
                    c=timestamps,
                    cmap='viridis',
                    s=20,
                    alpha=0.7,
                    edgecolors='white',
                    linewidths=0.3,
                    norm=time_norm
                )

            # Configure axes
            ax.set_xlim(0, self.video_width)
            ax.set_ylim(self.video_height, 0)
            ax.set_aspect('equal')
            ax.set_title(f"{title}\n({len(data)} points)", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.2)
            ax.set_xlabel('X (px)')
            ax.set_ylabel('Y (px)')

        plt.suptitle('Progress Map - Downsampling Comparison', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Comparison saved to: {output_path}")
        plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate progress map with intelligent downsampling"
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--mapped-csv", required=True, help="Path to mapped gaze CSV")
    parser.add_argument("--ocr-csv", help="Path to OCR results CSV (optional)")
    parser.add_argument("--output", default="progress_map.png", help="Output image path")
    parser.add_argument(
        "--downsample",
        choices=['none', 'spatial', 'temporal', 'adaptive'],
        default='adaptive',
        help="Downsampling method"
    )
    parser.add_argument("--target-points", type=int, default=500,
                       help="Target points for adaptive downsampling")
    parser.add_argument("--show-trajectory", action="store_true", default=True,
                       help="Show reading trajectory")
    parser.add_argument("--show-text", action="store_true", default=True,
                       help="Show OCR text boxes")
    parser.add_argument("--show-heatmap", action="store_true",
                       help="Show heatmap overlay")
    parser.add_argument("--background-frame", type=int,
                       help="Frame number for background image")
    parser.add_argument("--comparison", action="store_true",
                       help="Generate comparison figure of all methods")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.mapped_csv):
        print(f"❌ Error: Mapped CSV not found: {args.mapped_csv}")
        return 1

    # Create generator
    generator = ProgressMapGenerator(
        video_path=args.video,
        mapped_csv=args.mapped_csv,
        ocr_csv=args.ocr_csv
    )

    # Generate comparison or single map
    if args.comparison:
        output = args.output.replace('.png', '_comparison.png')
        generator.generate_comparison_figure(output)
    else:
        generator.generate_progress_map(
            output_path=args.output,
            downsample_method=args.downsample,
            target_points=args.target_points,
            show_trajectory=args.show_trajectory,
            show_text=args.show_text and args.ocr_csv is not None,
            show_heatmap=args.show_heatmap,
            background_frame=args.background_frame
        )

    print("\n" + "="*80)
    print("✅ PROGRESS MAP GENERATION COMPLETE")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
