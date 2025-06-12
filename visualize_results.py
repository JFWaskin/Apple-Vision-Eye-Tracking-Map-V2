#!/usr/bin/env python3
"""
Visualize mapping results from the eye-tracking to text mapping process.
This script creates visualizations to help analyze the mapping quality.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_mapping_distribution(mapped_data, output_file="mapped_visualization.png"):
    """
    Create visualization of the mapping distribution between gaze points and text.
    """
    # Check if mapped data is available
    if mapped_data is None or len(mapped_data) == 0:
        print("No mapping data available for visualization")
        return
    
    print(f"Creating visualization with {len(mapped_data)} mapped points...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Eye-Tracking to Text Mapping Analysis", fontsize=16)
    
    # 1. Scatter plot of gaze positions
    ax1 = axes[0, 0]
    ax1.scatter(mapped_data['gaze_x'], mapped_data['gaze_y'], 
                alpha=0.5, s=3, c=mapped_data['confidence'], cmap='viridis')
    ax1.set_title("Mapped Gaze Positions")
    ax1.set_xlabel("X Position (pixels)")
    ax1.set_ylabel("Y Position (pixels)")
    ax1.set_xlim(0, 1920)  # Assuming 1080p video
    ax1.set_ylim(0, 1080)
    ax1.invert_yaxis()  # Invert Y-axis to match image coordinates
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram of mapping distances
    ax2 = axes[0, 1]
    ax2.hist(mapped_data['distance'], bins=50, alpha=0.7, color='blue')
    ax2.set_title("Mapping Distance Distribution")
    ax2.set_xlabel("Distance (pixels)")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)
    
    # Add vertical line for median
    median_distance = mapped_data['distance'].median()
    ax2.axvline(median_distance, color='red', linestyle='dashed', linewidth=2)
    ax2.text(median_distance * 1.1, ax2.get_ylim()[1] * 0.9, 
             f'Median: {median_distance:.1f} px', color='red')
    
    # 3. Source vs Target distribution
    ax3 = axes[1, 0]
    category_counts = mapped_data['category'].value_counts()
    if not category_counts.empty:
        categories = category_counts.index.tolist()
        counts = category_counts.values
        ax3.bar(categories, counts, alpha=0.7, color=['blue', 'green'])
        ax3.set_title("Source vs Target Text Distribution")
        ax3.set_xlabel("Text Category")
        ax3.set_ylabel("Count")
        
        # Add percentage labels
        total = sum(counts)
        for i, count in enumerate(counts):
            percentage = count / total * 100
            ax3.text(i, count + (max(counts) * 0.02), 
                    f'{count}\n({percentage:.1f}%)', 
                    ha='center', va='bottom')
    else:
        ax3.text(0.5, 0.5, "No category data available", 
                ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Confidence distribution
    ax4 = axes[1, 1]
    ax4.hist(mapped_data['confidence'], bins=20, alpha=0.7, color='green')
    ax4.set_title("OCR Confidence Distribution")
    ax4.set_xlabel("Confidence (%)")
    ax4.set_ylabel("Frequency")
    ax4.grid(True, alpha=0.3)
    
    # Add vertical line for mean confidence
    mean_conf = mapped_data['confidence'].mean()
    ax4.axvline(mean_conf, color='red', linestyle='dashed', linewidth=2)
    ax4.text(mean_conf * 0.9, ax4.get_ylim()[1] * 0.9, 
             f'Mean: {mean_conf:.1f}%', color='red')
    
    # Add mapping statistics as text
    mapping_stats = (
        f"Total mapped points: {len(mapped_data)}\n"
        f"Average distance: {mapped_data['distance'].mean():.2f} pixels\n"
        f"Average confidence: {mapped_data['confidence'].mean():.2f}%\n"
        f"Unique text entries: {mapped_data['text'].nunique()}\n"
    )
    
    fig.text(0.5, 0.02, mapping_stats, ha='center', 
            bbox=dict(boxstyle='round', alpha=0.1))
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    
    # Close the figure to free memory
    plt.close(fig)

def main():
    """Main function to run visualization."""
    parser = argparse.ArgumentParser(description="Visualize eye-tracking to text mapping results")
    parser.add_argument("-i", "--input", default="mapped_gaze_to_text.csv", 
                        help="Input CSV file with mapping results")
    parser.add_argument("-o", "--output", default="mapped_visualization.png", 
                        help="Output PNG file")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Load mapping data
    try:
        mapped_data = pd.read_csv(args.input)
        print(f"Loaded {len(mapped_data)} mapped points from {args.input}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create visualization
    visualize_mapping_distribution(mapped_data, args.output)

if __name__ == "__main__":
    main() 