#!/usr/bin/env python3
"""
Utility script to check OCR results from the eye-tracking to text mapping process.
Displays OCR bounding boxes on video frames to verify text recognition quality.
"""

import argparse
import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def display_ocr_results(video_path, ocr_csv, frame_num=None, output_image=None):
    """
    Display OCR results on a specific frame from the video.
    If frame_num is None, uses the first frame that has OCR results.
    """
    # Load OCR data
    try:
        ocr_data = pd.read_csv(ocr_csv)
        print(f"Loaded {len(ocr_data)} OCR entries from {ocr_csv}")
    except Exception as e:
        print(f"Error loading OCR data: {e}")
        return
    
    # Make sure required columns exist
    required_cols = ['frame_num', 'text', 'conf', 'left', 'top', 'width', 'height']
    if not all(col in ocr_data.columns for col in required_cols):
        print(f"Missing required columns in OCR data. Available columns: {ocr_data.columns.tolist()}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video dimensions: {frame_width}x{frame_height}, Total frames: {total_frames}")
    
    # If frame_num is not specified, find the first frame with OCR data
    if frame_num is None:
        available_frames = ocr_data['frame_num'].unique()
        if len(available_frames) == 0:
            print("No frames with OCR data found")
            cap.release()
            return
        
        frame_num = int(available_frames[0])
        print(f"Using first available frame: {frame_num}")
    
    # Check if the requested frame has OCR data
    frame_ocr = ocr_data[ocr_data['frame_num'] == frame_num]
    if len(frame_ocr) == 0:
        print(f"No OCR data found for frame {frame_num}")
        # Try to find a frame with OCR data
        available_frames = ocr_data['frame_num'].unique()
        if len(available_frames) > 0:
            alt_frame = int(available_frames[0])
            print(f"Try using frame {alt_frame} which has OCR data")
        cap.release()
        return
    
    # Read the requested frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error reading frame {frame_num}")
        return
    
    # Convert BGR to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=(16, 9))
    plt.imshow(frame_rgb)
    
    # Add OCR boxes
    for _, row in frame_ocr.iterrows():
        # Get bounding box coordinates
        left = row['left']
        top = row['top']
        width = row['width']
        height = row['height']
        text = row['text']
        conf = row['conf']
        
        # Draw rectangle
        color = 'green' if conf >= 70 else ('yellow' if conf >= 40 else 'red')
        rect = Rectangle(
            (left, top), width, height,
            linewidth=1, edgecolor=color, facecolor='none', alpha=0.7
        )
        plt.gca().add_patch(rect)
        
        # Add text label with confidence
        plt.text(
            left, top - 5,
            f"{text} ({conf:.1f}%)",
            color=color, fontsize=8, alpha=0.8,
            bbox=dict(facecolor='white', alpha=0.5, pad=0)
        )
    
    # Add separator line if present in OCR data
    if 'category' in frame_ocr.columns:
        # Calculate average separator position
        try:
            source_rows = frame_ocr[frame_ocr['category'] == 'source']
            target_rows = frame_ocr[frame_ocr['category'] == 'target']
            
            if len(source_rows) > 0 and len(target_rows) > 0:
                # Get average y position for source and target
                avg_source_y = source_rows['top'].mean() + source_rows['height'].mean()
                avg_target_y = target_rows['top'].mean()
                
                # Draw separator line
                separator_y = (avg_source_y + avg_target_y) / 2
                plt.axhline(y=separator_y, color='blue', linestyle='--', alpha=0.7)
                plt.text(
                    10, separator_y - 10,
                    "Source/Target Separator",
                    color='blue', fontsize=10, alpha=0.8,
                    bbox=dict(facecolor='white', alpha=0.5, pad=1)
                )
        except Exception as e:
            print(f"Error drawing separator: {e}")
    
    # Set title
    plt.title(f"OCR Results - Frame {frame_num} - {len(frame_ocr)} text entries")
    
    # Add statistics
    avg_conf = frame_ocr['conf'].mean()
    high_conf = len(frame_ocr[frame_ocr['conf'] >= 70])
    med_conf = len(frame_ocr[(frame_ocr['conf'] >= 40) & (frame_ocr['conf'] < 70)])
    low_conf = len(frame_ocr[frame_ocr['conf'] < 40])
    
    stats_text = (
        f"Average confidence: {avg_conf:.1f}%\n"
        f"High confidence (≥70%): {high_conf} entries\n"
        f"Medium confidence (40-70%): {med_conf} entries\n"
        f"Low confidence (<40%): {low_conf} entries\n"
    )
    
    plt.figtext(
        0.02, 0.02, stats_text,
        bbox=dict(facecolor='white', alpha=0.8, pad=5)
    )
    
    # Remove axes
    plt.axis('off')
    
    # Save or display
    if output_image:
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_image}")
    else:
        plt.show()
    
    # Close figure
    plt.close()

def main():
    """Main function to check OCR results."""
    parser = argparse.ArgumentParser(description="Check OCR results on video frames")
    parser.add_argument("-v", "--video", default="沈若枢/User1_241227135925/User1_241227135925.mp4", 
                       help="Path to video file")
    parser.add_argument("-o", "--ocr", default="ocr_results.csv", 
                       help="Path to OCR results CSV")
    parser.add_argument("-f", "--frame", type=int, default=None, 
                       help="Frame number to check (default: first frame with OCR)")
    parser.add_argument("-s", "--save", default=None, 
                       help="Save visualization to image file instead of displaying")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.isfile(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    if not os.path.isfile(args.ocr):
        print(f"Error: OCR file not found: {args.ocr}")
        return
    
    # Display OCR results
    display_ocr_results(args.video, args.ocr, args.frame, args.save)

if __name__ == "__main__":
    main() 