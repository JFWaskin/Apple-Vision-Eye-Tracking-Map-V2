#!/usr/bin/env python3
"""
Utility functions for the eye-tracking analysis system.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import glob
import random
import tkinter as tk
from tkinter import ttk

def find_video_files(directory):
    """
    Find all video files in a directory (recursively).
    
    Args:
        directory (str): Directory to search
        
    Returns:
        list: List of video file paths
    """
    video_extensions = ('.mp4', '.avi', '.mov')
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(directory, f"**/*{ext}"), recursive=True))
    
    return sorted(video_files)

def find_csv_files(directory):
    """
    Find all CSV files in a directory (recursively).
    
    Args:
        directory (str): Directory to search
        
    Returns:
        list: List of CSV file paths
    """
    csv_files = glob.glob(os.path.join(directory, "**/*.csv"), recursive=True)
    return sorted(csv_files)

def count_files_in_directory(directory):
    """
    Count files in directory by type.
    
    Args:
        directory (str): Directory to search
        
    Returns:
        dict: Dictionary containing counts for video, csv, and other files
    """
    counts = {
        'video': 0,
        'csv': 0,
        'other': 0
    }
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                counts['video'] += 1
            elif file.lower().endswith('.csv'):
                counts['csv'] += 1
            else:
                counts['other'] += 1
    
    return counts

def generate_output_path(base_dir, filename, timestamp=True):
    """
    Generate an output file path with optional timestamp.
    
    Args:
        base_dir (str): Base directory
        filename (str): Filename
        timestamp (bool): Whether to include timestamp
        
    Returns:
        str: Output file path
    """
    os.makedirs(base_dir, exist_ok=True)
    
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, ext = os.path.splitext(filename)
        filename = f"{base_name}_{timestamp_str}{ext}"
    
    return os.path.join(base_dir, filename)

def save_config(config, filepath):
    """
    Save configuration to a JSON file.
    
    Args:
        config (dict): Configuration to save
        filepath (str): File path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def load_config(filepath, default=None):
    """
    Load configuration from a JSON file.
    
    Args:
        filepath (str): File path
        default (dict, optional): Default configuration if file doesn't exist
        
    Returns:
        dict: Loaded configuration
    """
    if not os.path.exists(filepath):
        return default if default is not None else {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config from {filepath}: {e}")
        return default if default is not None else {}

def visualize_mapped_data(frame, ocr_data, gaze_data=None, title=None):
    """
    Visualize OCR and gaze data on a frame.
    
    Args:
        frame (numpy.ndarray): Video frame
        ocr_data (pd.DataFrame): OCR data for the frame
        gaze_data (pd.DataFrame, optional): Gaze data for the frame
        title (str, optional): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(frame)
    
    # Draw OCR bounding boxes
    if ocr_data is not None and not ocr_data.empty:
        for _, row in ocr_data.iterrows():
            x, y = row['left'], row['top']
            w, h = row['width'], row['height']
            
            # Choose color based on category if available
            color = 'r'
            if 'category' in row and row['category'] == 'target':
                color = 'g'
                
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add text label
            ax.text(x, y-5, row['text'], color='yellow', fontsize=8, 
                   backgroundcolor='black', ha='left', va='bottom')
    
    # Draw gaze points
    if gaze_data is not None and not gaze_data.empty:
        ax.scatter(gaze_data['gaze_x'], gaze_data['gaze_y'], color='cyan', s=30, alpha=0.7)
    
    if title:
        ax.set_title(title)
    
    ax.axis('off')
    fig.tight_layout()
    
    return fig

def calculate_basic_statistics(data, group_by=None):
    """
    Calculate basic statistics for the data.
    
    Args:
        data (pd.DataFrame): Data to analyze
        group_by (str, optional): Column to group by
        
    Returns:
        pd.DataFrame: Statistics
    """
    if group_by and group_by in data.columns:
        grouped = data.groupby(group_by)
        stats = pd.DataFrame({
            'count': grouped.size(),
            'mean': grouped.mean(numeric_only=True).mean(axis=1) if not grouped.mean(numeric_only=True).empty else np.nan,
            'std': grouped.std(numeric_only=True).mean(axis=1) if not grouped.std(numeric_only=True).empty else np.nan
        })
        return stats
    else:
        # Calculate overall statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return pd.DataFrame({
                'count': len(data),
                'mean': data[numeric_cols].mean().mean(),
                'std': data[numeric_cols].std().mean()
            }, index=[0])
        else:
            return pd.DataFrame({
                'count': len(data),
                'mean': np.nan,
                'std': np.nan
            }, index=[0])

def merge_dataframes(dataframes, keys=None):
    """
    Merge multiple dataframes with keys.
    
    Args:
        dataframes (list): List of dataframes to merge
        keys (list, optional): List of keys for each dataframe
        
    Returns:
        pd.DataFrame: Merged dataframe
    """
    if not dataframes:
        return pd.DataFrame()
    
    if keys and len(keys) == len(dataframes):
        return pd.concat(dataframes, keys=keys, names=['source'])
    else:
        return pd.concat(dataframes, ignore_index=True)

def get_output_path(processed_root, participant, run_name=None):
    """Generate output path for a participant and optionally a specific run."""
    participant_output = os.path.join(processed_root, participant)
    
    if run_name:
        return os.path.join(participant_output, run_name)
    else:
        return participant_output

def csv_inspect(filepath, sample_rows=5):
    """
    Analyze a CSV file and return information about its format and contents.
    
    Args:
        filepath: Path to the CSV file
        sample_rows: Number of rows to sample
        
    Returns:
        Dictionary containing information about the CSV
    """
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}
    
    try:
        # Get file size
        file_size = os.path.getsize(filepath) / 1024  # Size in KB
        
        # Read header row
        with open(filepath, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
        
        # Try to read the file with pandas
        try:
            df = pd.read_csv(filepath, nrows=sample_rows)
            rows = df.to_dict('records')
            dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        except Exception as e:
            rows = []
            dtypes = {}
        
        # Count rows
        with open(filepath, 'r', encoding='utf-8') as f:
            row_count = sum(1 for _ in f) - 1  # Subtract header
        
        # Analyze format
        format_analysis = {
            "appears_to_be_eye_tracking": any("gaze" in col.lower() for col in header),
            "has_timestamp": any("time" in col.lower() for col in header),
            "has_coordinates": any(col.lower() in ("x", "y") for col in header) or 
                               any("point.x" in col.lower() or "point.y" in col.lower() for col in header),
        }
        
        # Return file analysis
        return {
            "filepath": filepath,
            "filesize_kb": file_size,
            "row_count": row_count,
            "column_count": len(header),
            "columns": header,
            "sample_rows": rows,
            "column_types": dtypes,
            "format_analysis": format_analysis
        }
        
    except Exception as e:
        return {"error": f"Error analyzing CSV: {str(e)}"}

def generate_sample_csv(output_path, format_type="standard"):
    """
    Generate a sample CSV file with eye tracking data in the specified format.
    
    Args:
        output_path: Path to save the CSV file
        format_type: Type of CSV format to generate (standard, simple, visionpro)
    
    Returns:
        Path to the generated file
    """
    # Generate some sample data
    n_rows = 1000
    timestamps = np.arange(0, n_rows * 10000, 10000)  # timestamps in microseconds
    
    # Generate random gaze points
    left_x = np.random.random(n_rows)
    left_y = np.random.random(n_rows)
    right_x = np.random.random(n_rows)
    right_y = np.random.random(n_rows)
    
    # Create sample data based on format
    if format_type == "standard":
        data = {
            'timestampUs': timestamps,
            'leftGaze.gazePointValid': np.ones(n_rows),
            'rightGaze.gazePointValid': np.ones(n_rows),
            'leftGaze.gazePoint.x': left_x,
            'leftGaze.gazePoint.y': left_y,
            'rightGaze.gazePoint.x': right_x,
            'rightGaze.gazePoint.y': right_y
        }
    elif format_type == "simple":
        data = {
            'timestamp': timestamps,
            'x': (left_x + right_x) / 2,
            'y': (left_y + right_y) / 2,
            'valid': np.ones(n_rows)
        }
    elif format_type == "visionpro":
        data = {
            'timestamp': timestamps,
            'leftX': left_x,
            'leftY': left_y,
            'rightX': right_x,
            'rightY': right_y,
            'leftValid': np.ones(n_rows),
            'rightValid': np.ones(n_rows)
        }
    else:
        raise ValueError(f"Unknown format type: {format_type}")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    return output_path

def create_resizable_sidebar(parent, min_width=350, weight=1):
    """
    Create a resizable and scrollable sidebar panel.
    
    Args:
        parent: The parent widget (usually a PanedWindow)
        min_width: Minimum width for the sidebar
        weight: Weight of the sidebar in the parent PanedWindow
    
    Returns:
        tuple: (container, content_frame) where:
            - container is the outer frame to add to PanedWindow
            - content_frame is the inner frame where content should be added
    """
    # Create container frame with minimum width
    container = ttk.Frame(parent, width=min_width)
    
    # Make sure it doesn't shrink below minimum width but can grow
    container.pack_propagate(False)
    container.grid_propagate(False)
    
    # Create outer frame for scrollbar
    scroll_frame = ttk.Frame(container)
    scroll_frame.pack(fill=tk.BOTH, expand=True)
    
    # Create canvas and scrollbar
    canvas = tk.Canvas(scroll_frame, highlightthickness=0)
    scrollbar = ttk.Scrollbar(scroll_frame, orient=tk.VERTICAL, command=canvas.yview)
    
    # Inner frame for controls (will be placed inside canvas)
    content_frame = ttk.Frame(canvas)
    
    # Configure canvas scrolling
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Pack scrollbar and canvas
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Create window in canvas for controls
    canvas_window = canvas.create_window((0, 0), window=content_frame, anchor=tk.NW)
    
    # Update scroll region when inner frame size changes
    def configure_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        # Make sure inner frame width matches canvas width
        canvas_width = event.width
        canvas.itemconfig(canvas_window, width=canvas_width)
    
    content_frame.bind("<Configure>", configure_scroll_region)
    
    # Update canvas width when container resizes
    def configure_canvas_width(event):
        canvas_width = event.width - scrollbar.winfo_width()
        canvas.itemconfig(canvas_window, width=canvas_width)
    
    container.bind("<Configure>", configure_canvas_width)
    
    # Add mouse wheel scrolling
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    # Bind mousewheel only when mouse is over the canvas
    canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
    canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
    
    return container, content_frame 