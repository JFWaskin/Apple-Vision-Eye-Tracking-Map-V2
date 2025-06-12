#!/usr/bin/env python3
"""
CSV Analyzer for Eye-Tracking Data
This utility helps diagnose issues with CSV files used in the eye-tracking system.
"""

import os
import sys
import argparse
import pandas as pd
import json
from utils import csv_inspect, generate_sample_csv

def analyze_csv(csv_path, sample_rows=5, verbose=True):
    """Analyze a CSV file and print information about it."""
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return False
        
    print(f"\n=== Analyzing CSV file: {csv_path} ===")
    
    # Basic file info
    file_size = os.path.getsize(csv_path) / 1024  # KB
    print(f"File size: {file_size:.2f} KB")
    
    # Get file info using csv_inspect
    info = csv_inspect(csv_path, sample_rows)
    
    if "error" in info:
        print(f"Error analyzing file: {info['error']}")
        return False
        
    # Print summary
    print(f"Rows: {info['row_count']}")
    print(f"Columns: {info['column_count']}")
    
    # Format analysis
    format_analysis = info['format_analysis']
    print("\nFormat analysis:")
    print(f"- Appears to be eye-tracking data: {format_analysis['appears_to_be_eye_tracking']}")
    print(f"- Has timestamp column: {format_analysis['has_timestamp']}")
    print(f"- Has coordinate columns: {format_analysis['has_coordinates']}")
    
    # Print columns
    print("\nColumns:")
    for i, col in enumerate(info['columns']):
        print(f"  {i+1}. {col}")
    
    # Print sample rows
    if verbose and info['sample_rows']:
        print("\nSample data:")
        for i, row in enumerate(info['sample_rows']):
            if i < sample_rows:
                print(f"  Row {i+1}: {row}")
    
    # Check for common eye-tracking columns
    standard_cols = [
        'timestampUs', 'timestamp', 'time',
        'leftGaze.gazePointValid', 'rightGaze.gazePointValid', 'leftValid', 'rightValid',
        'leftGaze.gazePoint.x', 'leftGaze.gazePoint.y', 
        'rightGaze.gazePoint.x', 'rightGaze.gazePoint.y',
        'x', 'y', 'leftX', 'leftY', 'rightX', 'rightY'
    ]
    
    print("\nCommon eye-tracking columns check:")
    found_cols = []
    for col in standard_cols:
        found = any(c.lower() == col.lower() or col.lower() in c.lower() for c in info['columns'])
        if found:
            found_cols.append(col)
            print(f"  ✓ Found column similar to '{col}'")
    
    if not found_cols:
        print("  ✗ No standard eye-tracking columns found")
    
    # Make recommendations
    print("\nRecommendations:")
    if not format_analysis['appears_to_be_eye_tracking'] and not format_analysis['has_coordinates']:
        print("  ⚠️ This doesn't appear to be eye-tracking data")
        print("  ⚠️ The CSV should contain gaze coordinates (x,y) or similar columns")
    
    if not format_analysis['has_timestamp']:
        print("  ⚠️ No timestamp column detected - this is required for processing")
    
    # Check format compatibility with simple_ocr_map.py
    standard_format = all(any(std_col.lower() in col.lower() for col in info['columns']) 
                         for std_col in ['timestampUs', 'leftGaze.gazePointValid', 'rightGaze.gazePointValid', 
                                         'leftGaze.gazePoint.x', 'leftGaze.gazePoint.y',
                                         'rightGaze.gazePoint.x', 'rightGaze.gazePoint.y'])
    
    simple_format = all(col in info['columns'] for col in ['x', 'y', 'timestamp'])
    
    visionpro_format = all(any(std_col.lower() in col.lower() for col in info['columns']) 
                          for std_col in ['leftX', 'leftY', 'rightX', 'rightY', 'timestamp'])
    
    print("\nFormat compatibility:")
    if standard_format:
        print("  ✓ CSV is compatible with standard format")
    elif simple_format:
        print("  ✓ CSV is compatible with simple (x,y) format")
    elif visionpro_format:
        print("  ✓ CSV is compatible with VisionPro format")
    else:
        print("  ✗ CSV format is not compatible with known eye-tracking formats")
        print("  ⚠️ Consider using the --convert option to convert to a compatible format")
    
    return True

def convert_csv(input_csv, output_csv, format_type="standard"):
    """Convert a CSV file to a standard eye-tracking format."""
    if not os.path.exists(input_csv):
        print(f"Error: File not found: {input_csv}")
        return False
        
    try:
        # Read the input CSV
        df = pd.read_csv(input_csv)
        print(f"Read {len(df)} rows from {input_csv}")
        
        # Check column names and try to map them
        cols = df.columns
        
        # Define possible mappings for each required column
        mappings = {
            'timestampUs': ['timestamp', 'time', 'timems', 'timeus', 'systemtimestamp'],
            'x': ['gazex', 'eyex', 'x_gaze', 'gaze_x', 'point_x'],
            'y': ['gazey', 'eyey', 'y_gaze', 'gaze_y', 'point_y'],
            'leftGaze.gazePoint.x': ['leftx', 'left_x', 'lefteye_x', 'left_gaze_x'],
            'leftGaze.gazePoint.y': ['lefty', 'left_y', 'lefteye_y', 'left_gaze_y'],
            'rightGaze.gazePoint.x': ['rightx', 'right_x', 'righteye_x', 'right_gaze_x'],
            'rightGaze.gazePoint.y': ['righty', 'right_y', 'righteye_y', 'right_gaze_y'],
            'leftGaze.gazePointValid': ['leftvalid', 'left_valid', 'validity_left'],
            'rightGaze.gazePointValid': ['rightvalid', 'right_valid', 'validity_right']
        }
        
        # Find matches for columns
        col_map = {}
        for target, alternatives in mappings.items():
            found = False
            for alt in alternatives:
                matches = [c for c in cols if alt.lower() == c.lower() or alt.lower() in c.lower()]
                if matches:
                    col_map[target] = matches[0]  # Use the first match
                    found = True
                    break
            
            if not found and target in df.columns:
                col_map[target] = target  # Use exact match if available
        
        # Print the mapping
        print("\nColumn mapping:")
        for target, source in col_map.items():
            print(f"  {source} -> {target}")
        
        # Handle different conversion types
        new_df = pd.DataFrame()
        
        if format_type == "standard":
            # Check for required columns
            required = ['timestampUs', 
                       'leftGaze.gazePoint.x', 'leftGaze.gazePoint.y', 
                       'rightGaze.gazePoint.x', 'rightGaze.gazePoint.y',
                       'leftGaze.gazePointValid', 'rightGaze.gazePointValid']
            
            # Create missing columns and use defaults
            for col in required:
                if col not in col_map:
                    if 'Valid' in col:
                        new_df[col] = 1  # Default validity
                    elif 'timestamp' in col.lower():
                        # Generate timestamps if missing
                        if 'timestampUs' not in col_map:
                            new_df['timestampUs'] = range(0, len(df) * 10000, 10000)
                    elif 'x' in col or 'y' in col:
                        # Use average for missing coordinates
                        if 'x' in col_map and 'y' in col_map:
                            if 'left' in col.lower() and 'x' in col:
                                new_df[col] = df[col_map['x']]
                            elif 'left' in col.lower() and 'y' in col:
                                new_df[col] = df[col_map['y']]
                            elif 'right' in col.lower() and 'x' in col:
                                new_df[col] = df[col_map['x']]
                            elif 'right' in col.lower() and 'y' in col:
                                new_df[col] = df[col_map['y']]
                else:
                    new_df[col] = df[col_map[col]]
        
        elif format_type == "simple":
            # Create a simple x,y,timestamp format
            new_df = pd.DataFrame()
            
            # Handle timestamp
            if 'timestampUs' in col_map:
                new_df['timestamp'] = df[col_map['timestampUs']]
            else:
                # Generate timestamps if missing
                new_df['timestamp'] = range(0, len(df) * 10000, 10000)
            
            # Handle x,y coordinates
            if 'x' in col_map and 'y' in col_map:
                new_df['x'] = df[col_map['x']]
                new_df['y'] = df[col_map['y']]
            elif 'leftGaze.gazePoint.x' in col_map and 'rightGaze.gazePoint.x' in col_map:
                new_df['x'] = (df[col_map['leftGaze.gazePoint.x']] + df[col_map['rightGaze.gazePoint.x']]) / 2
                new_df['y'] = (df[col_map['leftGaze.gazePoint.y']] + df[col_map['rightGaze.gazePoint.y']]) / 2
            
            # Add validity column
            new_df['valid'] = 1
        
        # Save the converted file
        new_df.to_csv(output_csv, index=False)
        print(f"\nSaved converted file with {len(new_df)} rows to {output_csv}")
        return True
    
    except Exception as e:
        print(f"Error converting CSV: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_example(output_path, format_type="standard"):
    """Generate an example CSV file in the specified format."""
    try:
        path = generate_sample_csv(output_path, format_type)
        print(f"Generated sample CSV file in {format_type} format: {path}")
        return True
    except Exception as e:
        print(f"Error generating sample CSV: {str(e)}")
        return False

def find_eyetracking_files(directory, verbose=True):
    """
    Scan a directory to find CSV files that appear to contain eye-tracking data.
    
    Args:
        directory: Directory to scan
        verbose: Whether to print detailed information
        
    Returns:
        List of dictionaries with information about potential eye-tracking files
    """
    import os
    import glob
    
    if verbose:
        print(f"\n=== Scanning for eye-tracking files in: {directory} ===")
    
    # Get all CSV files
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if verbose:
        print(f"Found {len(csv_files)} CSV files")
    
    # Check each file for eye-tracking data
    results = []
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        
        # Look for patterns in filename - raw data files often have "_raw" in the name
        is_raw = "_raw" in filename
        is_index = "_index" in filename
        
        # Basic inspection to determine if it's eye-tracking data
        info = csv_inspect(csv_path, sample_rows=1)
        
        if "error" in info:
            continue
            
        # Check if it looks like eye-tracking data
        format_analysis = info['format_analysis']
        
        # Create a score based on likelihood
        score = 0
        if is_raw:
            score += 5  # _raw files are very likely to be the ones we want
        if format_analysis['appears_to_be_eye_tracking']:
            score += 3
        if format_analysis['has_coordinates']:
            score += 3
        if format_analysis['has_timestamp']:
            score += 1
        
        # Penalize index files
        if is_index:
            score -= 5
        
        # Add to results if likely eye-tracking data
        if score > 0:
            file_info = {
                'path': csv_path,
                'filename': filename,
                'score': score,
                'columns': len(info['columns']),
                'rows': info['row_count'],
                'is_raw': is_raw,
                'has_coordinates': format_analysis['has_coordinates'],
                'has_timestamp': format_analysis['has_timestamp']
            }
            results.append(file_info)
    
    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    if verbose:
        if results:
            print("\nPotential eye-tracking data files (in order of likelihood):")
            for i, file_info in enumerate(results):
                print(f"{i+1}. {file_info['filename']} (score: {file_info['score']}, columns: {file_info['columns']})")
        else:
            print("No likely eye-tracking data files found")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="CSV Analyzer for Eye-Tracking Data")
    parser.add_argument("file", nargs="?", help="CSV file to analyze")
    parser.add_argument("--sample", "-s", type=int, default=5, help="Number of sample rows to show")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show more detailed output")
    parser.add_argument("--convert", "-c", help="Convert to a new CSV file with the specified name")
    parser.add_argument("--format", "-f", choices=["standard", "simple", "visionpro"], default="standard", 
                       help="Format to convert to (standard, simple, visionpro)")
    parser.add_argument("--generate", "-g", help="Generate an example CSV file with the specified name")
    parser.add_argument("--scan", help="Scan a directory for eye-tracking data files")
    
    args = parser.parse_args()
    
    if args.generate:
        return generate_example(args.generate, args.format)
    
    if args.scan:
        find_eyetracking_files(args.scan, verbose=True)
        return True
    
    if not args.file:
        parser.print_help()
        return False
    
    success = analyze_csv(args.file, args.sample, args.verbose)
    
    if success and args.convert:
        convert_csv(args.file, args.convert, args.format)
    
    return success

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 