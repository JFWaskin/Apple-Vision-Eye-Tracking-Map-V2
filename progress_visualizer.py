#!/usr/bin/env python3
"""
Progress Visualizer module for the eye-tracking analysis system.
This module provides visualization of translation progression over time periods.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import patches
import re
import math
import sys
import time
from datetime import datetime
import gc

# Import utility functions
from utils import create_resizable_sidebar

# Configure matplotlib for Chinese characters
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# Constants
SEPARATOR_OFFSET = 0.5  # Vertical offset threshold to separate source/target text

class ProgressVisualizer:
    """Visualizes eye-tracking progression data for translation analysis."""
    
    def __init__(self, processed_dir):
        """Initialize the progress visualizer."""
        self.processed_dir = processed_dir
        self.participants = {}  # Dictionary of participant -> list of run directories
        self.current_participant = None
        self.current_run = None
        self.current_period = None
        self.period_data = None
        
        # Visualization parameters
        self.source_text_order = []  # List of source text segments in display order
        self.target_text_order = []  # List of target text segments in display order
        self.source_colors = plt.cm.Blues  # Default colormap for source
        self.target_colors = plt.cm.Greens  # Default colormap for target
        self.orientation_color = 'yellow'  # Color for orientation phase
        self.revision_color = 'red'  # Color for revision phase
        
        # Loaded data
        self.gaze_data = None
        self.text_segments = None
        
    def create_ui(self, parent, project_manager):
        """Create the user interface for the progress visualizer."""
        self.parent = parent
        self.project_manager = project_manager
        
        # Create main frame with padding
        main_frame = ttk.Frame(parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create resizable paned window with sash that can be moved by user
        self.paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Create resizable sidebar using utility function
        left_container, left_panel = create_resizable_sidebar(self.paned_window)
        
        # Create right panel for visualization
        right_panel = ttk.Frame(self.paned_window)
        
        # Add panels to paned window with proper weights
        self.paned_window.add(left_container, weight=1)
        self.paned_window.add(right_panel, weight=3)
        
        # Configure left panel contents
        self.configure_left_panel(left_panel)
        
        # Configure right panel contents
        self.configure_right_panel(right_panel)
        
        # Initialize participant data
        self.refresh_participants()
        
    def configure_left_panel(self, parent):
        """Configure the left control panel."""
        # Participant selection
        participant_frame = ttk.LabelFrame(parent, text="Participant", padding=5)
        participant_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.participant_var = tk.StringVar()
        self.participant_dropdown = ttk.Combobox(participant_frame, textvariable=self.participant_var, state="readonly")
        self.participant_dropdown.pack(fill=tk.X, pady=5)
        self.participant_dropdown.bind("<<ComboboxSelected>>", self.on_participant_selected)
        
        ttk.Button(participant_frame, text="Refresh", command=self.refresh_participants).pack(fill=tk.X)
        
        # Run selection
        run_frame = ttk.LabelFrame(parent, text="Processing Run", padding=5)
        run_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.run_var = tk.StringVar()
        self.run_dropdown = ttk.Combobox(run_frame, textvariable=self.run_var, state="readonly")
        self.run_dropdown.pack(fill=tk.X, pady=5)
        self.run_dropdown.bind("<<ComboboxSelected>>", self.on_run_selected)
        
        # Period selection
        period_frame = ttk.LabelFrame(parent, text="Time Period", padding=5)
        period_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.period_var = tk.StringVar()
        self.period_dropdown = ttk.Combobox(period_frame, textvariable=self.period_var, state="readonly")
        self.period_dropdown.pack(fill=tk.X, pady=5)
        self.period_dropdown.bind("<<ComboboxSelected>>", self.on_period_selected)
        
        # Filtering options
        filter_frame = ttk.LabelFrame(parent, text="Filtering", padding=5)
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Source text filtering
        source_filter_frame = ttk.Frame(filter_frame)
        source_filter_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(source_filter_frame, text="Source Min Count:").pack(side=tk.LEFT, padx=(0, 5))
        self.source_threshold_var = tk.IntVar(value=0)
        source_threshold = ttk.Spinbox(source_filter_frame, from_=0, to=1000, 
                                    textvariable=self.source_threshold_var, width=5)
        source_threshold.pack(side=tk.RIGHT)
        
        # Target text filtering
        target_filter_frame = ttk.Frame(filter_frame)
        target_filter_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(target_filter_frame, text="Target Min Count:").pack(side=tk.LEFT, padx=(0, 5))
        self.target_threshold_var = tk.IntVar(value=0)
        target_threshold = ttk.Spinbox(target_filter_frame, from_=0, to=1000, 
                                    textvariable=self.target_threshold_var, width=5)
        target_threshold.pack(side=tk.RIGHT)
        
        # Downsampling option for performance
        downsample_frame = ttk.Frame(filter_frame)
        downsample_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(downsample_frame, text="Max Points (thousands):").pack(side=tk.LEFT, padx=(0, 5))
        self.max_points_var = tk.IntVar(value=20)  # Default to 20,000 points total
        max_points = ttk.Spinbox(downsample_frame, from_=1, to=200, 
                               textvariable=self.max_points_var, width=5)
        max_points.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Apply filter button
        ttk.Button(filter_frame, text="Apply Filters", command=self.apply_filters).pack(fill=tk.X, pady=(5, 0))
        
        # Visualization simplification controls
        simplify_frame = ttk.LabelFrame(parent, text="Visualization Simplification", padding=5)
        simplify_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Display mode selection
        display_mode_frame = ttk.Frame(simplify_frame)
        display_mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(display_mode_frame, text="Display Mode:").pack(side=tk.LEFT, padx=(0, 5))
        self.display_mode_var = tk.StringVar(value="Sampling")
        display_mode = ttk.Combobox(display_mode_frame, textvariable=self.display_mode_var, state="readonly",
                                   values=["All Points", "Sampling", "Clustering", "Summary"])
        display_mode.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Maximum points to display
        max_points_frame = ttk.Frame(simplify_frame)
        max_points_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(max_points_frame, text="Max Points:").pack(side=tk.LEFT, padx=(0, 5))
        self.max_points_var = tk.IntVar(value=500)  # Default to 500 total points
        max_points = ttk.Scale(max_points_frame, from_=100, to=5000, 
                              variable=self.max_points_var, orient=tk.HORIZONTAL)
        max_points.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Max points label
        self.max_points_label = ttk.Label(max_points_frame, text="500")
        self.max_points_label.pack(side=tk.RIGHT, padx=(0, 5))
        
        # Update max points label when scale changes
        def update_max_points_label(*args):
            self.max_points_label.config(text=str(self.max_points_var.get()))
        
        self.max_points_var.trace_add("write", update_max_points_label)
        
        # Focus on frequent items
        focus_frame = ttk.Frame(simplify_frame)
        focus_frame.pack(fill=tk.X, pady=5)
        
        self.prioritize_frequent_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(focus_frame, text="Prioritize Frequent Items", 
                       variable=self.prioritize_frequent_var).pack(side=tk.LEFT)
        
        # Time range selection
        time_range_frame = ttk.Frame(simplify_frame)
        time_range_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(time_range_frame, text="Time Range:").pack(side=tk.LEFT, padx=(0, 5))
        
        time_range_inner = ttk.Frame(time_range_frame)
        time_range_inner.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        self.time_range_start_var = tk.IntVar(value=0)  # Percentage of total time
        self.time_range_end_var = tk.IntVar(value=100)  # Percentage of total time
        
        self.time_range_start = ttk.Scale(time_range_inner, from_=0, to=100, 
                                        variable=self.time_range_start_var, orient=tk.HORIZONTAL)
        self.time_range_start.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.time_range_end = ttk.Scale(time_range_inner, from_=0, to=100, 
                                      variable=self.time_range_end_var, orient=tk.HORIZONTAL)
        self.time_range_end.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Apply simplification button
        ttk.Button(simplify_frame, text="Apply Simplification", command=self.apply_simplification).pack(fill=tk.X, pady=(5, 0))
        
        # Text ordering
        order_frame = ttk.LabelFrame(parent, text="Text Ordering", padding=5)
        order_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(order_frame, text="Source Text:").pack(anchor=tk.W, pady=(5, 0))
        
        source_frame = ttk.Frame(order_frame)
        source_frame.pack(fill=tk.X, pady=5)
        
        self.source_listbox = tk.Listbox(source_frame, height=6)
        source_scrollbar = ttk.Scrollbar(source_frame, orient=tk.VERTICAL, command=self.source_listbox.yview)
        self.source_listbox.configure(yscrollcommand=source_scrollbar.set)
        self.source_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        source_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        source_buttons = ttk.Frame(order_frame)
        source_buttons.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(source_buttons, text="Move Up", width=8, 
                  command=lambda: self.move_text_item(self.source_listbox, -1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(source_buttons, text="Move Down", width=8, 
                  command=lambda: self.move_text_item(self.source_listbox, 1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(source_buttons, text="Remove", width=8, 
                  command=lambda: self.remove_text_item(self.source_listbox)).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(order_frame, text="Target Text:").pack(anchor=tk.W, pady=(5, 0))
        
        target_frame = ttk.Frame(order_frame)
        target_frame.pack(fill=tk.X, pady=5)
        
        self.target_listbox = tk.Listbox(target_frame, height=6)
        target_scrollbar = ttk.Scrollbar(target_frame, orient=tk.VERTICAL, command=self.target_listbox.yview)
        self.target_listbox.configure(yscrollcommand=target_scrollbar.set)
        self.target_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        target_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        target_buttons = ttk.Frame(order_frame)
        target_buttons.pack(fill=tk.X)
        
        ttk.Button(target_buttons, text="Move Up", width=8, 
                  command=lambda: self.move_text_item(self.target_listbox, -1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(target_buttons, text="Move Down", width=8, 
                  command=lambda: self.move_text_item(self.target_listbox, 1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(target_buttons, text="Remove", width=8, 
                  command=lambda: self.remove_text_item(self.target_listbox)).pack(side=tk.LEFT, padx=2)
        
        # Auto-reorder buttons
        reorder_frame = ttk.Frame(order_frame)
        reorder_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(reorder_frame, text="Auto-Reorder Source (ST)", width=20, 
                  command=lambda: self.show_reorder_dialog(is_source=True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(reorder_frame, text="Auto-Reorder Target (TT)", width=20, 
                  command=lambda: self.show_reorder_dialog(is_source=False)).pack(side=tk.LEFT, padx=2)
        
        # Visualization controls
        viz_frame = ttk.LabelFrame(parent, text="Visualization Options", padding=5)
        viz_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Show orientation phase
        orientation_frame = ttk.Frame(viz_frame)
        orientation_frame.pack(fill=tk.X, pady=2)
        
        self.show_orientation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(orientation_frame, text="Show Orientation Phase", 
                       variable=self.show_orientation_var).pack(side=tk.LEFT)
        
        # Show revision phase
        revision_frame = ttk.Frame(viz_frame)
        revision_frame.pack(fill=tk.X, pady=2)
        
        self.show_revision_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(revision_frame, text="Show Revision Phase", 
                       variable=self.show_revision_var).pack(side=tk.LEFT)
        
        # Show fixation units
        fixation_frame = ttk.Frame(viz_frame)
        fixation_frame.pack(fill=tk.X, pady=2)
        
        self.show_fixation_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fixation_frame, text="Show Fixation Units", 
                       variable=self.show_fixation_var).pack(side=tk.LEFT)
        
        # Show connecting lines - Set to TRUE by default
        lines_frame = ttk.Frame(viz_frame)
        lines_frame.pack(fill=tk.X, pady=2)
        
        self.show_lines_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(lines_frame, text="Connect Points with Lines", 
                       variable=self.show_lines_var).pack(side=tk.LEFT)
        
        # Line density control
        line_density_frame = ttk.Frame(viz_frame)
        line_density_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(line_density_frame, text="Line Stride:").pack(side=tk.LEFT, padx=(0, 5))
        self.line_stride_var = tk.IntVar(value=5)  # Connect every 5th point
        line_stride = ttk.Spinbox(line_density_frame, from_=1, to=50, 
                                textvariable=self.line_stride_var, width=5)
        line_stride.pack(side=tk.RIGHT)
        
        # Visualization button
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.visualize_button = ttk.Button(button_frame, text="Generate Visualization", 
                                         command=self.generate_visualization)
        self.visualize_button.pack(fill=tk.X, pady=5)
        
        # Export button
        self.export_button = ttk.Button(button_frame, text="Export Visualization", 
                                      command=self.export_visualization, state=tk.DISABLED)
        self.export_button.pack(fill=tk.X)
        
    def configure_right_panel(self, parent):
        """Configure the right visualization panel."""
        # Create a frame for the visualization
        self.viz_frame = ttk.Frame(parent)
        self.viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a progress bar at the top (invisible until needed)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_frame = ttk.Frame(self.viz_frame)
        self.progress_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_label = ttk.Label(self.progress_frame, text="Processing...")
        self.progress_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, 
                                          length=200, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Hide progress bar initially
        self.progress_frame.pack_forget()
        
        # Add status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.viz_frame, textvariable=self.status_var)
        self.status_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Create frame for matplotlib canvas
        self.canvas_frame = ttk.Frame(self.viz_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        
        # Create matplotlib canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        self.toolbar.update()
        
        # Create a frame for legend and info
        self.info_frame = ttk.Frame(self.viz_frame)
        self.info_frame.pack(fill=tk.X, pady=5)
        
        # Legend and information
        ttk.Label(self.info_frame, text="Source: ", foreground='blue').pack(side=tk.LEFT)
        ttk.Label(self.info_frame, text="Target: ", foreground='green').pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(self.info_frame, text="Time →", foreground='black').pack(side=tk.RIGHT)
        
        # Add tooltip support - using canvas_frame as parent instead of canvas
        self.tooltip_text = tk.StringVar()
        self.tooltip = ttk.Label(self.canvas_frame, textvariable=self.tooltip_text, 
                              background='#FFFFCC', relief='solid', borderwidth=1)
        
        # Bind canvas events to the canvas widget
        self.canvas.get_tk_widget().bind("<Motion>", self.on_canvas_motion)
        self.canvas.get_tk_widget().bind("<Configure>", self.on_canvas_configure)
        
        # Dictionary to store plot elements and their info
        self.plot_elements = {}  # id -> (type, text, time, etc.)
    
    def on_canvas_motion(self, event):
        """Handle mouse motion on the canvas."""
        try:
            # Convert event coordinates to figure coordinates
            if not hasattr(self, 'figure') or not self.figure:
                return
                
            # Get the axes and figure
            ax = self.figure.gca()
            
            # Convert display coordinates to data coordinates
            inv = ax.transData.inverted()
            data_coords = inv.transform((event.x, event.y))
            x, y = data_coords
            
            # Find the closest data point
            if hasattr(self, 'period_data') and self.period_data is not None:
                # Find points within a small radius
                radius = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02  # 2% of x-axis range
                nearby_points = self.period_data[
                    (abs(self.period_data['timestamp'] - x) < radius)
                ]
                
                if not nearby_points.empty:
                    # Get the closest point
                    closest_point = nearby_points.iloc[
                        (nearby_points['timestamp'] - x).abs().idxmin()
                    ]
                    
                    # Show tooltip with point information
                    tooltip_text = (
                        f"Text: {closest_point['text']}\n"
                        f"Time: {closest_point['timestamp']:.0f}ms\n"
                        f"{'Source' if closest_point['is_source'] else 'Target'}"
                    )
                    
                    self.tooltip_text.set(tooltip_text)
                    self.tooltip.place(x=event.x + 15, y=event.y + 10)
                    return
        
            # Hide tooltip if no point is found
            self.tooltip.place_forget()
            
        except Exception as e:
            print(f"Error in on_canvas_motion: {e}")
        self.tooltip.place_forget()

    def on_canvas_configure(self, event):
        """Handle canvas resize event."""
        # For matplotlib, we don't need to manually handle canvas configuration
        pass

    def generate_visualization(self, use_simplified=False):
        """Generate the visualization of translation progression."""
        # Choose which dataset to use
        visualization_data = self.simplified_data if use_simplified and hasattr(self, 'simplified_data') else self.period_data
        
        if visualization_data is None:
            messagebox.showerror("Error", "No data loaded. Please select a time period first.")
            return
        
        try:
            print("\nDEBUG: Starting visualization generation")
            print(f"DEBUG: Using {'simplified' if use_simplified else 'full'} dataset with {len(visualization_data)} points")
            
            # Clear previous figure
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            print("DEBUG: Created figure and axes")
            
            # Get time range
            min_time = visualization_data['timestamp'].min()
            max_time = visualization_data['timestamp'].max()
            print(f"DEBUG: Time range: {min_time:.2f} to {max_time:.2f}")
            
            # Create dual axes for source and target
            ax_source = ax
            ax_target = ax.twinx()
            print("DEBUG: Created dual axes")
            
            # Plot source text fixations (blue circles)
            source_data = visualization_data[visualization_data['is_source'] == True]
            print(f"DEBUG: Processing {len(source_data)} source points")
            if not source_data.empty:
                ax_source.scatter(source_data['timestamp'], 
                               source_data['text_order'],
                               c='blue',
                               marker='o',
                               s=source_data['duration'] * 50,  # Increased size scaling
                               alpha=0.6,
                               label='Source Fixations')
            
            # Plot target text fixations (green diamonds)
            target_data = visualization_data[visualization_data['is_source'] == False]
            print(f"DEBUG: Processing {len(target_data)} target points")
            if not target_data.empty:
                ax_target.scatter(target_data['timestamp'],
                               target_data['text_order'],
                               c='green',
                               marker='D',
                               s=target_data['duration'] * 50,  # Increased size scaling
                               alpha=0.6,
                               label='Target Fixations')
            
            # Add orientation phase (yellow oval)
            orientation_data = visualization_data[visualization_data['phase'] == 'orientation']
            print(f"DEBUG: Found {len(orientation_data)} points in orientation phase")
            if not orientation_data.empty:
                orientation_start = orientation_data['timestamp'].min()
                orientation_end = orientation_data['timestamp'].max()
                orientation_y = orientation_data['text_order'].mean()
                orientation_height = orientation_data['text_order'].max() - orientation_data['text_order'].min()
                
                ax.add_patch(patches.Ellipse((orientation_start, orientation_y),
                                           width=orientation_end - orientation_start,
                                           height=max(1.0, orientation_height),  # Ensure minimum height
                                           facecolor='yellow',
                                           alpha=0.3,
                                           label='Orientation Phase'))
            
            # Add revision phase (red rectangle)
            revision_data = visualization_data[visualization_data['phase'] == 'revision']
            print(f"DEBUG: Found {len(revision_data)} points in revision phase")
            if not revision_data.empty:
                revision_start = revision_data['timestamp'].min()
                revision_end = revision_data['timestamp'].max()
                revision_y = revision_data['text_order'].mean()
                revision_height = max(1.0, revision_data['text_order'].max() - revision_data['text_order'].min())
                
                ax.add_patch(patches.Rectangle((revision_start, revision_y - revision_height/2),
                                             width=revision_end - revision_start,
                                             height=revision_height,
                                             facecolor='red',
                                             alpha=0.3,
                                             label='Revision Phase'))
            
            # Set axis labels and properties
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Source Text Order')
            ax_target.set_ylabel('Target Text Order')
            
            # Set time range with padding
            time_padding = (max_time - min_time) * 0.05  # 5% padding
            ax.set_xlim(min_time - time_padding, max_time + time_padding)
            
            # Set y-axis limits with padding
            source_padding = 0.5
            target_padding = 0.5
            ax_source.set_ylim(-source_padding, len(self.source_text_order) - 1 + source_padding)
            ax_target.set_ylim(-target_padding, len(self.target_text_order) - 1 + target_padding)
            
            # Add text labels for source and target
            for i, text in enumerate(self.source_text_order):
                ax_source.text(min_time - time_padding, i, text, 
                             horizontalalignment='right', verticalalignment='center')
            
            for i, text in enumerate(self.target_text_order):
                ax_target.text(max_time + time_padding, i, text, 
                             horizontalalignment='left', verticalalignment='center')
            
            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_target.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # Adjust layout to prevent text cutoff
            self.figure.tight_layout()
            
            # Update the canvas
            self.canvas.draw()
            
            print("DEBUG: Visualization complete")
            
        except Exception as e:
            print(f"\nDEBUG: Error in visualization: {str(e)}")
            print("DEBUG: Stack trace:")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Error generating visualization: {str(e)}")
            
        # Enable export button
        self.export_button.config(state=tk.NORMAL)
    
    def export_visualization(self):
        """Export the visualization as an image."""
        if not hasattr(self, 'canvas') or not self.canvas:
            messagebox.showerror("Error", "No visualization available to export.")
            return
        
        try:
            # Ask for save location
            file_path = filedialog.asksaveasfilename(
                title="Export Visualization",
                filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")]
            )
            
            if not file_path:
                return  # User cancelled
                
            # Add .png extension if not provided
            if not file_path.lower().endswith('.png'):
                file_path += '.png'
            
            # Get canvas bbox
            bbox = self.canvas.bbox("all")
            if not bbox:
                messagebox.showerror("Error", "No content to export.")
                return
                
            # Create image from canvas
            x0, y0, x1, y1 = bbox
            width = x1 - x0
            height = y1 - y0
            
            # Use Pillow to create image
            from PIL import Image, ImageDraw
            
            # Create a new image with white background
            image = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(image)
            
            # Function to map canvas coords to image coords
            def map_coords(x, y):
                return x - x0, y - y0
            
            # Draw all canvas items to the image
            for item_id in self.canvas.find_all():
                item_type = self.canvas.type(item_id)
                coords = self.canvas.coords(item_id)
                options = {opt: self.canvas.itemcget(item_id, opt) for opt in 
                          ['fill', 'outline', 'width', 'dash']}
                
                if item_type == 'line':
                    # Draw line
                    points = [map_coords(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                    if len(points) >= 2:
                        draw.line(points, fill=options.get('fill', 'black'), 
                                width=int(options.get('width', '1')))
                
                elif item_type == 'oval':
                    # Draw oval/circle
                    x1, y1, x2, y2 = coords
                    mapped_coords = map_coords(x1, y1) + map_coords(x2, y2)
                    draw.ellipse(mapped_coords, fill=options.get('fill', ''), 
                               outline=options.get('outline', 'black'))
                
                elif item_type == 'text':
                    # Skip text - we'll add it separately in a real implementation
                    pass
            
            # Save the image
            image.save(file_path)
            
            messagebox.showinfo("Export Complete", f"Visualization exported to {file_path}")
            
        except Exception as e:
            print(f"Export error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Error exporting visualization: {str(e)}")
    
    def show_reorder_dialog(self, is_source=True):
        """Show dialog to paste text for auto-reordering."""
        dialog = tk.Toplevel(self.parent)
        dialog.title("Auto-Reorder Text")
        dialog.geometry("500x400")
        dialog.transient(self.parent)
        dialog.grab_set()
        
        ttk.Label(dialog, text=f"Paste {'source' if is_source else 'target'} text in the desired order:").pack(pady=(10, 5))
        ttk.Label(dialog, text="You can paste individual lines or a complete paragraph.").pack(pady=(0, 5))
        ttk.Label(dialog, text="Words will be matched to the text segments, and ordered from bottom to top.").pack(pady=(0, 10))
        
        # Text area for pasting
        text_frame = ttk.Frame(dialog)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        text_widget = tk.Text(text_frame)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Cancel", 
                  command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Apply Reordering", 
                  command=lambda: self.apply_reordering(text_widget.get("1.0", tk.END), is_source, dialog)).pack(side=tk.RIGHT)
    
    def apply_reordering(self, text_content, is_source, dialog):
        """Apply the reordering based on pasted text."""
        # Close the dialog
        dialog.destroy()
        
        # Get current items from the appropriate listbox
        listbox = self.source_listbox if is_source else self.target_listbox
        current_items = {}
        
        for i in range(listbox.size()):
            item_text = listbox.get(i)
            # Extract the text part without the count
            text = item_text.split(" (")[0]
            count = item_text.split("(")[1].split(")")[0] if "(" in item_text else "0"
            current_items[text] = count
        
        # Apply the robust reordering algorithm
        self._apply_robust_reordering(text_content, is_source, current_items, listbox)
            
    def _apply_robust_reordering(self, text_content, is_source, current_items, listbox):
        """A robust reordering algorithm that works for both Chinese and English texts."""
        import re
        
        # Identify if the content is primarily Chinese
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', text_content))
        print(f"Text identified as {'Chinese' if is_chinese else 'non-Chinese (likely English)'}")
        
        # Store original text segments and their counts
        segments = list(current_items.keys())
        segment_counts = {text: current_items[text] for text in segments}
        
        # Step 1: Normalize the input text and segments based on language type
        if is_chinese:
            # For Chinese, remove all spaces and punctuation
            norm_input = re.sub(r'[^\u4e00-\u9fff]', '', text_content)
            norm_segments = {text: re.sub(r'[^\u4e00-\u9fff]', '', text) for text in segments}
        else:
            # For English, convert to lowercase and keep only alphanumeric chars
            norm_input = re.sub(r'[^a-zA-Z0-9\s]', ' ', text_content.lower())
            norm_input = re.sub(r'\s+', ' ', norm_input).strip()
            norm_segments = {text: re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower()) for text in segments}
            norm_segments = {text: re.sub(r'\s+', ' ', norm_segments[text]).strip() for text in segments}
        
        # Step 2: Calculate position scores for each segment
        # We'll find the earliest occurrence of each segment in the input text
        position_scores = {}
        
        for text in segments:
            norm_text = norm_segments[text]
            if not norm_text:  # Skip empty segments
                position_scores[text] = float('inf')
                continue
            
            # Find position of this segment in the normalized input
            if is_chinese:
                # For Chinese, we'll look for the longest matching substring
                best_pos = float('inf')
                best_length = 0
                
                # Try to find the longest matching substring
                for i in range(len(norm_text)):
                    for j in range(i+1, len(norm_text)+1):
                        substr = norm_text[i:j]
                        if len(substr) > 1:  # Only consider substrings of length > 1
                            pos = norm_input.find(substr)
                            if pos != -1 and (len(substr) > best_length or 
                                             (len(substr) == best_length and pos < best_pos)):
                                best_pos = pos
                                best_length = len(substr)
                
                if best_length > 0:
                    position_scores[text] = best_pos
                else:
                    # If no substring found, use the position of the first character
                    first_char = norm_text[0] if norm_text else ''
                    pos = norm_input.find(first_char)
                    position_scores[text] = pos if pos != -1 else float('inf')
            else:
                # For English, we'll look for the highest concentration of matching words
                words = norm_text.split()
                if not words:
                    position_scores[text] = float('inf')
                    continue
                
                input_words = norm_input.split()
                positions = []
                
                for word in words:
                    if len(word) > 2:  # Only consider words with more than 2 characters
                        for i, input_word in enumerate(input_words):
                            if word == input_word:
                                positions.append(i)
                                break
                
                if positions:
                    # Use the average position of found words
                    position_scores[text] = sum(positions) / len(positions)
                else:
                    position_scores[text] = float('inf')
        
        # Step 3: Order segments by their position in the input text
        # Sort segments by position score (lower score = earlier in text)
        ordered_segments = sorted(segments, key=lambda x: 
                                 float('inf') if position_scores[x] == float('inf') 
                                 else position_scores[x])
        
        # Report stats on matching
        matched_count = sum(1 for text in segments if position_scores[text] != float('inf'))
        print(f"Matched {matched_count} out of {len(segments)} segments")
        
        # Step 4: Prepare the final ordered list, ensuring proper display order
        # The visualization expects items to be ordered from top to bottom in the listbox
        # but displays them from bottom to top
        
        # We'll reverse the list to ensure bottom-to-top display order
        ordered_segments.reverse()
        
        # Update the listbox with the new order
        listbox.delete(0, tk.END)
        for text in ordered_segments:
            count = segment_counts[text]
            listbox.insert(tk.END, f"{text} ({count})")
        
        # Update the text order
        if is_source:
            self.source_text_order = ordered_segments
        else:
            self.target_text_order = ordered_segments
        
        # Show success message
        messagebox.showinfo("Reordering Complete", 
                           f"Successfully reordered the text segments.\n"
                           f"Matched {matched_count} out of {len(segments)} segments.")

    # Remove the old reordering methods as they're no longer needed
    def _apply_chinese_reordering(self, text_content, is_source, current_items, listbox):
        """THIS METHOD IS DEPRECATED - Using _apply_robust_reordering instead"""
        self._apply_robust_reordering(text_content, is_source, current_items, listbox)
    
    def _apply_english_reordering(self, text_content, is_source, current_items, listbox):
        """THIS METHOD IS DEPRECATED - Using _apply_robust_reordering instead"""
        self._apply_robust_reordering(text_content, is_source, current_items, listbox)
    
    def refresh_participants(self):
        """Refresh the participant list."""
        # Get participants from processed output dir
        participants = []
        for item in os.listdir(self.processed_dir):
            if os.path.isdir(os.path.join(self.processed_dir, item)) and not item.startswith('.'):
                participants.append(item)
        
        # Update participants dictionary
        for participant in participants:
            self.participants[participant] = []
            participant_dir = os.path.join(self.processed_dir, participant)
            for item in os.listdir(participant_dir):
                if os.path.isdir(os.path.join(participant_dir, item)) and item.startswith('run_'):
                    self.participants[participant].append(item)
            
            # Sort runs by timestamp (newest first)
            self.participants[participant].sort(reverse=True)
        
        # Update dropdown
        self.participant_dropdown['values'] = participants
        
        # Reset if current selection is not valid
        if self.current_participant not in participants:
            if participants:
                self.participant_var.set(participants[0])
                self.on_participant_selected()
            else:
                self.participant_var.set("")
                self.run_var.set("")
                self.period_var.set("")
                self.run_dropdown['values'] = []
                self.period_dropdown['values'] = []
    
    def on_participant_selected(self, event=None):
        """Handle participant selection."""
        participant = self.participant_var.get()
        if not participant:
            return
        
        self.current_participant = participant
        
        # Update run dropdown
        runs = self.participants.get(participant, [])
        self.run_dropdown['values'] = runs
        
        if runs:
            self.run_var.set(runs[0])
            self.on_run_selected()
        else:
            self.run_var.set("")
            self.period_var.set("")
            self.period_dropdown['values'] = []
    
    def on_run_selected(self, event=None):
        """Handle run selection event."""
        if self.run_var.get():
            self.current_run = self.run_var.get()
            
            # Load periods for this run
            run_dir = os.path.join(self.processed_dir, self.current_participant, self.current_run)
            periods = []
            
            # Look specifically for ocr_results_period_N.csv files instead of period_N_start_to_end.csv
            if os.path.exists(run_dir):
                for item in os.listdir(run_dir):
                    if os.path.isfile(os.path.join(run_dir, item)) and item.startswith('ocr_results_period_') and item.endswith('.csv'):
                        # Extract just the period name without .csv extension
                        period_name = item.replace('.csv', '')
                        periods.append(period_name)
            
            # Sort periods numerically by period number
            periods.sort(key=lambda x: int(x.split('_')[2]) if len(x.split('_')) > 2 and x.split('_')[2].isdigit() else 0)
            
            # Update period dropdown
            self.period_dropdown['values'] = periods
            
            # Reset period selection
            if periods:
                self.period_var.set(periods[0])
                self.on_period_selected()
            else:
                self.period_var.set("")
                self.period_data = None
                
    def on_period_selected(self, event=None):
        """Handle period selection event."""
        if self.period_var.get():
            self.current_period = self.period_var.get()
            self.load_period_data()
    
    def load_period_data(self):
        """Load data for the selected time period."""
        if not self.current_period:
            return
            
        try:
            # Load directly from the period file
            period_file = os.path.join(self.processed_dir, self.current_participant, 
                                     self.current_run, f"{self.current_period}.csv")
            
            print(f"\nDEBUG: Loading period file: {period_file}")
            
            if not os.path.exists(period_file):
                print(f"DEBUG: Error - File not found: {period_file}")
                messagebox.showerror("Error", f"Period file not found: {period_file}")
                return
                
            # Load data with proper columns
            print("DEBUG: Reading CSV file...")
            self.period_data = pd.read_csv(period_file)
            print(f"DEBUG: Loaded {len(self.period_data)} rows")
            print(f"DEBUG: Columns found: {', '.join(self.period_data.columns)}")
            
            # Create timestamp if it doesn't exist
            if 'timestamp' not in self.period_data.columns and 'frame_num' in self.period_data.columns:
                print("DEBUG: Converting frame_num to timestamp")
                # Convert frame numbers to timestamps (assuming 30fps)
                self.period_data['timestamp'] = self.period_data['frame_num'] * (1000.0 / 30.0)
                print(f"DEBUG: Created timestamps from frame_num: min={self.period_data['timestamp'].min()}, max={self.period_data['timestamp'].max()}")
            
            # Convert category to is_source boolean if needed
            if 'is_source' not in self.period_data.columns:
                print("DEBUG: Converting category to is_source boolean")
                self.period_data['is_source'] = self.period_data['category'] == 'source'
                print(f"DEBUG: Found {len(self.period_data[self.period_data['is_source']])} source items")
            
            # Ensure text_order exists
            if 'text_order' not in self.period_data.columns:
                print("DEBUG: Creating text_order column")
                # Create text_order based on text position in source/target lists
                source_texts = self.period_data[self.period_data['is_source']]['text'].unique()
                target_texts = self.period_data[~self.period_data['is_source']]['text'].unique()
                
                print(f"DEBUG: Found {len(source_texts)} unique source texts and {len(target_texts)} unique target texts")
                
                source_text_to_order = {text: i for i, text in enumerate(source_texts)}
                target_text_to_order = {text: i for i, text in enumerate(target_texts)}
                
                self.period_data['text_order'] = self.period_data.apply(
                    lambda row: source_text_to_order.get(row['text'], 0) if row['is_source'] 
                    else target_text_to_order.get(row['text'], 0), axis=1
                )
                print("DEBUG: Created text_order column")
            
            # Ensure duration exists for visualization sizing
            if 'duration' not in self.period_data.columns:
                print("DEBUG: Creating duration column")
                # Use confidence as a proxy for duration if available, otherwise use default
                if 'conf' in self.period_data.columns:
                    self.period_data['duration'] = self.period_data['conf'] / 100.0  # Scale confidence to 0-1
                else:
                    self.period_data['duration'] = 0.5  # Default duration
                print("DEBUG: Created duration column")
            
            # Ensure phase column exists
            if 'phase' not in self.period_data.columns:
                print("DEBUG: Creating phase column")
                # Determine phase based on timestamp and text order
                self.period_data['phase'] = 'production'  # Default phase
                
                # Find orientation phase (first reading of source text)
                first_source_read = self.period_data[
                    (self.period_data['is_source']) & 
                    (self.period_data['timestamp'] == self.period_data['timestamp'].min())
                ]
                if not first_source_read.empty:
                    self.period_data.loc[
                        (self.period_data['timestamp'] <= first_source_read['timestamp'].max()) &
                        (self.period_data['is_source']),
                        'phase'
                    ] = 'orientation'
                
                # Find revision phase (revisiting previously translated text)
                if not self.period_data[~self.period_data['is_source']].empty:
                    target_timestamps = self.period_data[~self.period_data['is_source']]['timestamp']
                    for idx, row in self.period_data.iterrows():
                        if row['is_source'] and row['timestamp'] > target_timestamps.min():
                            self.period_data.loc[idx, 'phase'] = 'revision'
                
                print("DEBUG: Created phase column")
            
            # Update text lists
            print("DEBUG: Updating text listboxes")
            source_texts = self.period_data[self.period_data['is_source']]['text'].unique()
            target_texts = self.period_data[~self.period_data['is_source']]['text'].unique()
            self.update_text_listboxes(source_texts, target_texts)
            
            # Print final data summary
            print("\nDEBUG: Final Data Summary:")
            print(f"Total rows: {len(self.period_data)}")
            print(f"Source texts: {len(source_texts)}")
            print(f"Target texts: {len(target_texts)}")
            print(f"Time range: {self.period_data['timestamp'].min():.2f} to {self.period_data['timestamp'].max():.2f}")
            print(f"Phases: {self.period_data['phase'].unique()}")
            
            # Enable visualization
            self.visualize_button.config(state=tk.NORMAL)
            print("DEBUG: Data loading complete\n")
                
        except Exception as e:
            print(f"\nDEBUG: Error loading period data: {str(e)}")
            print("DEBUG: Stack trace:")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Error loading period data: {str(e)}")
            self.period_data = None
    
    def parse_period_time_range(self, period):
        """Parse period identifier to get time range."""
        try:
            # Handle ocr_results_period_N format
            if period.startswith('ocr_results_period_'):
                # For this format, we'll read the actual data file to determine time range
                period_file = os.path.join(self.processed_dir, self.current_participant, 
                                         self.current_run, f"{period}.csv")
                if os.path.exists(period_file):
                    df = pd.read_csv(period_file)
                    if 'timestamp' in df.columns:
                        return df['timestamp'].min(), df['timestamp'].max()
                return 0, float('inf')  # Default range if file can't be read
                
            # Fallback to MM:SS-MM:SS format
            if '-' in period:
                start_str, end_str = period.split('-')
                start_min, start_sec = map(int, start_str.split(':'))
                end_min, end_sec = map(int, end_str.split(':'))
                return start_min * 60 + start_sec, end_min * 60 + end_sec
            
            # If neither format works, use a default range
            return 0, float('inf')
            
        except Exception as e:
            print(f"Period parsing error: {e}")
            # Use a default range instead of showing error
            return 0, float('inf')
    
    def update_text_listboxes(self, source_texts, target_texts):
        """Update the source and target text listboxes."""
        # Clear listboxes
        self.source_listbox.delete(0, tk.END)
        self.target_listbox.delete(0, tk.END)
        
        # Check if we have stored counts from emergency sampling
        if hasattr(self, '_source_counts') and self._source_counts is not None:
            source_counts = self._source_counts
            # Reset after use to avoid stale data
            del self._source_counts
        else:
            # Use normal counts from period data
            source_counts = self.period_data[self.period_data['category'] == 'source']['text'].value_counts()
        
        if hasattr(self, '_target_counts') and self._target_counts is not None:
            target_counts = self._target_counts
            # Reset after use to avoid stale data
            del self._target_counts
        else:
            # Only compute if 'target' exists in the data
            if 'target' in self.period_data['category'].values:
                target_counts = self.period_data[self.period_data['category'] == 'target']['text'].value_counts()
            else:
                target_counts = pd.Series()
        
        # Process and sort the text segments for better visualization
        # For source texts, try to sort them in a logical order (by frequency of appearance)
        
        # Convert Series to list to avoid ambiguous truth value errors
        source_texts_list = []
        if hasattr(source_texts, 'index'):  # It's a Series
            source_texts_list = sorted(source_texts.index.tolist(), key=lambda x: (-source_counts.get(x, 0), x))
        else:  # It's already a list or other iterable
            source_texts_list = sorted(source_texts, key=lambda x: (-source_counts.get(x, 0), x))
        
        # Add source texts
        for text in source_texts_list:
            count = source_counts.get(text, 0)
            self.source_listbox.insert(tk.END, f"{text} ({count})")
        
        # Add target texts
        target_texts_list = []
        if target_texts is not None and len(target_texts) > 0:
            if hasattr(target_texts, 'index'):  # It's a Series
                target_texts_list = sorted(target_texts.index.tolist(), key=lambda x: (-target_counts.get(x, 0), x))
            else:  # It's already a list or other iterable
                target_texts_list = sorted(target_texts, key=lambda x: (-target_counts.get(x, 0), x))
                
            for text in target_texts_list:
                count = target_counts.get(text, 0)
                self.target_listbox.insert(tk.END, f"{text} ({count})")
        
        # Store the current text order (without count annotation)
        self.source_text_order = source_texts_list
        self.target_text_order = target_texts_list
        
        # Update UI to reflect the new data
        self.visualize_button.config(state=tk.NORMAL if len(source_texts_list) > 0 else tk.DISABLED)
    
    def move_text_item(self, listbox, direction):
        """Move a selected item up or down in the listbox."""
        selected = listbox.curselection()
        if not selected:
            return
        
        index = selected[0]
        if direction < 0 and index == 0:
            return  # Already at the top
        if direction > 0 and index == listbox.size() - 1:
            return  # Already at the bottom
        
        # Get the text of the selected item
        text = listbox.get(index)
        
        # Delete the item
        listbox.delete(index)
        
        # Insert at new position
        new_index = index + direction
        listbox.insert(new_index, text)
        
        # Select the item at its new position
        listbox.selection_set(new_index)
        
        # Update text order
        if listbox == self.source_listbox:
            self.source_text_order = [listbox.get(i) for i in range(listbox.size())]
        else:
            self.target_text_order = [listbox.get(i) for i in range(listbox.size())]
    
    def remove_text_item(self, listbox):
        """Remove a selected item from the listbox."""
        selected = listbox.curselection()
        if not selected:
            return
        
        # Delete the item
        listbox.delete(selected[0])
        
        # Update text order
        if listbox == self.source_listbox:
            self.source_text_order = [listbox.get(i) for i in range(listbox.size())]
        else:
            self.target_text_order = [listbox.get(i) for i in range(listbox.size())]
    
    def add_fixation_units(self):
        """Add fixation units to the visualization."""
        if self.period_data is None or len(self.period_data) == 0:
            return
        
        try:
            # Get timestamp range
            min_time = self.period_data['timestamp'].min()
            
            # Process source fixation units
            self.add_axis_fixation_units(
                is_source=True, 
                data_df=self.period_data[self.period_data['category'] == 'source'].copy(),
                min_time=min_time
            )
            
            # Process target fixation units if target axis exists
            if self.target_axis and 'target' in self.period_data['category'].values:
                self.add_axis_fixation_units(
                    is_source=False, 
                    data_df=self.period_data[self.period_data['category'] == 'target'].copy(),
                    min_time=min_time
                )
                
        except Exception as e:
            print(f"Error adding fixation units: {e}")
            import traceback
            traceback.print_exc()
            
    def add_axis_fixation_units(self, is_source, data_df, min_time):
        """Add fixation units to a specific axis."""
        # Create normalized time column
        data_df['normalized_time'] = data_df['timestamp'] - min_time
        data_df = data_df.sort_values('normalized_time')
        
        # Simple fixation unit detection: group by text and proximity in time
        fixation_units = []
        current_unit = {'text': None, 'start': 0, 'end': 0, 'position': 0}
        
        # Create mappings from text to position in the display order
        if is_source:
            text_order = [self.source_listbox.get(i).split(" (")[0] for i in range(self.source_listbox.size())]
            text_to_pos = {text: i for i, text in enumerate(text_order)}
            target_axis = self.plot
        else:
            text_order = [self.target_listbox.get(i).split(" (")[0] for i in range(self.target_listbox.size())]
            text_to_pos = {text: i for i, text in enumerate(text_order)}
            target_axis = self.target_axis
        
        # Threshold for time gap between fixations in the same unit (ms)
        time_threshold = 300
        
        for idx, row in data_df.iterrows():
            row_text = str(row['text'])
            if row_text not in text_to_pos:
                continue
            
            position = text_to_pos[row_text]
            time = row['normalized_time']
            
            if current_unit['text'] is None:
                # First fixation
                current_unit = {
                    'text': row_text,
                    'start': time,
                    'end': time,
                    'position': position
                }
            elif (row_text == current_unit['text'] and 
                  time - current_unit['end'] < time_threshold):
                # Same text and close in time - extend the current unit
                current_unit['end'] = time
            else:
                # New fixation unit
                fixation_units.append(current_unit)
                current_unit = {
                    'text': row_text,
                    'start': time,
                    'end': time,
                    'position': position
                }
        
        # Add the last unit
        if current_unit['text'] is not None:
            fixation_units.append(current_unit)
        
        # Choose colors for fixation units based on source/target
        if is_source:
            rect_color = 'royalblue'
            text_color = 'darkblue'
        else:
            rect_color = 'forestgreen'
            text_color = 'darkgreen'
            
        # Draw rectangles for fixation units
        for i, unit in enumerate(fixation_units):
            rect = patches.Rectangle(
                (unit['start'], unit['position'] - 0.4),
                unit['end'] - unit['start'], 0.8,
                alpha=0.25, facecolor=rect_color, edgecolor='black', linewidth=1
            )
            target_axis.add_patch(rect)
            
            # Add label with duration
            duration = int(unit['end'] - unit['start'])
            target_axis.text(
                unit['start'] + (unit['end'] - unit['start'])/2, 
                unit['position'] + 0.3,
                f"{duration}ms", 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8,
                color=text_color,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none')
            )
    
    def apply_filters(self):
        """Apply filtering based on fixation count thresholds."""
        if self.period_data is None:
            messagebox.showinfo("No Data", "Please load data first before applying filters.")
            return
            
        source_threshold = self.source_threshold_var.get()
        target_threshold = self.target_threshold_var.get()
        
        try:
            # Get original unfiltered lists
            source_counts = self.period_data[self.period_data['category'] == 'source']['text'].value_counts()
            if 'target' in self.period_data['category'].values:
                target_counts = self.period_data[self.period_data['category'] == 'target']['text'].value_counts()
            else:
                target_counts = pd.Series()
            
            # Apply source filter
            filtered_source_texts = [text for text in self.source_text_order 
                                    if source_counts.get(text, 0) >= source_threshold]
            
            # Apply target filter
            filtered_target_texts = [text for text in self.target_text_order 
                                    if target_counts.get(text, 0) >= target_threshold]
            
            # Update listboxes with filtered items
            self.source_listbox.delete(0, tk.END)
            for text in filtered_source_texts:
                count = source_counts.get(text, 0)
                self.source_listbox.insert(tk.END, f"{text} ({count})")
            
            self.target_listbox.delete(0, tk.END)
            for text in filtered_target_texts:
                count = target_counts.get(text, 0)
                self.target_listbox.insert(tk.END, f"{text} ({count})")
            
            # Update stored text orders
            self.source_text_order = filtered_source_texts
            self.target_text_order = filtered_target_texts
            
            # Generate updated visualization
            self.generate_visualization()
            
            messagebox.showinfo("Filtering Applied", 
                              f"Filtered to {len(filtered_source_texts)} source texts and {len(filtered_target_texts)} target texts.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error applying filters: {e}")
            import traceback
            traceback.print_exc()
            
    def apply_simplification(self):
        """Apply visualization simplification settings."""
        if self.period_data is None:
            messagebox.showinfo("No Data", "Please load data first before applying simplification.")
            return
        
        try:
            print("\nDEBUG: Applying visualization simplification")
            
            # Get current settings
            mode = self.display_mode_var.get()
            max_points = self.max_points_var.get()
            time_start_pct = self.time_range_start_var.get() / 100.0  # Convert to percentage
            time_end_pct = self.time_range_end_var.get() / 100.0      # Convert to percentage
            
            # Create a copy of the original data to simplify
            simplified_data = self.period_data.copy()
            original_size = len(simplified_data)
            
            print(f"DEBUG: Original data size: {original_size} points")
            print(f"DEBUG: Simplification mode: {mode}")
            print(f"DEBUG: Max points: {max_points}")
            print(f"DEBUG: Time range: {time_start_pct*100}% to {time_end_pct*100}%")
            
            # Apply time range filter
            time_min = simplified_data['timestamp'].min()
            time_max = simplified_data['timestamp'].max()
            time_range = time_max - time_min
            
            start_time = time_min + (time_range * time_start_pct)
            end_time = time_min + (time_range * time_end_pct)
            
            simplified_data = simplified_data[
                (simplified_data['timestamp'] >= start_time) & 
                (simplified_data['timestamp'] <= end_time)
            ]
            
            time_filtered_size = len(simplified_data)
            print(f"DEBUG: After time filtering: {time_filtered_size} points")
            
            # If All Points mode, skip further simplification
            if mode == "All Points":
                print("DEBUG: Using all points mode - no further simplification")
                # No further simplification needed
                pass
                
            # Apply sampling in Sampling mode
            elif mode == "Sampling":
                print("DEBUG: Using sampling mode")
                # Calculate sampling rate to get approximately max_points
                sample_size = min(max_points, time_filtered_size)
                if time_filtered_size > sample_size:
                    sampling_rate = max(int(time_filtered_size / sample_size), 1)
                    print(f"DEBUG: Sampling rate: every {sampling_rate}th point")
                    
                    # Separate source and target data for better balanced sampling
                    source_data = simplified_data[simplified_data['is_source'] == True]
                    target_data = simplified_data[simplified_data['is_source'] == False]
                    
                    source_sample = source_data.iloc[::sampling_rate]
                    target_sample = target_data.iloc[::sampling_rate]
                    
                    # Combine the samples
                    simplified_data = pd.concat([source_sample, target_sample]).sort_values('timestamp')
                    
                    print(f"DEBUG: After sampling: {len(simplified_data)} points")
                
            # Apply clustering in Clustering mode
            elif mode == "Clustering":
                print("DEBUG: Using clustering mode")
                # Simple clustering by time and text
                from sklearn.cluster import KMeans
                
                # First, we separate source and target texts
                source_data = simplified_data[simplified_data['is_source'] == True]
                target_data = simplified_data[simplified_data['is_source'] == False]
                
                # Function to cluster data
                def cluster_points(data, max_clusters):
                    if len(data) <= max_clusters or len(data) == 0:
                        return data
                    
                    # Extract features for clustering
                    features = data[['timestamp', 'text_order']].copy()
                    # Normalize features
                    features = (features - features.mean()) / features.std()
                    
                    # Apply K-means clustering
                    n_clusters = min(max_clusters, len(data) // 2)
                    print(f"DEBUG: Clustering {len(data)} points into {n_clusters} clusters")
                    
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    data['cluster'] = kmeans.fit_predict(features)
                    
                    # Get cluster centers (take the point closest to each center)
                    clustered_results = []
                    for cluster_id in range(n_clusters):
                        cluster_points = data[data['cluster'] == cluster_id]
                        if len(cluster_points) > 0:
                            # Take the point closest to the center of the cluster
                            centroid = cluster_points[['timestamp', 'text_order']].mean()
                            idx = ((cluster_points['timestamp'] - centroid['timestamp'])**2 + 
                                  (cluster_points['text_order'] - centroid['text_order'])**2).idxmin()
                            clustered_results.append(data.loc[idx])
                    
                    return pd.DataFrame(clustered_results)
                
                # Calculate number of clusters based on max_points
                source_max = max(10, int(max_points * len(source_data) / 
                                      (len(source_data) + len(target_data))))
                target_max = max(10, max_points - source_max)
                
                # Apply clustering
                clustered_source = cluster_points(source_data, source_max)
                clustered_target = cluster_points(target_data, target_max)
                
                # Combine the clustered data
                simplified_data = pd.concat([clustered_source, clustered_target]).sort_values('timestamp')
                print(f"DEBUG: After clustering: {len(simplified_data)} points")
                
            # Apply summarization in Summary mode
            elif mode == "Summary":
                print("DEBUG: Using summary mode")
                # Create a summary of the data using text_order as the primary organization
                source_data = simplified_data[simplified_data['is_source'] == True]
                target_data = simplified_data[simplified_data['is_source'] == False]
                
                # Function to summarize data by text_order
                def summarize_by_text(data, max_summaries):
                    if len(data) == 0:
                        return data
                        
                    # Group by text_order
                    text_groups = data.groupby('text_order')
                    summary_rows = []
                    
                    for text_order, group in text_groups:
                        # For each text, take first appearance, last appearance, and duration
                        first_point = group.iloc[group['timestamp'].argmin()]
                        last_point = group.iloc[group['timestamp'].argmax()]
                        
                        # Add summary rows
                        summary_rows.append(first_point)
                        if not first_point.equals(last_point):
                            summary_rows.append(last_point)
                    
                    result = pd.DataFrame(summary_rows)
                    
                    # If still too many points, sample them
                    if len(result) > max_summaries:
                        result = result.sample(max_summaries)
                    
                    return result.sort_values('timestamp')
                
                # Calculate max summaries based on max_points
                source_max = max(5, int(max_points * len(source_data) / 
                                     (len(source_data) + len(target_data))))
                target_max = max(5, max_points - source_max)
                
                # Apply summarization
                summarized_source = summarize_by_text(source_data, source_max)
                summarized_target = summarize_by_text(target_data, target_max)
                
                # Combine the summarized data
                simplified_data = pd.concat([summarized_source, summarized_target]).sort_values('timestamp')
                print(f"DEBUG: After summarization: {len(simplified_data)} points")
            
            # Store the simplified data for visualization
            self.simplified_data = simplified_data
            print(f"DEBUG: Final simplified data size: {len(simplified_data)} points")
            
            # Generate visualization with simplified data
            self.generate_visualization(use_simplified=True)
            
            # Show confirmation
            messagebox.showinfo("Simplification Applied", 
                              f"Applied {mode} mode with {max_points} max points.\n"
                              f"Time range: {time_start_pct*100:.0f}% to {time_end_pct*100:.0f}%\n"
                              f"Reduced from {original_size} to {len(simplified_data)} points.")
            
        except Exception as e:
            print(f"DEBUG: Error applying simplification: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Error applying simplification: {str(e)}")
            
            # Fall back to normal visualization
            self.generate_visualization()