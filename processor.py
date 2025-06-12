#!/usr/bin/env python3
"""
Processor module for the eye-tracking analysis system.
Handles data processing using the core functions from simple_ocr_map.py.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import queue
import importlib.util
import json
from datetime import datetime
import subprocess
import pandas as pd
import time

# Import utility functions
from utils import create_resizable_sidebar

class Processor:
    """Handles processing of eye-tracking data."""
    
    def __init__(self, input_dir, processed_dir):
        """Initialize the processor."""
        self.input_dir = input_dir
        self.processed_dir = processed_dir
        self.processing_queue = queue.Queue()
        self.log_queue = queue.Queue()
        self.processing_thread = None
        self.log_thread = None
        self.is_processing = False
        
        # Load the core processing module
        try:
            spec = importlib.util.spec_from_file_location("simple_ocr_map", "simple_ocr_map.py")
            self.ocr_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.ocr_module)
            self.ocr_module_loaded = True
        except Exception as e:
            print(f"Error loading OCR module: {e}")
            self.ocr_module_loaded = False
    
    def create_ui(self, parent, project_manager, period_manager):
        """Create the user interface for the processor."""
        self.parent = parent
        self.project_manager = project_manager
        self.period_manager = period_manager
        self.participant_var = tk.StringVar()  # Initialize the participant_var
        
        # Create main frame
        main_frame = ttk.Frame(parent, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create paned window to allow resizing
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel using the utility function
        left_container, left_panel_content = create_resizable_sidebar(paned_window, min_width=350)
        
        # Add a label frame around the content for visual organization
        left_panel = ttk.LabelFrame(left_panel_content, text="Processing Settings", padding=10)
        left_panel.pack(fill=tk.X, expand=True)
        
        # Participant selection
        participant_frame = ttk.Frame(left_panel)
        participant_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(participant_frame, text="Participants:").pack(anchor=tk.W)
        
        self.participant_listbox = tk.Listbox(participant_frame, selectmode=tk.EXTENDED, height=10)
        self.participant_listbox.pack(fill=tk.X, pady=5)
        
        participant_buttons = ttk.Frame(participant_frame)
        participant_buttons.pack(fill=tk.X)
        
        ttk.Button(participant_buttons, text="Select All", 
                  command=self.select_all_participants).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(participant_buttons, text="Clear Selection", 
                  command=self.clear_participant_selection).pack(side=tk.LEFT)
        ttk.Button(participant_buttons, text="Find Data Files", 
                  command=self.find_data_files).pack(side=tk.RIGHT)
        
        # Processing parameters
        params_frame = ttk.LabelFrame(left_panel, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=10)
        
        # Frame interval
        interval_frame = ttk.Frame(params_frame)
        interval_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(interval_frame, text="Frame Interval:").pack(side=tk.LEFT)
        self.frame_interval_var = tk.IntVar(value=5)
        ttk.Spinbox(interval_frame, from_=1, to=30, textvariable=self.frame_interval_var, 
                   width=5).pack(side=tk.RIGHT)
        
        # Confidence threshold
        conf_frame = ttk.Frame(params_frame)
        conf_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(conf_frame, text="Confidence Threshold (%):").pack(side=tk.LEFT)
        self.confidence_var = tk.IntVar(value=60)
        ttk.Spinbox(conf_frame, from_=0, to=100, textvariable=self.confidence_var, 
                   width=5).pack(side=tk.RIGHT)
        
        # Distance threshold
        dist_frame = ttk.Frame(params_frame)
        dist_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(dist_frame, text="Distance Threshold (px):").pack(side=tk.LEFT)
        self.distance_var = tk.IntVar(value=10)
        ttk.Spinbox(dist_frame, from_=1, to=100, textvariable=self.distance_var, 
                   width=5).pack(side=tk.RIGHT)
        
        # Word level processing
        word_frame = ttk.Frame(params_frame)
        word_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(word_frame, text="Word-Level Processing:").pack(side=tk.LEFT)
        self.word_level_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(word_frame, variable=self.word_level_var).pack(side=tk.RIGHT)
        
        # Performance options
        perf_frame = ttk.LabelFrame(left_panel, text="Performance Options", padding=10)
        perf_frame.pack(fill=tk.X, pady=10)
        
        # Fast Mode option
        fast_mode_frame = ttk.Frame(perf_frame)
        fast_mode_frame.pack(fill=tk.X, pady=5)
        ttk.Label(fast_mode_frame, text="Fast OCR Mode:").pack(side=tk.LEFT)
        self.fast_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fast_mode_frame, variable=self.fast_mode_var).pack(side=tk.RIGHT)
        
        # Frame Preprocessing option
        preprocess_frame = ttk.Frame(perf_frame)
        preprocess_frame.pack(fill=tk.X, pady=5)
        ttk.Label(preprocess_frame, text="Preprocess Frames:").pack(side=tk.LEFT)
        self.preprocess_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preprocess_frame, variable=self.preprocess_var).pack(side=tk.RIGHT)
        
        # Adaptive Batch Sizing option
        adaptive_frame = ttk.Frame(perf_frame)
        adaptive_frame.pack(fill=tk.X, pady=5)
        ttk.Label(adaptive_frame, text="Adaptive Batch Sizing:").pack(side=tk.LEFT)
        self.adaptive_batch_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(adaptive_frame, variable=self.adaptive_batch_var).pack(side=tk.RIGHT)
        
        # Frame Cache option
        cache_frame = ttk.Frame(perf_frame)
        cache_frame.pack(fill=tk.X, pady=5)
        ttk.Label(cache_frame, text="Enable Frame Caching:").pack(side=tk.LEFT)
        self.cache_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cache_frame, variable=self.cache_var).pack(side=tk.RIGHT)
        
        # Parallel Processing option
        parallel_frame = ttk.Frame(perf_frame)
        parallel_frame.pack(fill=tk.X, pady=5)
        ttk.Label(parallel_frame, text="Parallel Processing:").pack(side=tk.LEFT)
        self.parallel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parallel_frame, variable=self.parallel_var).pack(side=tk.RIGHT)
        
        # Batch Size
        batch_size_frame = ttk.Frame(perf_frame)
        batch_size_frame.pack(fill=tk.X, pady=5)
        ttk.Label(batch_size_frame, text="Initial Batch Size:").pack(side=tk.LEFT)
        self.batch_size_var = tk.IntVar(value=256)
        ttk.Spinbox(batch_size_frame, from_=64, to=512, textvariable=self.batch_size_var, 
                   width=5).pack(side=tk.RIGHT)
        
        # Time period processing
        period_frame = ttk.LabelFrame(left_panel, text="Time Periods", padding=10)
        period_frame.pack(fill=tk.X, pady=10)
        
        self.period_var = tk.StringVar(value="all_periods")
        ttk.Radiobutton(period_frame, text="Use All Defined Periods", 
                       variable=self.period_var, value="all_periods").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(period_frame, text="Process Entire Video", 
                       variable=self.period_var, value="no_periods").pack(anchor=tk.W, pady=2)
        
        # Process button
        process_frame = ttk.Frame(left_panel)
        process_frame.pack(fill=tk.X, pady=10)
        
        self.process_button = ttk.Button(process_frame, text="Process Selected Participants", 
                                        command=self.process_participants)
        self.process_button.pack(fill=tk.X, pady=5)
        
        self.cancel_button = ttk.Button(process_frame, text="Cancel Processing", 
                                       command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.pack(fill=tk.X)
        
        # Progress bar
        progress_frame = ttk.Frame(left_panel)
        progress_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(progress_frame, text="Progress:").pack(anchor=tk.W)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=200)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Create right panel for log output with a label frame
        right_container = ttk.Frame(paned_window)
        right_frame = ttk.LabelFrame(right_container, text="Processing Log", padding=10)
        right_frame.pack(fill=tk.BOTH, expand=True)
        
        # Log text widget
        self.log_text = tk.Text(right_frame, wrap=tk.WORD, width=50, height=20)
        log_scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add panels to paned window
        paned_window.add(left_container, weight=1)
        paned_window.add(right_container, weight=2)
        
        # Disable editing of log
        self.log_text.configure(state=tk.DISABLED)
        
        # Set up log queue and processing
        self.log_queue = queue.Queue()
        self.root = self.parent.winfo_toplevel()
        self.root.after(100, self.process_log_queue)
        
        # Load participants
        self.refresh_participants()
        
        # Initialize processing state
        self.is_processing = False
        self.ocr_module = self.import_ocr_module()
    
    def find_data_files(self):
        """Help the user locate eye-tracking data files."""
        try:
            # Get selected participants
            selected_indices = self.participant_listbox.curselection()
            if not selected_indices:
                messagebox.showinfo("Selection Required", "Please select at least one participant.")
                return
            
            # Extract the first selected participant
            participant = self.participant_listbox.get(selected_indices[0])
            participant_dir = self.project_manager.get_participant_data_path(participant)
            
            # Create results window
            results_window = tk.Toplevel(self.parent)
            results_window.title(f"Eye-Tracking Data Files - {participant}")
            results_window.geometry("800x500")
            results_window.minsize(600, 400)
            
            # Create frame for content
            frame = ttk.Frame(results_window, padding=10)
            frame.pack(fill=tk.BOTH, expand=True)
            
            # Header
            ttk.Label(frame, text=f"Eye-Tracking Data Files for {participant}", 
                     font=("Arial", 12, "bold")).pack(pady=(0, 10))
            
            # Create treeview for files
            columns = ("filename", "score", "columns", "rows", "is_raw", "has_coords", "has_time")
            tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
            
            # Define column headings
            tree.heading("filename", text="Filename")
            tree.heading("score", text="Score")
            tree.heading("columns", text="Columns")
            tree.heading("rows", text="Rows")
            tree.heading("is_raw", text="Is Raw")
            tree.heading("has_coords", text="Has Coordinates")
            tree.heading("has_time", text="Has Timestamp")
            
            # Column widths
            tree.column("filename", width=250)
            tree.column("score", width=60)
            tree.column("columns", width=70)
            tree.column("rows", width=70)
            tree.column("is_raw", width=60)
            tree.column("has_coords", width=120)
            tree.column("has_time", width=120)
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            # Pack tree and scrollbar
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Status label
            status_var = tk.StringVar()
            status_var.set("Scanning for files...")
            status = ttk.Label(frame, textvariable=status_var)
            status.pack(pady=10)
            
            # Details frame
            details_frame = ttk.LabelFrame(frame, text="File Details", padding=10)
            details_frame.pack(fill=tk.X, pady=10)
            
            details_text = tk.Text(details_frame, wrap=tk.WORD, height=10)
            details_scroll = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=details_text.yview)
            details_text.configure(yscrollcommand=details_scroll.set)
            
            details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            details_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Button frame
            button_frame = ttk.Frame(frame)
            button_frame.pack(fill=tk.X, pady=10)
            
            def view_file():
                selected = tree.selection()
                if not selected:
                    return
                
                item = tree.item(selected[0])
                file_path = item['values'][7]  # Full path is hidden as the last value
                
                try:
                    # Try to open the file with default app
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', file_path])
                    elif sys.platform == 'win32':  # Windows
                        os.startfile(file_path)
                    else:  # Linux
                        subprocess.run(['xdg-open', file_path])
                except Exception as e:
                    messagebox.showerror("Error", f"Could not open file: {str(e)}")
            
            def analyze_file():
                selected = tree.selection()
                if not selected:
                    return
                
                item = tree.item(selected[0])
                file_path = item['values'][7]  # Full path is hidden as the last value
                
                try:
                    # Run CSV analyzer
                    subprocess.Popen(["python", "csv_analyzer.py", file_path, "--verbose"])
                except Exception as e:
                    messagebox.showerror("Error", f"Could not analyze file: {str(e)}")
            
            ttk.Button(button_frame, text="View File", command=view_file).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Analyze File", command=analyze_file).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Close", command=results_window.destroy).pack(side=tk.RIGHT, padx=5)
            
            # When selection changes, update details
            def on_select(event):
                selected = tree.selection()
                if not selected:
                    return
                
                item = tree.item(selected[0])
                file_path = item['values'][7]  # Full path is hidden as the last value
                
                # Load file details
                try:
                    from csv_analyzer import csv_inspect
                    info = csv_inspect(file_path, sample_rows=10)
                    
                    # Clear and update text
                    details_text.delete(1.0, tk.END)
                    
                    if "error" in info:
                        details_text.insert(tk.END, f"Error: {info['error']}")
                    else:
                        details_text.insert(tk.END, f"File: {os.path.basename(file_path)}\n")
                        details_text.insert(tk.END, f"Size: {info['filesize_kb']:.1f} KB\n")
                        details_text.insert(tk.END, f"Rows: {info['row_count']}\n")
                        details_text.insert(tk.END, f"Columns: {info['column_count']}\n\n")
                        
                        details_text.insert(tk.END, "Column names:\n")
                        for i, col in enumerate(info['columns']):
                            details_text.insert(tk.END, f"  {i+1}. {col}\n")
                    
                except Exception as e:
                    details_text.delete(1.0, tk.END)
                    details_text.insert(tk.END, f"Error analyzing file: {str(e)}")
            
            tree.bind("<<TreeviewSelect>>", on_select)
            
            # Function to scan for files
            def scan_files():
                try:
                    from csv_analyzer import find_eyetracking_files
                    files = find_eyetracking_files(participant_dir, verbose=False)
                    
                    # Update tree with results
                    for item in tree.get_children():
                        tree.delete(item)
                    
                    for file_info in files:
                        tree.insert("", tk.END, values=(
                            file_info['filename'],
                            file_info['score'],
                            file_info['columns'],
                            file_info['rows'],
                            "Yes" if file_info['is_raw'] else "No",
                            "Yes" if file_info['has_coordinates'] else "No",
                            "Yes" if file_info['has_timestamp'] else "No",
                            file_info['path']  # Hidden path
                        ))
                    
                    status_var.set(f"Found {len(files)} potential eye-tracking data files")
                    
                    # Select first item if available
                    if files:
                        first_item = tree.get_children()[0]
                        tree.selection_set(first_item)
                        tree.focus(first_item)
                        on_select(None)  # Trigger selection event
                    
                except Exception as e:
                    status_var.set(f"Error scanning files: {str(e)}")
                    messagebox.showerror("Error", f"Error scanning for files: {str(e)}")
            
            # Start scanning in a separate thread
            import threading
            threading.Thread(target=scan_files, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error finding data files: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def refresh_participants(self):
        """Refresh the participant dropdown."""
        participants = self.project_manager.get_selected_participants()
        
        self.participant_listbox.delete(0, tk.END)
        self.participant_listbox.insert(0, *participants)
        
        if participants and (not self.participant_var.get() or self.participant_var.get() not in participants):
            self.participant_var.set(participants[0])
            self.load_participant_info()
        elif not participants:
            self.participant_var.set("")
    
    def load_participant_info(self, event=None):
        """Load information about the selected participant."""
        participant = self.participant_var.get()
        if not participant:
            return
        
        # Get periods for this participant
        periods = self.period_manager.get_periods(participant)
        
        # Log info about participant
        self.log_message(f"Selected participant: {participant}")
        self.log_message(f"Defined time periods: {len(periods)}")
        
        # Check for input files
        participant_dir = self.project_manager.get_participant_data_path(participant)
        video_files = []
        csv_files = []
        
        for root, _, files in os.walk(participant_dir):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    video_files.append(os.path.join(root, file))
                elif file.lower().endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        self.log_message(f"Found {len(video_files)} video files and {len(csv_files)} CSV files")
    
    def log_message(self, message):
        """Add a message to the log."""
        self.log_queue.put(message)
    
    def start_log_monitor(self):
        """Start the log monitoring thread."""
        def monitor_log():
            while True:
                try:
                    message = self.log_queue.get(timeout=0.1)
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    self.log_text.config(state=tk.NORMAL)
                    self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
                    self.log_text.see(tk.END)
                    self.log_text.config(state=tk.DISABLED)
                    
                    self.log_queue.task_done()
                except queue.Empty:
                    pass
        
        self.log_thread = threading.Thread(target=monitor_log, daemon=True)
        self.log_thread.start()
    
    def start_processing(self):
        """Start the processing operation."""
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing is already in progress")
            return
            
        if not self.ocr_module_loaded:
            messagebox.showerror("Error", "OCR module not loaded. Check simple_ocr_map.py file.")
            return
        
        # Get processing configuration
        participant = self.participant_var.get() if not self.process_all_var.get() else None
        frame_interval = self.frame_interval_var.get()
        confidence_threshold = self.confidence_var.get()
        distance_threshold = self.distance_var.get()
        word_level = self.word_level_var.get()
        mode = self.mode_var.get()
        
        # Validate inputs
        if not participant and not self.process_all_var.get():
            messagebox.showerror("Error", "No participant selected")
            return
            
        participants_to_process = []
        if self.process_all_var.get():
            participants_to_process = self.project_manager.get_selected_participants()
            if not participants_to_process:
                messagebox.showerror("Error", "No participants selected in Project Management")
                return
        else:
            participants_to_process = [participant]
        
        # Update UI state
        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.status_var.set("Processing...")
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self.process_participants,
            args=(participants_to_process, frame_interval, confidence_threshold, distance_threshold, word_level, mode),
            daemon=True
        )
        self.processing_thread.start()
    
    def process_participants(self):
        """Process the selected participants."""
        if not self.ocr_module:
            messagebox.showerror("Error", "OCR module not loaded")
            return
            
        # Get selected participants
        selected_indices = self.participant_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("Selection Required", "Please select at least one participant.")
            return
            
        # Get processing parameters
        frame_interval = self.frame_interval_var.get()
        confidence_threshold = self.confidence_var.get()
        distance_threshold = self.distance_var.get()
        word_level = self.word_level_var.get()
        mode = self.period_var.get()
        
        # Get performance options
        fast_mode = self.fast_mode_var.get()
        preprocess = self.preprocess_var.get()
        adaptive_batch = self.adaptive_batch_var.get()
        use_cache = self.cache_var.get()
        parallel_processing = self.parallel_var.get()
        batch_size = self.batch_size_var.get()
        
        # Update UI state
        self.is_processing = True
        self.process_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.progress_var.set(0)
        
        # Get list of participants to process
        participants = [self.participant_listbox.get(i) for i in selected_indices]
        total_participants = len(participants)
        
        def process_thread():
            try:
                # Process each participant
                for i, participant in enumerate(participants):
                    if not self.is_processing:
                        break
                        
                    progress_offset = i / total_participants * 100
                    progress_segment = 1 / total_participants * 100
                    
                    # Log the participant being processed
                    self.log_message(f"Processing participant: {participant} ({i+1}/{total_participants})")
                    
                    # Process the participant
                    success = self.process_participant(
                        participant, 
                        frame_interval, 
                        confidence_threshold, 
                        distance_threshold, 
                        word_level, 
                        mode, 
                        progress_offset, 
                        progress_segment,
                        fast_mode,
                        preprocess,
                        adaptive_batch,
                        use_cache,
                        parallel_processing,
                        batch_size
                    )
                    
                    if not success:
                        self.log_message(f"Error processing participant {participant}")
                    
                    # Update progress
                    self.parent.after(0, lambda: self.progress_var.set((i+1) / total_participants * 100))
                    
                # Processing complete
                if self.is_processing:
                    self.log_message("Processing complete")
                    self.parent.after(0, lambda: messagebox.showinfo("Processing Complete", "All participants processed successfully"))
                else:
                    self.log_message("Processing cancelled")
                    self.parent.after(0, lambda: messagebox.showinfo("Processing Cancelled", "Processing was cancelled"))
                    
            except Exception as e:
                import traceback
                self.log_message(f"Error in processing thread: {str(e)}")
                self.log_message(traceback.format_exc())
                self.parent.after(0, lambda: messagebox.showerror("Processing Error", f"An error occurred during processing: {str(e)}"))
                
            finally:
                # Reset UI state
                self.is_processing = False
                self.parent.after(0, lambda: self.process_button.config(state=tk.NORMAL))
                self.parent.after(0, lambda: self.cancel_button.config(state=tk.DISABLED))
        
        # Start the thread
        threading.Thread(target=process_thread, daemon=True).start()

    def process_participant(self, participant, frame_interval, confidence_threshold, distance_threshold, word_level, mode, progress_offset, progress_segment, fast_mode=False, preprocess=True, adaptive_batch=True, use_cache=True, parallel_processing=True, batch_size=256):
        """Process a single participant."""
        try:
            # Get participant directory in output folder
            output_dir = os.path.join(self.processed_dir, participant)
            os.makedirs(output_dir, exist_ok=True)
            
            # Get participant input directory
            participant_data_path = self.project_manager.get_participant_data_path(participant)
            
            if not os.path.exists(participant_data_path):
                self.log_message(f"Error: Participant data directory not found: {participant_data_path}")
                return False
                
            # Create a new run directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(output_dir, f"run_{timestamp}")
            os.makedirs(run_dir, exist_ok=True)
            
            self.log_message(f"Created run directory: {run_dir}")
            
            # Find video and csv files
            video_files = []
            csv_files = []
            
            # First look in named participant directory
            for root, _, files in os.walk(participant_data_path):
                for file in files:
                    if file.lower().endswith(('.mp4', '.mov', '.avi')):
                        video_files.append(os.path.join(root, file))
                    elif file.lower().endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if not video_files:
                self.log_message(f"Error: No video files found for participant {participant}")
                return False
                
            if not csv_files:
                self.log_message(f"Error: No CSV files found for participant {participant}")
                return False
                
            # Select newest video and csv file
            video_file = max(video_files, key=os.path.getmtime)
            csv_file = max(csv_files, key=os.path.getmtime)
            
            self.log_message(f"Selected video file: {os.path.basename(video_file)}")
            self.log_message(f"Selected CSV file: {os.path.basename(csv_file)}")
            
            # Determine time periods to process
            time_periods = None
            periods_json = os.path.join(output_dir, "periods.json")
            
            if mode == "all_periods" and os.path.exists(periods_json):
                try:
                    with open(periods_json, 'r') as f:
                        periods_data = json.load(f)
                        if 'periods' in periods_data:
                            time_periods = [p['time_range'] for p in periods_data['periods']]
                            self.log_message(f"Using {len(time_periods)} defined time periods")
                except Exception as e:
                    self.log_message(f"Error loading time periods: {str(e)}")
                    time_periods = None
            
            # Save configuration
            config = {
                'video_file': video_file,
                'csv_file': csv_file,
                'frame_interval': frame_interval,
                'confidence_threshold': confidence_threshold,
                'distance_threshold': distance_threshold,
                'word_level': word_level,
                'time_periods': time_periods,
                'timestamp': timestamp,
                'fast_mode': fast_mode,
                'preprocess': preprocess,
                'adaptive_batch': adaptive_batch,
                'use_cache': use_cache,
                'parallel_processing': parallel_processing,
                'batch_size': batch_size
            }
            
            with open(os.path.join(run_dir, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            
            self.log_message("Saved processing configuration")
            
            # Set up OCR arguments
            output_file = os.path.join(run_dir, "mapped_gaze_to_text.csv")
            ocr_results_filename = "ocr_results.csv"
            
            args_dict = {
                'video': video_file,
                'csv': csv_file,
                'interval': frame_interval,
                'confidence': confidence_threshold,
                'distance': distance_threshold,
                'output': output_file,
                'word_level': word_level,
                'fast_mode': fast_mode,
                'no_preprocess': not preprocess,
                'no_adaptive': not adaptive_batch,
                'no_cache': not use_cache,
                'no_parallel': not parallel_processing,
                'batch_size': batch_size
            }
            
            if time_periods:
                args_dict['time_periods'] = time_periods
            
            self.log_message(f"Starting OCR processing with args: {args_dict}")
            
            # Process with OCR module
            success = self._process_with_args(args_dict, progress_offset, progress_segment, run_dir, ocr_results_filename)
            
            if success:
                self.log_message(f"Completed processing for participant {participant}")
                return True
            else:
                self.log_message(f"Processing failed for participant {participant}")
                return False
                
        except Exception as e:
            import traceback
            self.log_message(f"Error processing participant {participant}: {str(e)}")
            self.log_message(traceback.format_exc())
            return False
    
    def _process_with_args(self, args_dict, progress_offset, progress_segment, output_dir, ocr_results_filename):
        """Helper method to process with the given arguments."""
        # Capture stdout and stderr to direct to our log
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            # Setup for progress updates
            def progress_callback(current, total):
                if not self.is_processing:
                    return False  # Signal to cancel processing
                    
                progress = (current / total) * progress_segment
                self.parent.after(0, lambda: self.progress_var.set(progress_offset + progress))
                return True  # Continue processing
            
            # Redirect stdout
            sys.stdout = LogCapture(self.log_queue)
            sys.stderr = LogCapture(self.log_queue)
            
            # Log the arguments being passed to the module
            self.log_message(f"Processing with args: {args_dict}")
            
            # Run the processing directly with the dictionary
            result = self.ocr_module.process_with_args(args_dict, progress_callback)
            
            # Copy OCR results to output directory
            if os.path.exists("ocr_results.csv"):
                os.rename("ocr_results.csv", os.path.join(output_dir, ocr_results_filename))
                
            return result  # Return success/failure status
                
        except Exception as e:
            self.log_message(f"Error processing: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
            return False
            
        finally:
            # Restore stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    
    def cancel_processing(self):
        """Cancel the current processing operation."""
        if not self.is_processing:
            return
            
        self.is_processing = False
        self.status_var.set("Cancelling...")
        self.log_message("Cancelling processing...")
    
    def view_results(self):
        """View the processing results."""
        participant = self.participant_var.get()
        if not participant:
            messagebox.showerror("Error", "No participant selected")
            return
            
        output_dir = self.project_manager.get_participant_output_path(participant)
        
        if not os.path.exists(output_dir):
            messagebox.showerror("Error", "No results found for this participant")
            return
            
        try:
            # On macOS, use open to open Finder at the output directory
            if sys.platform == "darwin":
                os.system(f"open '{output_dir}'")
            # On Windows, use explorer
            elif sys.platform == "win32":
                os.system(f"explorer '{output_dir}'")
            # On Linux, try to use xdg-open
            else:
                os.system(f"xdg-open '{output_dir}'")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open results directory: {str(e)}")
    
    def select_all_participants(self):
        """Select all participants in the listbox."""
        self.participant_listbox.select_set(0, tk.END)
    
    def clear_participant_selection(self):
        """Clear the participant selection."""
        self.participant_listbox.selection_clear(0, tk.END)
    
    def import_ocr_module(self):
        """Import the OCR module."""
        try:
            spec = importlib.util.spec_from_file_location("simple_ocr_map", "simple_ocr_map.py")
            ocr_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ocr_module)
            self.log_message("OCR module loaded successfully")
            return ocr_module
        except Exception as e:
            self.log_message(f"Error loading OCR module: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
            return None

    def process_log_queue(self):
        """Process messages in the log queue."""
        try:
            while True:
                message = self.log_queue.get_nowait()
                
                # Update log text
                self.log_text.configure(state=tk.NORMAL)
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
                self.log_text.see(tk.END)
                self.log_text.configure(state=tk.DISABLED)
                
                # Process any pending events
                self.root.update_idletasks()
        except queue.Empty:
            pass
        finally:
            # Schedule to run again
            self.root.after(100, self.process_log_queue)


class LogCapture:
    """Capture stdout/stderr for logging."""
    
    def __init__(self, log_queue):
        self.log_queue = log_queue
        self.buffer = ""
    
    def write(self, text):
        if text == '\n':
            if self.buffer:
                self.log_queue.put(self.buffer)
                self.buffer = ""
        else:
            self.buffer += text
    
    def flush(self):
        if self.buffer:
            self.log_queue.put(self.buffer)
            self.buffer = "" 