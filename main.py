#!/usr/bin/env python3
"""
Main entry point for the eye-tracking analysis system.
This module provides a user interface to coordinate all features.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, Menu
import webbrowser
from datetime import datetime
import subprocess

# Import our modules
from project_manager import ProjectManager
from period_manager import PeriodManager
from processor import Processor
from block_analyzer import BlockAnalyzer
from progress_visualizer import ProgressVisualizer

# Create processed output directory
PROCESSED_OUTPUT_DIR = "processed_output"
os.makedirs(PROCESSED_OUTPUT_DIR, exist_ok=True)

# Root directory for input data
INPUT_DATA_DIR = "input"

class EyeTrackingAnalysisApp:
    """Main application class for the eye-tracking analysis system."""
    
    def __init__(self, root):
        """Initialize the application."""
        self.root = root
        root.title("Eye-Tracking Analysis System")
        root.geometry("1200x800")
        
        # Configure styles for better UI
        self.configure_styles()
        
        # Create the notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize modules
        self.project_manager = ProjectManager(INPUT_DATA_DIR, PROCESSED_OUTPUT_DIR)
        self.period_manager = PeriodManager()
        self.processor = Processor(INPUT_DATA_DIR, PROCESSED_OUTPUT_DIR)
        self.block_analyzer = BlockAnalyzer(PROCESSED_OUTPUT_DIR)
        self.progress_visualizer = ProgressVisualizer(PROCESSED_OUTPUT_DIR)
        
        # Create tabs
        self.create_tabs()
        
        # Create menu
        self.create_menu()
        
        # Set up event handling for tabs
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
    
    def configure_styles(self):
        """Configure ttk styles for the application."""
        style = ttk.Style()
        
        # Configure the Sash style for PanedWindow to make it more visible and easier to grab
        style.configure("TPanedwindow", background="lightgray")
        style.configure("TPanedwindow.Sash", background="gray", sashthickness=5, sashrelief=tk.RAISED)
        
        # Configure frame styles
        style.configure("TLabelframe", borderwidth=2)
        style.configure("TLabelframe.Label", font=("Arial", 10, "bold"))
        
        # Configure button styles
        style.configure("TButton", padding=5, font=("Arial", 9))
        
        # Configure entry widgets
        style.configure("TEntry", padding=2)
    
    def create_tabs(self):
        """Create the tab interface."""
        # Project Management tab
        project_tab = ttk.Frame(self.notebook)
        self.notebook.add(project_tab, text="Project Management")
        self.project_manager.create_ui(project_tab)
        
        # Time Periods tab
        periods_tab = ttk.Frame(self.notebook)
        self.notebook.add(periods_tab, text="Time Periods")
        self.period_manager.create_ui(periods_tab, self.project_manager)
        
        # Processing tab
        processing_tab = ttk.Frame(self.notebook)
        self.notebook.add(processing_tab, text="Processing")
        self.processor.create_ui(processing_tab, self.project_manager, self.period_manager)
        
        # Block Analysis tab
        analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(analysis_tab, text="Block Analysis")
        self.block_analyzer.create_ui(analysis_tab, self.project_manager)
        
        # Progress Visualizer tab
        progress_tab = ttk.Frame(self.notebook)
        self.notebook.add(progress_tab, text="Progress Map")
        self.progress_visualizer.create_ui(progress_tab, self.project_manager)
    
    def create_menu(self):
        """Create the application menu."""
        menu_bar = Menu(self.root)
        
        # File menu
        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Tools menu
        tools_menu = Menu(menu_bar, tearoff=0)
        tools_menu.add_command(label="CSV Analyzer", command=self.run_csv_analyzer)
        tools_menu.add_command(label="Generate Sample CSV", command=self.generate_sample_csv)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)
        
        # Help menu
        help_menu = Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)
    
    def on_tab_change(self, event):
        """Handle tab change events."""
        selected_tab = self.notebook.select()
        tab_text = self.notebook.tab(selected_tab, "text")
        
        if tab_text == "Time Periods":
            # Refresh participant list when switching to Time Periods tab
            self.period_manager.refresh_participants()
        elif tab_text == "Processing":
            # Refresh participant list when switching to Processing tab
            self.processor.refresh_participants()
        elif tab_text == "Block Analysis":
            # Refresh participant list when switching to Block Analysis tab
            self.block_analyzer.refresh_participants()
        elif tab_text == "Progress Map":
            # Refresh participant list when switching to Progress Map tab
            self.progress_visualizer.refresh_participants()
    
    def show_about(self):
        """Show information about the application."""
        about_window = tk.Toplevel(self.root)
        about_window.title("About")
        about_window.geometry("400x300")
        about_window.resizable(False, False)
        
        # Center the window
        about_window.geometry("+%d+%d" % (
            self.root.winfo_rootx() + (self.root.winfo_width() // 2) - 200,
            self.root.winfo_rooty() + (self.root.winfo_height() // 2) - 150
        ))
        
        # Add content
        frame = ttk.Frame(about_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Eye-Tracking Analysis System", font=("Arial", 16, "bold")).pack(pady=(0, 10))
        ttk.Label(frame, text="Version 1.0").pack()
        ttk.Label(frame, text="A modular system for analyzing eye-tracking data").pack(pady=(10, 20))
        
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Label(frame, text="© 2025 Eye-Tracking Research Team").pack(pady=(10, 0))
        
        ttk.Button(frame, text="Close", command=about_window.destroy).pack(pady=(20, 0))
    
    def run_csv_analyzer(self):
        """Run the CSV analyzer tool."""
        # First check if the user has Python installed
        try:
            subprocess.Popen(["python", "csv_analyzer.py"])
        except Exception as e:
            tk.messagebox.showerror("Error", f"Could not run CSV Analyzer: {str(e)}")
    
    def generate_sample_csv(self):
        """Generate a sample CSV file."""
        sample_window = tk.Toplevel(self.root)
        sample_window.title("Generate Sample CSV")
        sample_window.geometry("400x200")
        sample_window.resizable(False, False)
        
        # Center the window
        sample_window.geometry("+%d+%d" % (
            self.root.winfo_rootx() + (self.root.winfo_width() // 2) - 200,
            self.root.winfo_rooty() + (self.root.winfo_height() // 2) - 100
        ))
        
        # Add content
        frame = ttk.Frame(sample_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Generate Sample CSV File", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        # Format selection
        format_frame = ttk.Frame(frame)
        format_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(format_frame, text="Format:").pack(side=tk.LEFT, padx=(0, 10))
        
        format_var = tk.StringVar(value="standard")
        standard_rb = ttk.Radiobutton(format_frame, text="Standard", variable=format_var, value="standard")
        standard_rb.pack(side=tk.LEFT, padx=5)
        
        simple_rb = ttk.Radiobutton(format_frame, text="Simple", variable=format_var, value="simple")
        simple_rb.pack(side=tk.LEFT, padx=5)
        
        visionpro_rb = ttk.Radiobutton(format_frame, text="VisionPro", variable=format_var, value="visionpro")
        visionpro_rb.pack(side=tk.LEFT, padx=5)
        
        # Filename
        filename_frame = ttk.Frame(frame)
        filename_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(filename_frame, text="Filename:").pack(side=tk.LEFT, padx=(0, 10))
        
        filename_var = tk.StringVar(value="sample_eye_tracking.csv")
        ttk.Entry(filename_frame, textvariable=filename_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Generate button
        def do_generate():
            try:
                from utils import generate_sample_csv
                filepath = os.path.join(INPUT_DATA_DIR, filename_var.get())
                generate_sample_csv(filepath, format_var.get())
                tk.messagebox.showinfo("Success", f"Sample CSV generated: {filepath}")
                sample_window.destroy()
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to generate sample CSV: {str(e)}")
        
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=(10, 0))
        
        ttk.Button(button_frame, text="Generate", command=do_generate).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=sample_window.destroy).pack(side=tk.LEFT, padx=5)

def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = EyeTrackingAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()