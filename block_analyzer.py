#!/usr/bin/env python3
"""
Block Analyzer module for the eye-tracking analysis system.
Handles block-level analysis of eye-tracking data.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import utility functions
from utils import create_resizable_sidebar

class BlockAnalyzer:
    """Handles block-level analysis of eye-tracking data."""
    
    def __init__(self, processed_dir):
        """Initialize the block analyzer."""
        self.processed_dir = processed_dir
        self.blocks = {}  # Dictionary of block_id -> block data
        self.analysis_results = {}  # Analysis results cache
    
    def create_ui(self, parent, project_manager):
        """Create the block analysis UI."""
        self.parent = parent
        self.project_manager = project_manager
        
        # Create a frame with padding
        main_frame = ttk.Frame(parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Section title
        title_label = ttk.Label(main_frame, text="Block Analysis", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10), anchor=tk.W)
        
        # Create paned window for resizable panels
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel for block management with resizable sidebar
        left_container, left_content = create_resizable_sidebar(paned_window, min_width=350)
        
        # Left frame for block management
        left_frame = ttk.LabelFrame(left_content, text="Block Management")
        left_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create right panel for analysis with resizable sidebar
        right_container, right_content = create_resizable_sidebar(paned_window, min_width=350)
        
        # Right frame for analysis and visualization
        right_frame = ttk.LabelFrame(right_content, text="Analysis")
        right_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add panels to paned window
        paned_window.add(left_container, weight=1)
        paned_window.add(right_container, weight=1)
        
        # Block management section
        self.setup_block_management(left_frame)
        
        # Analysis section
        self.setup_analysis_section(right_frame)
    
    def setup_block_management(self, parent):
        """Set up the block management UI."""
        # Block list
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(list_frame, text="Defined Blocks:").pack(anchor=tk.W)
        
        # Create columns for the treeview
        columns = ("id", "name", "participants", "type")
        self.block_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=8)
        
        # Define headings
        self.block_tree.heading("id", text="ID")
        self.block_tree.heading("name", text="Name")
        self.block_tree.heading("participants", text="Participants")
        self.block_tree.heading("type", text="Type")
        
        # Set column widths
        self.block_tree.column("id", width=50)
        self.block_tree.column("name", width=150)
        self.block_tree.column("participants", width=150)
        self.block_tree.column("type", width=100)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.block_tree.yview)
        self.block_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.block_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.block_tree.bind("<<TreeviewSelect>>", self.on_block_selected)
        
        # Block creation section
        create_frame = ttk.LabelFrame(parent, text="Create New Block")
        create_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Block name
        name_frame = ttk.Frame(create_frame)
        name_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(name_frame, text="Block Name:").pack(side=tk.LEFT)
        self.block_name_var = tk.StringVar()
        ttk.Entry(name_frame, textvariable=self.block_name_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Block type
        type_frame = ttk.Frame(create_frame)
        type_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(type_frame, text="Block Type:").pack(side=tk.LEFT)
        self.block_type_var = tk.StringVar(value="participant_data")
        type_combo = ttk.Combobox(type_frame, textvariable=self.block_type_var, state="readonly")
        type_combo["values"] = ["participant_data", "period_data", "custom_data"]
        type_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Block source (participant or run)
        source_frame = ttk.Frame(create_frame)
        source_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(source_frame, text="Data Source:").pack(side=tk.LEFT)
        self.data_source_var = tk.StringVar(value="latest_run")
        source_combo = ttk.Combobox(source_frame, textvariable=self.data_source_var, state="readonly")
        source_combo["values"] = ["latest_run", "specific_run", "all_runs"]
        source_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Participant selection
        participant_frame = ttk.Frame(create_frame)
        participant_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(participant_frame, text="Participants:").pack(side=tk.LEFT)
        self.participant_listbox = tk.Listbox(participant_frame, selectmode=tk.MULTIPLE, height=4)
        self.participant_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        participant_scrollbar = ttk.Scrollbar(participant_frame, orient=tk.VERTICAL, command=self.participant_listbox.yview)
        participant_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.participant_listbox.config(yscrollcommand=participant_scrollbar.set)
        
        # Buttons
        button_frame = ttk.Frame(create_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Create Block", command=self.create_block).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Refresh Participants", command=self.refresh_participants).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete Selected Block", command=self.delete_block).pack(side=tk.LEFT, padx=5)
        
        # Load existing blocks
        self.load_blocks()
        
        # Initial refresh
        self.refresh_participants()
    
    def setup_analysis_section(self, parent):
        """Set up the analysis UI."""
        # Analysis options
        options_frame = ttk.Frame(parent)
        options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(options_frame, text="Analysis Type:").pack(side=tk.LEFT)
        self.analysis_type_var = tk.StringVar(value="descriptive")
        analysis_combo = ttk.Combobox(options_frame, textvariable=self.analysis_type_var, state="readonly", width=15)
        analysis_combo["values"] = ["descriptive", "comparative", "significance"]
        analysis_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(options_frame, text="Metric:").pack(side=tk.LEFT)
        self.metric_var = tk.StringVar(value="fixation_duration")
        metric_combo = ttk.Combobox(options_frame, textvariable=self.metric_var, state="readonly", width=15)
        metric_combo["values"] = ["fixation_duration", "fixation_count", "gaze_percentage", "text_coverage"]
        metric_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(options_frame, text="Run Analysis", command=self.run_analysis).pack(side=tk.LEFT, padx=5)
        
        # Results section
        results_frame = ttk.Frame(parent)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a frame for the visualization
        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=results_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Export frame
        export_frame = ttk.Frame(parent)
        export_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(export_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Save Figure", command=self.save_figure).pack(side=tk.LEFT, padx=5)
    
    def refresh_participants(self):
        """Refresh the participant list."""
        self.participant_listbox.delete(0, tk.END)
        
        participants = []
        if os.path.exists(self.processed_dir):
            participants = [d for d in os.listdir(self.processed_dir) 
                           if os.path.isdir(os.path.join(self.processed_dir, d)) and not d.startswith('.')]
        
        for participant in participants:
            self.participant_listbox.insert(tk.END, participant)
    
    def load_blocks(self):
        """Load existing blocks from config file."""
        config_path = os.path.join(self.processed_dir, "blocks.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.blocks = json.load(f)
                
                # Update the treeview
                self.update_block_list()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load blocks: {e}")
                self.blocks = {}
    
    def save_blocks(self):
        """Save blocks to config file."""
        config_path = os.path.join(self.processed_dir, "blocks.json")
        
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.blocks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save blocks: {e}")
    
    def update_block_list(self):
        """Update the block treeview."""
        # Clear existing items
        for item in self.block_tree.get_children():
            self.block_tree.delete(item)
        
        # Add blocks to the treeview
        for block_id, block in self.blocks.items():
            participants = ", ".join(block.get("participants", []))
            if len(participants) > 25:
                participants = participants[:22] + "..."
                
            self.block_tree.insert("", tk.END, values=(
                block_id,
                block.get("name", "Unnamed"),
                participants,
                block.get("type", "unknown")
            ))
    
    def create_block(self):
        """Create a new analysis block."""
        name = self.block_name_var.get().strip()
        block_type = self.block_type_var.get()
        data_source = self.data_source_var.get()
        
        if not name:
            messagebox.showerror("Error", "Block name is required")
            return
        
        # Get selected participants
        selected_indices = self.participant_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "No participants selected")
            return
            
        participants = [self.participant_listbox.get(i) for i in selected_indices]
        
        # Create a unique ID for the block
        block_id = f"block_{len(self.blocks) + 1}"
        
        # Create the block
        self.blocks[block_id] = {
            "id": block_id,
            "name": name,
            "type": block_type,
            "data_source": data_source,
            "participants": participants,
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        # Update the treeview
        self.update_block_list()
        
        # Save blocks
        self.save_blocks()
        
        # Clear input fields
        self.block_name_var.set("")
        
        messagebox.showinfo("Success", f"Block '{name}' created successfully")
    
    def delete_block(self):
        """Delete the selected block."""
        selected_items = self.block_tree.selection()
        if not selected_items:
            messagebox.showerror("Error", "No block selected")
            return
        
        if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete the selected block?"):
            for item in selected_items:
                values = self.block_tree.item(item, "values")
                block_id = values[0]
                
                # Remove from dictionary
                if block_id in self.blocks:
                    del self.blocks[block_id]
            
            # Update the treeview
            self.update_block_list()
            
            # Save blocks
            self.save_blocks()
            
            messagebox.showinfo("Success", "Block deleted successfully")
    
    def on_block_selected(self, event):
        """Handle block selection event."""
        selected_items = self.block_tree.selection()
        if not selected_items:
            return
            
        # Get the first selected item
        item = selected_items[0]
        values = self.block_tree.item(item, "values")
        block_id = values[0]
        
        # Load block data if not already in memory
        self.load_block_data(block_id)
    
    def load_block_data(self, block_id):
        """Load data for a specific block."""
        if block_id not in self.blocks:
            return
            
        block = self.blocks[block_id]
        
        # If data is already loaded, don't reload
        if "data" in block:
            return
            
        # Load data for each participant in the block
        all_data = []
        
        for participant in block["participants"]:
            participant_dir = os.path.join(self.processed_dir, participant)
            
            if not os.path.exists(participant_dir):
                continue
                
            # Find the appropriate run based on data_source
            if block["data_source"] == "latest_run":
                # Find the latest run directory
                run_dirs = [d for d in os.listdir(participant_dir) 
                          if os.path.isdir(os.path.join(participant_dir, d)) and d.startswith("run_")]
                
                if not run_dirs:
                    continue
                    
                # Sort by name (which includes timestamp)
                run_dirs.sort(reverse=True)
                run_dir = os.path.join(participant_dir, run_dirs[0])
                
                # Load data
                csv_path = os.path.join(run_dir, "mapped_gaze_to_text.csv")
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        df["participant"] = participant
                        all_data.append(df)
                    except Exception as e:
                        print(f"Error loading data for {participant}: {e}")
            
            elif block["data_source"] == "all_runs":
                # Find all run directories
                run_dirs = [d for d in os.listdir(participant_dir) 
                          if os.path.isdir(os.path.join(participant_dir, d)) and d.startswith("run_")]
                
                for run_dir_name in run_dirs:
                    run_dir = os.path.join(participant_dir, run_dir_name)
                    
                    # Load data
                    csv_path = os.path.join(run_dir, "mapped_gaze_to_text.csv")
                    if os.path.exists(csv_path):
                        try:
                            df = pd.read_csv(csv_path)
                            df["participant"] = participant
                            df["run"] = run_dir_name
                            all_data.append(df)
                        except Exception as e:
                            print(f"Error loading data for {participant}/{run_dir_name}: {e}")
        
        # Combine all data
        if all_data:
            block["data"] = pd.concat(all_data, ignore_index=True)
        else:
            block["data"] = pd.DataFrame()
            
        # Save updated block
        self.save_blocks()
    
    def run_analysis(self):
        """Run the selected analysis on the selected block."""
        selected_items = self.block_tree.selection()
        if not selected_items:
            messagebox.showerror("Error", "No block selected")
            return
            
        # Get the first selected item
        item = selected_items[0]
        values = self.block_tree.item(item, "values")
        block_id = values[0]
        
        # Make sure block data is loaded
        self.load_block_data(block_id)
        
        if block_id not in self.blocks or "data" not in self.blocks[block_id]:
            messagebox.showerror("Error", "Block data not available")
            return
            
        block = self.blocks[block_id]
        
        if block["data"].empty:
            messagebox.showerror("Error", "No data available for this block")
            return
        
        # Get analysis parameters
        analysis_type = self.analysis_type_var.get()
        metric = self.metric_var.get()
        
        # Run the appropriate analysis
        if analysis_type == "descriptive":
            self.run_descriptive_analysis(block, metric)
        elif analysis_type == "comparative":
            self.run_comparative_analysis(block, metric)
        elif analysis_type == "significance":
            self.run_significance_analysis(block, metric)
    
    def run_descriptive_analysis(self, block, metric):
        """Run descriptive statistics analysis."""
        data = block["data"]
        
        # Clear the figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if metric == "fixation_duration":
            # Group by participant and calculate mean fixation duration
            if "participant" in data.columns:
                results = data.groupby("participant")["timestamp"].count().reset_index()
                results.columns = ["participant", "count"]
                
                # Plot results
                ax.bar(results["participant"], results["count"])
                ax.set_xlabel("Participant")
                ax.set_ylabel("Fixation Count")
                ax.set_title(f"Fixation Count by Participant - {block['name']}")
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                self.figure.tight_layout()
                
        elif metric == "gaze_percentage":
            # Calculate percentage of gaze on different text categories
            if "category" in data.columns:
                results = data.groupby("category").size() / len(data) * 100
                
                # Plot results
                ax.pie(results, labels=results.index, autopct='%1.1f%%')
                ax.set_title(f"Gaze Distribution by Text Category - {block['name']}")
                
        # Update the canvas
        self.canvas.draw()
        
        # Cache the results
        self.analysis_results[f"{block['id']}_{analysis_type}_{metric}"] = {
            "type": "descriptive",
            "metric": metric,
            "block_name": block["name"],
            "timestamp": pd.Timestamp.now().isoformat(),
            "results": results.to_dict() if isinstance(results, pd.DataFrame) else results.to_dict()
        }
    
    def run_comparative_analysis(self, block, metric):
        """Run comparative analysis."""
        data = block["data"]
        
        # Clear the figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if metric == "fixation_duration" and "participant" in data.columns and "category" in data.columns:
            # Group by participant and category
            results = data.groupby(["participant", "category"]).size().unstack(fill_value=0)
            
            # Plot results
            results.plot(kind="bar", ax=ax)
            ax.set_xlabel("Participant")
            ax.set_ylabel("Fixation Count")
            ax.set_title(f"Fixation Count by Participant and Category - {block['name']}")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            self.figure.tight_layout()
        
        # Update the canvas
        self.canvas.draw()
    
    def run_significance_analysis(self, block, metric):
        """Run significance testing analysis."""
        data = block["data"]
        
        # For simplicity, just show a sample t-test if there are at least two participants
        if "participant" in data.columns and len(data["participant"].unique()) >= 2:
            participants = data["participant"].unique()
            if len(participants) < 2:
                messagebox.showinfo("Info", "Need at least two participants for significance testing")
                return
                
            # Get data for the first two participants
            p1 = participants[0]
            p2 = participants[1]
            
            # Clear the figure
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Plot distribution for each participant
            data[data["participant"] == p1]["timestamp"].plot.kde(ax=ax, label=p1)
            data[data["participant"] == p2]["timestamp"].plot.kde(ax=ax, label=p2)
            
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Density")
            ax.set_title(f"Timestamp Distribution Comparison - {block['name']}")
            ax.legend()
            
            # Update the canvas
            self.canvas.draw()
            
            # Display results
            messagebox.showinfo("Significance Analysis", 
                               f"Performed distribution comparison between {p1} and {p2}.\n"
                               "Statistical significance testing would be performed here.")
    
    def export_results(self):
        """Export analysis results."""
        # Check if there are any results to export
        if not self.analysis_results:
            messagebox.showinfo("Info", "No analysis results to export")
            return
            
        # Ask for save location
        save_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if not save_path:
            return
            
        # Save results
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, ensure_ascii=False, indent=4)
            messagebox.showinfo("Success", f"Results exported to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {e}")
    
    def save_figure(self):
        """Save the current figure."""
        # Check if there is a figure to save
        if not hasattr(self, 'figure'):
            messagebox.showinfo("Info", "No figure to save")
            return
            
        # Ask for save location
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")]
        )
        
        if not save_path:
            return
            
        # Save figure
        try:
            self.figure.savefig(save_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Figure saved to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save figure: {e}") 