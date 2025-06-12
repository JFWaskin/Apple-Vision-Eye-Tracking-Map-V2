#!/usr/bin/env python3
"""
Project Manager module for the eye-tracking analysis system.
Handles participant selection and project management.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import shutil

# Import utility functions
from utils import create_resizable_sidebar

class ProjectManager:
    """Manages participant selection and project settings."""
    
    def __init__(self, input_dir, processed_dir):
        """Initialize the project manager."""
        self.input_dir = input_dir
        self.processed_dir = processed_dir
        self.selected_participants = []
        self.config_file = os.path.join(processed_dir, "project_config.json")
        self.load_config()
    
    def create_ui(self, parent):
        """Create the project management UI."""
        self.parent = parent
        
        # Create a frame with padding
        main_frame = ttk.Frame(parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Section title
        title_label = ttk.Label(main_frame, text="Participant Management", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10), anchor=tk.W)
        
        # Create paned window for resizable panels
        paned_window = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Create upper section for participant selection with resizable sidebar
        upper_container, upper_content = create_resizable_sidebar(paned_window, min_width=0)
        
        # Participant selection area
        selection_frame = ttk.LabelFrame(upper_content, text="Select Participants")
        selection_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for the listboxes and buttons
        list_frame = ttk.Frame(selection_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Available participants
        avail_frame = ttk.Frame(list_frame)
        avail_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(avail_frame, text="Available Participants:").pack(anchor=tk.W)
        
        self.available_listbox = tk.Listbox(avail_frame, selectmode=tk.EXTENDED, height=15)
        self.available_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        avail_scrollbar = ttk.Scrollbar(avail_frame, orient=tk.VERTICAL, command=self.available_listbox.yview)
        avail_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.available_listbox.config(yscrollcommand=avail_scrollbar.set)
        
        # Buttons for moving participants
        btn_frame = ttk.Frame(list_frame)
        btn_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(btn_frame, text=">>", command=self.add_participants).pack(pady=5)
        ttk.Button(btn_frame, text="<<", command=self.remove_participants).pack(pady=5)
        
        # Selected participants
        sel_frame = ttk.Frame(list_frame)
        sel_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(sel_frame, text="Selected Participants:").pack(anchor=tk.W)
        
        self.selected_listbox = tk.Listbox(sel_frame, selectmode=tk.EXTENDED, height=15)
        self.selected_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sel_scrollbar = ttk.Scrollbar(sel_frame, orient=tk.VERTICAL, command=self.selected_listbox.yview)
        sel_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.selected_listbox.config(yscrollcommand=sel_scrollbar.set)
        
        # Buttons for actions
        action_frame = ttk.Frame(selection_frame)
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(action_frame, text="Refresh Participants", command=self.refresh_participants).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Save Selection", command=self.save_config).pack(side=tk.LEFT, padx=5)
        
        # Add upper container to paned window
        paned_window.add(upper_container, weight=3)
        
        # Create lower section for statistics with resizable sidebar
        lower_container, lower_content = create_resizable_sidebar(paned_window, min_width=0)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(lower_content, text="Project Statistics")
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        self.stats_text = tk.Text(stats_frame, height=5, width=80, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.stats_text.config(state=tk.DISABLED)
        
        # Add lower container to paned window
        paned_window.add(lower_container, weight=1)
        
        # Load initial data
        self.refresh_participants()
        self.update_statistics()
    
    def load_config(self):
        """Load project configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.selected_participants = config.get('selected_participants', [])
            except Exception as e:
                print(f"Error loading config: {e}")
                self.selected_participants = []
    
    def save_config(self):
        """Save project configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            config = {
                'selected_participants': self.selected_participants
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            messagebox.showinfo("Success", "Project configuration saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")
    
    def refresh_participants(self):
        """Refresh the list of available participants."""
        self.available_listbox.delete(0, tk.END)
        self.selected_listbox.delete(0, tk.END)
        
        # Get available participants
        available_participants = []
        if os.path.exists(self.input_dir):
            available_participants = [d for d in os.listdir(self.input_dir) 
                                     if os.path.isdir(os.path.join(self.input_dir, d)) and not d.startswith('.')]
        
        # Populate available list
        for participant in available_participants:
            if participant not in self.selected_participants:
                self.available_listbox.insert(tk.END, participant)
        
        # Populate selected list
        for participant in self.selected_participants:
            if participant in available_participants:
                self.selected_listbox.insert(tk.END, participant)
            else:
                # If participant no longer exists, remove from selected list
                self.selected_participants.remove(participant)
        
        self.update_statistics()
    
    def add_participants(self):
        """Add selected participants to the project."""
        selected_indices = self.available_listbox.curselection()
        if not selected_indices:
            return
        
        # Get selected participants
        to_add = [self.available_listbox.get(i) for i in selected_indices]
        
        # Add to selected list
        for participant in to_add:
            if participant not in self.selected_participants:
                self.selected_participants.append(participant)
        
        # Refresh lists
        self.refresh_participants()
    
    def remove_participants(self):
        """Remove participants from the project."""
        selected_indices = self.selected_listbox.curselection()
        if not selected_indices:
            return
        
        # Get selected participants
        to_remove = [self.selected_listbox.get(i) for i in selected_indices]
        
        # Remove from selected list
        for participant in to_remove:
            if participant in self.selected_participants:
                self.selected_participants.remove(participant)
        
        # Refresh lists
        self.refresh_participants()
    
    def update_statistics(self):
        """Update project statistics."""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        total_available = self.available_listbox.size()
        total_selected = self.selected_listbox.size()
        
        # Check processed data
        processed_participants = []
        if os.path.exists(self.processed_dir):
            processed_participants = [d for d in os.listdir(self.processed_dir) 
                                     if os.path.isdir(os.path.join(self.processed_dir, d))]
        
        # Count files in input directories
        input_files = 0
        for participant in self.selected_participants:
            participant_dir = os.path.join(self.input_dir, participant)
            if os.path.exists(participant_dir):
                for root, _, files in os.walk(participant_dir):
                    input_files += len(files)
        
        stats = (
            f"Total available participants: {total_available}\n"
            f"Selected participants: {total_selected}\n"
            f"Processed participants: {len(processed_participants)}\n"
            f"Total input files for selected participants: {input_files}\n"
        )
        
        self.stats_text.insert(tk.END, stats)
        self.stats_text.config(state=tk.DISABLED)
    
    def get_selected_participants(self):
        """Return the list of selected participants."""
        return self.selected_participants
    
    def get_participant_data_path(self, participant):
        """Get the data path for a specific participant."""
        return os.path.join(self.input_dir, participant)
    
    def get_participant_output_path(self, participant):
        """Get the output path for a specific participant."""
        path = os.path.join(self.processed_dir, participant)
        os.makedirs(path, exist_ok=True)
        return path 