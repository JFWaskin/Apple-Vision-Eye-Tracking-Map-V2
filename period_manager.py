#!/usr/bin/env python3
"""
Period Manager module for the eye-tracking analysis system.
Handles time period configuration for processing.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
import json
import re

# Import utility functions
from utils import create_resizable_sidebar

class PeriodManager:
    """Manages time periods for eye-tracking analysis."""
    
    def __init__(self):
        """Initialize the period manager."""
        self.periods = {}  # Dictionary of participant -> list of periods
        self.current_participant = None
    
    def create_ui(self, parent, project_manager):
        """Create the time periods UI."""
        self.parent = parent
        self.project_manager = project_manager
        
        # Create a frame with padding
        main_frame = ttk.Frame(parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Section title
        title_label = ttk.Label(main_frame, text="Time Period Management", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10), anchor=tk.W)
        
        # Create paned window for resizable panels
        paned_window = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Top section for participant selection and adding periods
        top_frame = ttk.Frame(paned_window)
        
        # Participant selection
        sel_frame = ttk.Frame(top_frame)
        sel_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(sel_frame, text="Select Participant:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.participant_var = tk.StringVar()
        self.participant_dropdown = ttk.Combobox(sel_frame, textvariable=self.participant_var, state="readonly")
        self.participant_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.participant_dropdown.bind("<<ComboboxSelected>>", self.load_participant_periods)
        
        ttk.Button(sel_frame, text="Refresh List", command=self.refresh_participants).pack(side=tk.LEFT, padx=5)
        
        # Period input section
        input_frame = ttk.LabelFrame(top_frame, text="Add Time Period")
        input_frame.pack(fill=tk.X, pady=10)
        
        entry_frame = ttk.Frame(input_frame)
        entry_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(entry_frame, text="Start Time (MM:SS):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.start_time_var = tk.StringVar()
        ttk.Entry(entry_frame, textvariable=self.start_time_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(entry_frame, text="End Time (MM:SS):").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.end_time_var = tk.StringVar()
        ttk.Entry(entry_frame, textvariable=self.end_time_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(entry_frame, text="Description:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.description_var = tk.StringVar()
        ttk.Entry(entry_frame, textvariable=self.description_var, width=40).grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky=tk.EW)
        
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(button_frame, text="Add Period", command=self.add_period).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Fields", command=self.clear_fields).pack(side=tk.LEFT, padx=5)
        
        # Add top frame to paned window
        paned_window.add(top_frame, weight=1)
        
        # Bottom section for period list using the resizable sidebar
        bottom_container, bottom_content = create_resizable_sidebar(paned_window, min_width=0)
        
        # Period list section
        list_frame = ttk.LabelFrame(bottom_content, text="Time Periods")
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for the listbox and scrollbar
        period_list_frame = ttk.Frame(list_frame)
        period_list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create columns
        columns = ("start", "end", "description")
        self.period_tree = ttk.Treeview(period_list_frame, columns=columns, show="headings")
        
        # Define headings
        self.period_tree.heading("start", text="Start Time")
        self.period_tree.heading("end", text="End Time")
        self.period_tree.heading("description", text="Description")
        
        # Set column widths
        self.period_tree.column("start", width=100)
        self.period_tree.column("end", width=100)
        self.period_tree.column("description", width=300)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(period_list_frame, orient=tk.VERTICAL, command=self.period_tree.yview)
        self.period_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.period_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons for actions
        action_frame = ttk.Frame(list_frame)
        action_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(action_frame, text="Remove Selected", command=self.remove_period).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Save Periods", command=self.save_periods).pack(side=tk.LEFT, padx=5)
        
        # Add bottom container to paned window
        paned_window.add(bottom_container, weight=2)
        
        # Initial refresh
        self.refresh_participants()
    
    def refresh_participants(self):
        """Refresh the participant dropdown."""
        participants = self.project_manager.get_selected_participants()
        
        self.participant_dropdown["values"] = participants
        
        if participants and (not self.participant_var.get() or self.participant_var.get() not in participants):
            self.participant_var.set(participants[0])
            self.load_participant_periods()
        elif not participants:
            self.participant_var.set("")
            self.current_participant = None
            self.clear_period_list()
    
    def load_participant_periods(self, event=None):
        """Load periods for the selected participant."""
        participant = self.participant_var.get()
        if not participant:
            return
        
        self.current_participant = participant
        self.clear_period_list()
        
        # Load periods from config file
        config_path = os.path.join(self.project_manager.get_participant_output_path(participant), "periods.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if participant in self.periods:
                        self.periods[participant] = data.get("periods", [])
                    else:
                        self.periods[participant] = data.get("periods", [])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load periods: {e}")
                self.periods[participant] = []
        else:
            self.periods[participant] = []
        
        # Populate the treeview
        for period in self.periods[participant]:
            self.period_tree.insert("", tk.END, values=(
                period["start_time"],
                period["end_time"],
                period["description"]
            ))
    
    def clear_period_list(self):
        """Clear the period list."""
        for item in self.period_tree.get_children():
            self.period_tree.delete(item)
    
    def validate_time_format(self, time_str):
        """Validate that time is in MM:SS format."""
        if not time_str:
            return False
            
        pattern = r'^\d{1,2}:\d{2}$'
        if not re.match(pattern, time_str):
            return False
            
        try:
            minutes, seconds = map(int, time_str.split(':'))
            if seconds >= 60:
                return False
            return True
        except ValueError:
            return False
    
    def add_period(self):
        """Add a new time period."""
        if not self.current_participant:
            messagebox.showerror("Error", "No participant selected")
            return
        
        start_time = self.start_time_var.get().strip()
        end_time = self.end_time_var.get().strip()
        description = self.description_var.get().strip()
        
        # Validate inputs
        if not start_time or not end_time:
            messagebox.showerror("Error", "Start and end times are required")
            return
            
        if not self.validate_time_format(start_time) or not self.validate_time_format(end_time):
            messagebox.showerror("Error", "Times must be in MM:SS format")
            return
            
        # Compare start and end times
        start_minutes, start_seconds = map(int, start_time.split(':'))
        end_minutes, end_seconds = map(int, end_time.split(':'))
        
        start_total_seconds = start_minutes * 60 + start_seconds
        end_total_seconds = end_minutes * 60 + end_seconds
        
        if start_total_seconds >= end_total_seconds:
            messagebox.showerror("Error", "End time must be after start time")
            return
        
        # Create period object
        period = {
            "start_time": start_time,
            "end_time": end_time,
            "description": description,
            "start_seconds": start_total_seconds,
            "end_seconds": end_total_seconds
        }
        
        # Add to list and treeview
        if self.current_participant not in self.periods:
            self.periods[self.current_participant] = []
            
        self.periods[self.current_participant].append(period)
        self.period_tree.insert("", tk.END, values=(start_time, end_time, description))
        
        # Clear input fields
        self.clear_fields()
    
    def clear_fields(self):
        """Clear input fields."""
        self.start_time_var.set("")
        self.end_time_var.set("")
        self.description_var.set("")
    
    def remove_period(self):
        """Remove selected period."""
        selected_items = self.period_tree.selection()
        if not selected_items:
            messagebox.showerror("Error", "No period selected")
            return
            
        if self.current_participant not in self.periods:
            return
            
        for item in selected_items:
            values = self.period_tree.item(item, "values")
            start_time = values[0]
            end_time = values[1]
            
            # Find and remove the period from the list
            for i, period in enumerate(self.periods[self.current_participant]):
                if period["start_time"] == start_time and period["end_time"] == end_time:
                    self.periods[self.current_participant].pop(i)
                    break
                    
            self.period_tree.delete(item)
    
    def save_periods(self):
        """Save periods to config file."""
        if not self.current_participant:
            messagebox.showerror("Error", "No participant selected")
            return
            
        if self.current_participant not in self.periods:
            self.periods[self.current_participant] = []
        
        # Save to config file
        output_path = self.project_manager.get_participant_output_path(self.current_participant)
        config_path = os.path.join(output_path, "periods.json")
        
        try:
            os.makedirs(output_path, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump({"periods": self.periods[self.current_participant]}, f, ensure_ascii=False, indent=4)
            messagebox.showinfo("Success", f"Saved {len(self.periods[self.current_participant])} periods for {self.current_participant}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save periods: {e}")
    
    def get_periods(self, participant):
        """Get periods for a participant."""
        if participant not in self.periods:
            # Try to load from file
            config_path = os.path.join(self.project_manager.get_participant_output_path(participant), "periods.json")
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.periods[participant] = data.get("periods", [])
                except Exception:
                    self.periods[participant] = []
            else:
                self.periods[participant] = []
                
        return self.periods[participant] 