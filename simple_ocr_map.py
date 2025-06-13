#!/usr/bin/env python3
"""
Optimized eye-tracking to text mapping using Apple Vision Framework with Metal acceleration.
This is the integrated version combining the best features from previous implementations.
"""

import argparse
import logging
import os
import sys
import tempfile
import uuid
import time
import traceback
import threading
import queue
import re
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Try importing jieba for Chinese word segmentation
try:
    import jieba
    JIEBA_AVAILABLE = True
    print("Jieba Chinese word segmentation library is available")
except ImportError:
    JIEBA_AVAILABLE = False
    print("Jieba library not found. Using basic Chinese segmentation.")

# Configure for GPU acceleration and memory management
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable MPS fallback for Metal
os.environ['OPENCV_OPENCL_RUNTIME'] = 'disabled'  # Let Metal handle GPU tasks
os.environ['VN_METAL_DEBUG'] = '1'  # Enable Metal debugging for Vision Framework
os.environ['VN_MAX_CONCURRENT_OPERATIONS'] = '32'  # Maximize concurrent Vision operations
os.environ['LIBDISPATCH_COOPERATIVE_POOL_STRICT'] = '0'  # Allow cooperative thread pooling

# Global constants 
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO_PATH = os.path.join(WORKSPACE_ROOT, "沈若枢/User1_241227135925/User1_241227135925.mp4")
DEFAULT_CSV_PATH = os.path.join(WORKSPACE_ROOT, "沈若枢/User1_241227135925/User1_241227135925_raw.csv")
DEFAULT_FRAME_INTERVAL = 5
DEFAULT_CONFIDENCE_THRESHOLD = 60
DEFAULT_DISTANCE_THRESHOLD = 10
DEFAULT_OUTPUT_FILE = "mapped_gaze_to_text.csv"
SEPARATOR_OFFSET = 0.05  # Vertical offset for separating OCR text into source/target
CHINESE_CHAR_RANGE = ('\u4e00', '\u9fff')  # Unicode range for Chinese characters

# Performance optimization parameters - further tuned for maximum performance on M2 Max
MAX_THREADS = 12  # Reduced threads to prevent memory issues
MAPPING_BATCH_SIZE = 16000  # Reduced mapping batch size for better memory management
OCR_BATCH_SIZE = 64  # Reduced OCR batch size to avoid memory pressure
BATCH_REPORT_INTERVAL = 0.05  # Frequent progress updates remain unchanged
USE_IN_MEMORY_PROCESSING = True  # In-memory processing enabled for speed
GPU_MEMORY_LIMIT_MB = 2048  # Reduced memory limit to control GPU batch parallelism
# Performance enhancement - adaptive batch sizing based on real-time performance
ENABLE_ADAPTIVE_BATCH_SIZING = True  # Dynamically adjust batch sizes based on processing speed
ADAPTIVE_BATCH_MIN = 32   # Reduced minimum batch size for adaptive sizing
ADAPTIVE_BATCH_MAX = 128  # Reduced maximum batch size for adaptive sizing
USE_FAST_MODE = False     # Toggle between accurate and fast recognition mode
PREPROCESS_FRAMES = True  # Enable frame preprocessing to improve OCR recognition speed
USE_FRAME_CACHE = True    # Enable frame caching to avoid redundant OCR operations
FRAME_CACHE_SIZE = 1000   # Maximum number of frames to cache
PARALLEL_IMAGE_PROCESSING = True  # Enable parallel image processing pipeline

# Additional configuration
WORD_LEVEL_RECOGNITION = True  # Flag to enable word-level recognition
ENGLISH_WORD_MIN_LENGTH = 2  # Minimum length for English words
CHINESE_WORD_MIN_LENGTH = 1  # Minimum length for Chinese words

# Performance tracking
perf_stats = {
    "ocr_time": 0,
    "frame_processing_time": 0,
    "total_frames": 0,
    "total_ocr_calls": 0,
    "gpu_time": 0,
    "gpu_memory_peak": 0,
    "batch_processing_times": [],  # New tracker for adaptive batch sizing
    "last_batch_size": OCR_BATCH_SIZE,  # Track the last batch size used
    "last_processing_speed": 0,  # Track the last processing speed
    "cache_hits": 0,  # Count cache hits
    "cache_misses": 0,  # Count cache misses
    "preprocessing_time": 0  # Track preprocessing time
}

# Frame cache for avoiding redundant OCR (simple LRU cache)
frame_cache = {}  # Maps frame_num_hash -> OCR results
frame_cache_keys = []  # For LRU tracking
frame_preprocessing_cache = {}  # Maps original_hash -> processed_hash
frame_cache_lock = threading.Lock()

# Add an image preprocessing queue for parallel processing
image_preprocessing_queue = queue.Queue(maxsize=1000)
preprocessing_threads = []
preprocessing_stop_event = threading.Event()

# Check for Metal and Vision Framework support
try:
    from Foundation import NSURL, NSMutableDictionary
    import Vision
    import Metal
    import CoreML
    
    # Try to import CoreGraphics-related modules
    try:
        from CoreFoundation import CFDataCreate
        from CoreGraphics import CGColorSpaceCreateDeviceRGB, CGDataProviderCreateWithData, CGImageCreate
        HAS_CORE_GRAPHICS = True
        print("CoreGraphics modules are available")
    except ImportError:
        HAS_CORE_GRAPHICS = False
        print("CoreGraphics modules not available, will use file-based processing")
    
    # Check for Metal support
    devices = Metal.MTLCopyAllDevices()
    HAS_METAL = len(devices) > 0
    if HAS_METAL:
        device = Metal.MTLCreateSystemDefaultDevice()
        print(f"Metal GPU available: {device.name()}")
    else:
        print("No Metal GPU found")
    
    # Check Vision Framework
    VISION_AVAILABLE = True
    print("Apple Vision Framework is available")
except ImportError as e:
    VISION_AVAILABLE = False
    HAS_METAL = False
    HAS_CORE_GRAPHICS = False
    print(f"Apple Vision Framework is not available: {e}")

# Update in-memory setting based on module availability
USE_IN_MEMORY_PROCESSING = USE_IN_MEMORY_PROCESSING and HAS_CORE_GRAPHICS

# Configure simple logging directly to console for real-time feedback
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Progress tracking
progress_lock = threading.Lock()
total_progress = 0

# Global Vision Framework variables for reuse
global_text_request = None
global_vision_initialized = False

def initialize_vision_framework():
    """Initialize the Vision Framework once and store it for reuse."""
    global global_text_request, global_vision_initialized
    
    if global_vision_initialized:
        return global_text_request
        
    if not VISION_AVAILABLE:
        return None
    
    try:
        # Create text recognition request optimized for Metal performance    
        text_request = Vision.VNRecognizeTextRequest.alloc().init()
        
        # Force Metal GPU usage with environment variable
        os.environ['VN_FORCE_METAL'] = '1'
        
        # Use accurate recognition for better results, but faster mode for performance
        if hasattr(Vision, 'VNRequestTextRecognitionLevelAccurate'):
            # Choose between accurate and fast modes based on configuration
            recognition_level = Vision.VNRequestTextRecognitionLevelFast if USE_FAST_MODE else Vision.VNRequestTextRecognitionLevelAccurate
            text_request.setRecognitionLevel_(recognition_level)
            print(f"Using {'fast' if USE_FAST_MODE else 'accurate'} recognition mode")
        
        # Configure for word-level recognition
        if WORD_LEVEL_RECOGNITION and hasattr(text_request, 'setUsesLanguageCorrection_'):
            text_request.setUsesLanguageCorrection_(False)  # Disable language correction for raw word recognition
        
        # Add language recognition support for English and Chinese
        text_request.setRecognitionLanguages_(["zh-Hans", "zh-Hant", "en-US"])
        
        # Set minimum text height for word-level OCR
        if hasattr(text_request, 'setMinimumTextHeight:'):
            text_request.setMinimumTextHeight_(0.01)  # 1% of image height
        
        # Explicitly request GPU acceleration
        if HAS_METAL:
            # Create a dictionary with GPU preference
            options = NSMutableDictionary.dictionary()
            options.setValue_forKey_(True, "VNUseMetalAcceleration")
            options.setValue_forKey_(True, "VNUseGPU")
            options.setValue_forKey_(True, "VNPreferBackgroundProcessing")
            
            # Try to pass options directly to the request
            if hasattr(text_request, 'setOptions_'):
                text_request.setOptions_(options)
                
            # Set the request to use GPU
            text_request.setUsesCPUOnly_(False)
            
            # Use latest revision that supports Metal
            if hasattr(Vision, 'VNRecognizeTextRequestRevision2'):
                text_request.setRevision_(Vision.VNRecognizeTextRequestRevision2)
            elif hasattr(Vision, 'VNRecognizeTextRequestRevision3'):
                text_request.setRevision_(Vision.VNRecognizeTextRequestRevision3)
            else:
                text_request.setRevision_(2)  # Fallback to revision 2
            
            print("Initialized Vision request with Metal GPU acceleration (once for the entire session)")
        else:
            print("Metal GPU not available, using CPU processing")
        
        global_text_request = text_request
        global_vision_initialized = True
        return global_text_request
    
    except Exception as e:
        print(f"Error initializing Vision Framework: {e}")
        traceback.print_exc()
        return None

def load_eye_tracking_data(csv_path, encoding="utf-8"):
    """Load eye tracking data from CSV file with optimized memory usage."""
    print(f"Loading eye-tracking data from {csv_path}...")
    
    if not os.path.isfile(csv_path):
        print(f"Error: CSV not found: {csv_path}")
        return None
        
    try:
        # First, read the header to check available columns
        with open(csv_path, 'r', encoding=encoding) as f:
            header = f.readline().strip().split(',')
            print(f"CSV columns: {header}")
        
        # Define column mapping for different CSV formats
        standard_columns = [
            'timestampUs', 
            'leftGaze.gazePointValid', 'rightGaze.gazePointValid',
            'leftGaze.gazePoint.x', 'leftGaze.gazePoint.y',
            'rightGaze.gazePoint.x', 'rightGaze.gazePoint.y'
        ]
        
        # Alternative column names that might be in the file
        alt_column_mapping = {
            'timestampUs': ['timestamp', 'time', 'timestampus', 'timestamp_us'],
            'leftGaze.gazePointValid': ['leftgazevalid', 'left_valid', 'leftvalid'],
            'rightGaze.gazePointValid': ['rightgazevalid', 'right_valid', 'rightvalid'],
            'leftGaze.gazePoint.x': ['leftgazex', 'left_x', 'leftx', 'left_gaze_x'],
            'leftGaze.gazePoint.y': ['leftgazey', 'left_y', 'lefty', 'left_gaze_y'],
            'rightGaze.gazePoint.x': ['rightgazex', 'right_x', 'rightx', 'right_gaze_x'],
            'rightGaze.gazePoint.y': ['rightgazey', 'right_y', 'righty', 'right_gaze_y']
        }
        
        # Check if we need to use alternative column names
        column_map = {}
        missing_columns = []
        for std_col in standard_columns:
            if std_col in header:
                column_map[std_col] = std_col
            else:
                # Try to find alternative column name
                found = False
                if std_col in alt_column_mapping:
                    for alt_col in alt_column_mapping[std_col]:
                        if alt_col in header or alt_col.lower() in [h.lower() for h in header]:
                            # Find the case-insensitive match in the header
                            for h in header:
                                if h.lower() == alt_col.lower():
                                    column_map[std_col] = h
                                    found = True
                                    break
                            if found:
                                break
                
                if not found:
                    missing_columns.append(std_col)
        
        # Print column mapping for debugging
        print(f"Column mapping: {column_map}")
        
        if missing_columns:
            # Check if we have simple x,y coordinates as fallback
            if 'x' in header and 'y' in header and 'timestamp' in header:
                print("Using simple x,y coordinate format")
                # Read CSV with simple format
                df = pd.read_csv(csv_path, encoding=encoding)
                
                # Create needed columns
                df['avg_gaze_x'] = df['x']
                df['avg_gaze_y'] = df['y']
                df['timestampUs'] = df['timestamp']
                
                return df
            else:
                print(f"Missing columns: {missing_columns}")
                print("Available columns in CSV:")
                for col in header:
                    print(f"  - {col}")
                return None
        
        # Read in chunks with the mapped columns
        usecols = list(column_map.values())
        chunks = []
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=100000, encoding=encoding):
            # Rename columns to standard names
            inv_map = {v: k for k, v in column_map.items()}
            chunk = chunk.rename(columns=inv_map)
            
            # Filter for valid gaze points
            try:
                valid_chunk = chunk[
                    ((chunk['leftGaze.gazePointValid'] == 1) | (chunk['leftGaze.gazePointValid'] == True)) &
                    ((chunk['rightGaze.gazePointValid'] == 1) | (chunk['rightGaze.gazePointValid'] == True)) &
                    (chunk['timestampUs'].notna()) &
                    (chunk['leftGaze.gazePoint.x'].notna()) &
                    (chunk['leftGaze.gazePoint.y'].notna()) &
                    (chunk['rightGaze.gazePoint.x'].notna()) &
                    (chunk['rightGaze.gazePoint.y'].notna())
                ].copy()
                
                # Calculate average gaze position
                valid_chunk['avg_gaze_x'] = (valid_chunk['leftGaze.gazePoint.x'] + valid_chunk['rightGaze.gazePoint.x']) / 2
                valid_chunk['avg_gaze_y'] = (valid_chunk['leftGaze.gazePoint.y'] + valid_chunk['rightGaze.gazePoint.y']) / 2
                
                chunks.append(valid_chunk)
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
        
        valid_df = pd.concat(chunks) if chunks else pd.DataFrame()
        
        # Free memory
        del chunks
        
        total_rows = sum(1 for _ in open(csv_path)) - 1  # Subtract header
        print(f"Loaded {total_rows} rows from CSV")
        print(f"Kept {len(valid_df)} valid gaze points")
        return valid_df
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        traceback.print_exc()
        return None

def map_gaze_to_frames(gaze_data, fps):
    """Map gaze timestamps to video frame numbers using vectorized operations."""
    print(f"Mapping gaze points to frames (FPS: {fps})...")
    
    # Calculate frame number for each gaze point
    microseconds_per_frame = 1000000 / fps
    gaze_data['frame_num'] = (gaze_data['timestampUs'] / microseconds_per_frame).astype(int)
    
    return gaze_data

def scale_gaze_to_video(gaze_data, video_width, video_height):
    """Scale normalized gaze coordinates to video pixel coordinates."""
    print(f"Scaling gaze coordinates to {video_width}x{video_height}...")
    
    gaze_data['scaled_x'] = gaze_data['avg_gaze_x'] * video_width
    gaze_data['scaled_y'] = gaze_data['avg_gaze_y'] * video_height
    
    # Clip to valid range
    gaze_data['scaled_x'] = gaze_data['scaled_x'].clip(0, video_width - 1)
    gaze_data['scaled_y'] = gaze_data['scaled_y'].clip(0, video_height - 1)
    
    return gaze_data

def configure_vision_request_for_metal():
    """Configure Vision request to use Metal acceleration."""
    # Use the global initialization function instead of creating a new request
    return initialize_vision_framework()

def is_chinese_char(char):
    """Check if a character is Chinese."""
    return CHINESE_CHAR_RANGE[0] <= char <= CHINESE_CHAR_RANGE[1]

def contains_chinese(text):
    """Check if text contains any Chinese characters."""
    if not isinstance(text, str):
        return False
    return any(is_chinese_char(char) for char in text)

def segment_chinese_text(text):
    """
    Segment Chinese text into words using jieba if available,
    otherwise fall back to a simple segmentation approach.
    """
    if not text:
        return []
        
    # Use jieba for accurate Chinese word segmentation if available
    if JIEBA_AVAILABLE:
        try:
            # Cut text into words using jieba
            words = list(jieba.cut(text))
            # Filter out punctuation and whitespace
            words = [word for word in words if word.strip() and not all(char in '，。！？；：""''（）【】、…—《》' for char in word)]
            return words
        except Exception as e:
            print(f"Error using jieba for segmentation: {e}. Falling back to basic segmentation.")
    
    # Basic segmentation as fallback
    words = []
    current_word = ""
    current_is_chinese = None
    
    for char in text:
        is_chinese_current = is_chinese_char(char)
        
        # Skip spaces and punctuation
        if char.isspace() or (not char.isalnum() and not is_chinese_current):
            if current_word:
                words.append(current_word)
                current_word = ""
            current_is_chinese = None
            continue
        
        # Handle type change (Chinese to non-Chinese or vice versa)
        if current_is_chinese is not None and current_is_chinese != is_chinese_current:
            if current_word:
                words.append(current_word)
                current_word = ""
        
        current_word += char
        current_is_chinese = is_chinese_current
    
    # Add the last word if there is one
    if current_word:
        words.append(current_word)
    
    return words

def segment_english_text(text):
    """Segment English text into words."""
    # Remove punctuation and split by spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    words = [word for word in text.split() if len(word) >= ENGLISH_WORD_MIN_LENGTH]
    return words

def split_text_into_words(text):
    """Split text into individual words, handling both English and Chinese text."""
    if not text or not isinstance(text, str):
        return []
    
    # Check if text contains Chinese characters
    if contains_chinese(text):
        return segment_chinese_text(text)
    else:
        return segment_english_text(text)

def process_recognized_text_into_words(observations, frame_num, frame_shape, min_confidence=0):
    """Process text observations from Vision Framework into word-level results."""
    h, w, _ = frame_shape
    results = []
    
    for observation in observations:
        # Get recognized text and confidence
        text = observation.text()
        confidence = observation.confidence() * 100  # Convert to percentage
        
        if confidence < min_confidence:
            continue
        
        # Get bounding box for the overall text
        bbox = observation.boundingBox()
        left = bbox.origin.x * w
        top = bbox.origin.y * h
        width = bbox.size.width * w
        height = bbox.size.height * h
        
        # Try to get word-level observations if available
        word_level_observations = []
        
        # In newer versions of Vision Framework, we can get direct access to word boxes
        if hasattr(observation, 'textObservations'):
            word_level_observations = observation.textObservations()
        
        if word_level_observations:
            # Process each word observation
            for word_obs in word_level_observations:
                word_text = word_obs.text()
                word_conf = word_obs.confidence() * 100
                
                # Get word bounding box
                word_bbox = word_obs.boundingBox()
                word_left = word_bbox.origin.x * w
                word_top = word_bbox.origin.y * h
                word_width = word_bbox.size.width * w
                word_height = word_bbox.size.height * h
                
                results.append({
                    'frame_num': frame_num,
                    'text': word_text,
                    'conf': word_conf,
                    'left': word_left,
                    'top': word_top,
                    'width': word_width,
                    'height': word_height,
                    'center_x': word_left + (word_width / 2),
                    'center_y': word_top + (word_height / 2)
                })
        else:
            # When word-level bounding boxes are not available, we split the text and estimate positions
            words = split_text_into_words(text)
            
            if not words:
                continue
                
            # Calculate approximate word positions
            total_length = sum(len(word) for word in words)
            current_position = 0
            
            for word in words:
                # Skip very short words
                word_len = len(word)
                if (contains_chinese(word) and word_len < CHINESE_WORD_MIN_LENGTH) or \
                   (not contains_chinese(word) and word_len < ENGLISH_WORD_MIN_LENGTH):
                    continue
                
                # Estimate position based on character count
                word_ratio = word_len / total_length
                word_width = width * word_ratio
                
                # Calculate horizontal position
                position_ratio = current_position / total_length
                word_left = left + (width * position_ratio)
                
                # Update position for next word
                current_position += word_len
                
                results.append({
                    'frame_num': frame_num,
                    'text': word,
                    'conf': confidence,  # Use same confidence as parent observation
                    'left': word_left,
                    'top': top,
                    'width': word_width,
                    'height': height,
                    'center_x': word_left + (word_width / 2),
                    'center_y': top + (height / 2)
                })
    
    return results

def check_gpu_memory():
    """Check GPU memory usage and return True if it's safe to continue processing."""
    if not HAS_METAL:
        return True
        
    try:
        device = Metal.MTLCreateSystemDefaultDevice()
        if hasattr(device, 'currentAllocatedSize'):
            current_memory_mb = device.currentAllocatedSize() / (1024*1024)
            # Update peak memory tracking
            perf_stats["gpu_memory_peak"] = max(perf_stats["gpu_memory_peak"], current_memory_mb)
            # Return True if we're under the limit, False if we're over
            return current_memory_mb < GPU_MEMORY_LIMIT_MB
    except Exception as e:
        print(f"Error checking GPU memory: {e}")
    
    # If we can't check memory or there was an error, assume it's safe
    return True

def clear_gpu_memory():
    """Enhanced function to clear GPU memory by forcing garbage collection and releasing Metal resources."""
    import gc
    
    # Force garbage collection
    gc.collect()
    
    # On macOS, we can try to force release Metal resources
    if HAS_METAL:
        try:
            # Force a synchronization point
            device = Metal.MTLCreateSystemDefaultDevice()
            if hasattr(device, 'newCommandQueue'):
                command_queue = device.newCommandQueue()
                if command_queue:
                    command_buffer = command_queue.commandBuffer()
                    if command_buffer:
                        command_buffer.commit()
                        command_buffer.waitUntilCompleted()
                        
                    # Explicitly release the command buffer and queue
                    del command_buffer
                    del command_queue
                    
            # Check current memory usage and log
            if hasattr(device, 'currentAllocatedSize'):
                current_memory_mb = device.currentAllocatedSize() / (1024*1024)
                print(f"GPU memory after clearing: {current_memory_mb:.2f} MB")
                
            # Force autoreleasepool to empty
            try:
                from Foundation import NSAutoreleasePool
                pool = NSAutoreleasePool.alloc().init()
                del pool
            except:
                pass
                
        except Exception as e:
            print(f"Error clearing GPU memory: {e}")
            
    # Run garbage collection again
    gc.collect()

def get_cached_result(frame_num, frame_hash):
    """Try to get cached OCR result for a frame."""
    if not USE_FRAME_CACHE:
        return None
        
    cache_key = f"{frame_num}_{frame_hash}"
    with frame_cache_lock:
        if cache_key in frame_cache:
            # Update cache order (move to end = most recently used)
            frame_cache_keys.remove(cache_key)
            frame_cache_keys.append(cache_key)
            
            perf_stats["cache_hits"] += 1
            return frame_cache[cache_key]
    
    perf_stats["cache_misses"] += 1
    return None

def cache_frame_result(frame_num, frame_hash, results):
    """Cache OCR results for a frame."""
    if not USE_FRAME_CACHE:
        return
        
    cache_key = f"{frame_num}_{frame_hash}"
    with frame_cache_lock:
        # If we're at capacity, remove oldest item
        if len(frame_cache_keys) >= FRAME_CACHE_SIZE:
            oldest_key = frame_cache_keys.pop(0)
            if oldest_key in frame_cache:
                del frame_cache[oldest_key]
        
        # Add new item
        frame_cache[cache_key] = results
        frame_cache_keys.append(cache_key)

def compute_frame_hash(frame):
    """Compute a hash of frame content for caching purposes."""
    if not USE_FRAME_CACHE:
        return ""
        
    try:
        # Downscale the frame to compute a faster hash
        small_frame = cv2.resize(frame, (32, 32))
        # Convert to grayscale for more stable hashing
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        # Compute mean hash
        avg = gray.mean()
        # Convert to binary hash
        hash_value = 0
        for i in range(32):
            for j in range(32):
                if gray[i, j] > avg:
                    hash_value += 1 << (i * 32 + j)
                    
        # Return a string representation of the hash
        return str(hash_value % 10000000007)  # Use a prime number to reduce collisions
    except Exception as e:
        print(f"Error computing frame hash: {e}")
        return str(time.time())  # Fallback to timestamp as hash

def image_preprocessing_worker():
    """Worker function for parallel frame preprocessing."""
    while not preprocessing_stop_event.is_set():
        try:
            # Get a frame from the queue with a timeout
            item = image_preprocessing_queue.get(timeout=0.5)
            if item is None:
                image_preprocessing_queue.task_done()
                continue
                
            frame_num, frame, frame_hash = item
            
            # Check if already in cache
            cached_result = get_cached_result(frame_num, frame_hash)
            if cached_result is not None:
                # Already in cache, nothing to do
                image_preprocessing_queue.task_done()
                continue
                
            # Preprocess the frame
            preproc_start = time.time()
            processed_frame = preprocess_frame_for_ocr(frame)
            preproc_time = time.time() - preproc_start
            
            with progress_lock:
                perf_stats["preprocessing_time"] += preproc_time
            
            # Compute the hash for the processed frame
            processed_hash = compute_frame_hash(processed_frame)
            
            # Store the original frame hash to processed frame hash mapping
            # This will allow the main thread to find the preprocessed result
            with frame_cache_lock:
                if frame_hash not in frame_preprocessing_cache:
                    frame_preprocessing_cache[frame_hash] = processed_hash
                
            # Mark task as done
            image_preprocessing_queue.task_done()
        except queue.Empty:
            # Queue timeout, just continue
            continue
        except Exception as e:
            print(f"Error in preprocessing worker: {e}")
            traceback.print_exc()
            # Ensure we mark the task as done even on error
            try:
                image_preprocessing_queue.task_done()
            except:
                pass

def start_preprocessing_workers(num_workers=4):
    """Start worker threads for parallel frame preprocessing."""
    if not PARALLEL_IMAGE_PROCESSING:
        return
        
    global preprocessing_threads
    preprocessing_stop_event.clear()
    preprocessing_threads = []
    
    for _ in range(num_workers):
        t = threading.Thread(target=image_preprocessing_worker)
        t.daemon = True
        t.start()
        preprocessing_threads.append(t)
        
    print(f"Started {num_workers} preprocessing worker threads")

def stop_preprocessing_workers():
    """Stop all preprocessing worker threads."""
    if not PARALLEL_IMAGE_PROCESSING or not preprocessing_threads:
        return
        
    preprocessing_stop_event.set()
    for t in preprocessing_threads:
        t.join(timeout=1.0)
    preprocessing_threads.clear()
    
    # Clear queue
    while not image_preprocessing_queue.empty():
        try:
            image_preprocessing_queue.get_nowait()
            image_preprocessing_queue.task_done()
        except queue.Empty:
            break

def preprocess_frame_for_ocr(frame):
    """
    Preprocess frame to improve OCR performance by enhancing text visibility.
    """
    if not PREPROCESS_FRAMES:
        return frame
        
    try:
        # Convert to grayscale for better text extraction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        
        # Apply adaptive thresholding to enhance text contrast
        thresh = cv2.adaptiveThreshold(
            clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Optional: Apply morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Combine the original frame and the processed version for better OCR
        result = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
        return result
    except Exception as e:
        print(f"Frame preprocessing error: {e}")
        return frame  # Return original frame on error

def adjust_batch_size(processing_time, batch_size, frames_processed):
    """
    Dynamically adjust batch size based on processing performance.
    Returns an optimized batch size for next iteration.
    """
    if not ENABLE_ADAPTIVE_BATCH_SIZING:
        return OCR_BATCH_SIZE
        
    # Calculate frames per second
    fps = frames_processed / max(0.001, processing_time)
    
    # Store for statistics
    with progress_lock:
        perf_stats["last_processing_speed"] = fps
        perf_stats["batch_processing_times"].append((batch_size, fps))
        
        # Keep only the last 10 measurements
        if len(perf_stats["batch_processing_times"]) > 10:
            perf_stats["batch_processing_times"].pop(0)
    
    # Adaptive strategy: increase batch size if processing is fast, decrease if slow
    if fps > 10:  # If processing more than 10 frames per second, increase batch size
        new_size = min(ADAPTIVE_BATCH_MAX, int(batch_size * 1.2))
    elif fps < 5:  # If processing less than 5 frames per second, decrease batch size
        new_size = max(ADAPTIVE_BATCH_MIN, int(batch_size * 0.8))
    else:
        new_size = batch_size  # Keep the same batch size
        
    # Ensure new size is within bounds
    new_size = max(ADAPTIVE_BATCH_MIN, min(ADAPTIVE_BATCH_MAX, new_size))
    
    # Log significant changes
    if new_size != batch_size:
        print(f"Adjusting batch size: {batch_size} → {new_size} (processing speed: {fps:.2f} fps)")
        
    with progress_lock:
        perf_stats["last_batch_size"] = new_size
        
    return new_size

def perform_ocr_on_frame_batch(batch_frames):
    """Perform OCR on a batch of frames using Vision Framework with Metal acceleration."""
    # Access the global variables
    global USE_IN_MEMORY_PROCESSING, OCR_BATCH_SIZE
    
    if not VISION_AVAILABLE:
        print("Vision Framework not available")
        return []
        
    results = []
    # Get the global Vision request handler
    text_request = configure_vision_request_for_metal()
    
    if text_request is None:
        return []
    
    # Start timing the batch processing
    batch_start_time = time.time()
    
    # Disable verbose logging for repeated errors
    core_graphics_error_logged = False
    
    # Track temp files to ensure cleanup
    temp_files = []
    
    # Process the batch of frames
    processed_frames = 0
    
    try:
        # Process each frame in the batch
        for frame_num, frame in batch_frames:
            try:
                # Check GPU memory before processing each frame
                if not check_gpu_memory():
                    print("GPU memory high, pausing to clear memory...")
                    clear_gpu_memory()
                    time.sleep(0.1)  # Shorter pause to let resources be cleaned up
                
                frame_start_time = time.time()
                frame_results = []
                
                # Compute frame hash for caching
                frame_hash = compute_frame_hash(frame)
                
                # Check if we have a cached result
                cached_result = get_cached_result(frame_num, frame_hash)
                if cached_result is not None:
                    results.extend(cached_result)
                    processed_frames += 1
                    continue
                
                # Use local var to avoid changing global during parallel execution
                use_in_memory = USE_IN_MEMORY_PROCESSING
                
                # Preprocess frame to enhance text recognition
                if PREPROCESS_FRAMES:
                    preproc_start = time.time()
                    frame = preprocess_frame_for_ocr(frame)
                    with progress_lock:
                        perf_stats["preprocessing_time"] += time.time() - preproc_start
                
                if use_in_memory and HAS_CORE_GRAPHICS:
                    # Convert OpenCV image to CGImage for direct processing
                    try:
                        # Convert BGR to RGB (OpenCV uses BGR, but Apple frameworks expect RGB)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Create a data provider from the numpy array
                        height, width, channels = rgb_frame.shape
                        bytes_per_row = channels * width
                        
                        # Use CoreFoundation and CoreGraphics for image conversion
                        data = CFDataCreate(None, rgb_frame.tobytes(), len(rgb_frame.tobytes()))
                        provider = CGDataProviderCreateWithData(None, data, width * height * channels, None)
                        
                        cg_image = CGImageCreate(
                            width, height, 8, 8 * channels, bytes_per_row, 
                            CGColorSpaceCreateDeviceRGB(),
                            32, provider, None, False, 0
                        )
                        
                        # Create Vision request handler directly from CGImage
                        request_handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
                            cg_image, None
                        )
                        
                        # Track OCR time specifically
                        ocr_start = time.time()
                        success = request_handler.performRequests_error_([text_request], None)
                        ocr_time = time.time() - ocr_start
                        
                        with progress_lock:
                            perf_stats["ocr_time"] += ocr_time
                            perf_stats["total_ocr_calls"] += 1
                            perf_stats["gpu_time"] += ocr_time  # Approximate GPU time
                        
                        if not success:
                            print(f"OCR failed for frame {frame_num}")
                            continue
                        
                        # Process observations and break down into word-level results
                        observations = text_request.results()
                        frame_results = process_recognized_text_into_words(observations, frame_num, frame.shape)
                        
                        # Cache the results
                        cache_frame_result(frame_num, frame_hash, frame_results)
                        
                        # Release image resources explicitly
                        del cg_image
                        del provider
                        del data
                        del rgb_frame
                        del request_handler
                        
                    except Exception as e:
                        if not core_graphics_error_logged:
                            print(f"Error in in-memory processing: {str(e)}. Falling back to file-based method.")
                            core_graphics_error_logged = True
                        use_in_memory = False
                else:
                    use_in_memory = False
                
                # Fall back to file-based method if needed (note we're using local variable)
                if not use_in_memory:
                    # Save frame to temp file for Vision processing
                    temp_dir = tempfile.gettempdir()
                    temp_image_path = os.path.join(temp_dir, f"frame_{frame_num}_{uuid.uuid4()}.png")
                    temp_files.append(temp_image_path)  # Track for cleanup
                    
                    cv2.imwrite(temp_image_path, frame)
                    
                    try:
                        # Create Vision request handler
                        image_url = NSURL.fileURLWithPath_(temp_image_path)
                        request_handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
                            image_url, None
                        )
                        
                        # Explicitly set Metal preferences in options
                        if HAS_METAL:
                            options = NSMutableDictionary.dictionary()
                            options.setValue_forKey_(True, "VNUseMetalAcceleration")
                            options.setValue_forKey_(True, "VNUseGPU")
                            
                            # For file-based processing, try with options
                            request_handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
                                image_url, options
                            )
                        
                        # Track OCR time specifically
                        ocr_start = time.time()
                        success = request_handler.performRequests_error_([text_request], None)
                        ocr_time = time.time() - ocr_start
                        
                        with progress_lock:
                            perf_stats["ocr_time"] += ocr_time
                            perf_stats["total_ocr_calls"] += 1
                        
                        if not success:
                            print(f"OCR failed for frame {frame_num}")
                            continue
                        
                        # Process observations and break down into word-level results
                        observations = text_request.results()
                        frame_results = process_recognized_text_into_words(observations, frame_num, frame.shape)
                        
                        # Cache the results
                        cache_frame_result(frame_num, frame_hash, frame_results)
                        
                        # Clean up resources
                        del request_handler
                        del image_url
                        
                    finally:
                        # Always remove the temp file
                        try:
                            if os.path.exists(temp_image_path):
                                os.remove(temp_image_path)
                                temp_files.remove(temp_image_path)  # Remove from tracking list
                        except Exception as e:
                            pass  # Ignore errors when removing temp files
                
                # Add results
                results.extend(frame_results)
                processed_frames += 1
                
                # Calculate processing time for this frame
                frame_time = time.time() - frame_start_time
                with progress_lock:
                    perf_stats["frame_processing_time"] += frame_time
                    perf_stats["total_frames"] += 1
                
                # Clean up frame-specific resources
                del frame_results
                del frame
                
            except Exception as e:
                print(f"Error processing frame {frame_num}: {str(e)}")
                traceback.print_exc()
    
    finally:
        # Ensure all temp files are deleted
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
                
    # Calculate and update batch processing time
    batch_time = time.time() - batch_start_time
    
    # Adjust batch size for future processing
    if ENABLE_ADAPTIVE_BATCH_SIZING and processed_frames > 0:
        global OCR_BATCH_SIZE
        new_batch_size = adjust_batch_size(batch_time, OCR_BATCH_SIZE, processed_frames)
        OCR_BATCH_SIZE = new_batch_size
    
    # Clean up batch resources
    batch_frames = None
    
    # Force a memory cleanup after the batch
    clear_gpu_memory()
    
    return results

def time_to_frame(time_str, fps):
    """Convert time string in MM:SS format to frame number."""
    try:
        minutes, seconds = map(int, time_str.split(':'))
        total_seconds = minutes * 60 + seconds
        return int(total_seconds * fps)
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}. Use MM:SS format.")

def parse_time_periods(time_periods_str, fps):
    """Parse time periods string into list of frame ranges."""
    if not time_periods_str:
        return None
        
    frame_ranges = []
    period_pattern = r'(\d+:\d+)-(\d+:\d+)'
    
    for period in time_periods_str:
        match = re.match(period_pattern, period)
        if not match:
            raise ValueError(f"Invalid time period format: {period}. Use MM:SS-MM:SS format.")
            
        start_time, end_time = match.groups()
        start_frame = time_to_frame(start_time, fps)
        end_frame = time_to_frame(end_time, fps)
        
        if start_frame >= end_frame:
            print(f"Warning: Start time {start_time} is after end time {end_time}. Skipping this period.")
            continue
            
        frame_ranges.append((start_frame, end_frame))
        print(f"Added time period: {start_time}-{end_time} (frames {start_frame}-{end_frame})")
    
    return frame_ranges if frame_ranges else None

def is_frame_in_periods(frame_num, frame_ranges):
    """Check if a frame number is within any of the specified frame ranges."""
    if frame_ranges is None:
        return True
        
    for start_frame, end_frame in frame_ranges:
        if start_frame <= frame_num <= end_frame:
            return True
    
    return False

def is_timestamp_in_periods(timestamp_us, frame_ranges, fps):
    """Check if a timestamp (in microseconds) is within any of the specified frame ranges."""
    if frame_ranges is None:
        return True
        
    # Convert timestamp to frame number
    frame_num = int((timestamp_us / 1000000) * fps)
    
    return is_frame_in_periods(frame_num, frame_ranges)

def extract_text_from_video(video_path, frame_interval=5, max_frames=None, frame_ranges=None, chunk_size=None):
    """Extract text from video frames using Apple Vision Framework with threading.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    frame_interval : int, optional
        Interval between frames to sample.
    max_frames : int or None, optional
        Maximum number of frames to process.
    frame_ranges : list or None, optional
        Specific frame ranges to process.
    chunk_size : int or None, optional
        Maximum number of frames to load and process at once. If ``None`` a
        dynamic value will be selected based on resolution and sampling rate.
    """
    print(f"Extracting text from video {video_path} (interval: {frame_interval})...")
    
    # Force garbage collection at the beginning
    import gc
    gc.collect()
    
    # Initialize Vision Framework once
    if VISION_AVAILABLE:
        text_request = initialize_vision_framework()
        if text_request is None:
            print("Failed to initialize Vision Framework")
            return None
        print("Vision Framework initialized successfully")
    
    # Reset performance stats
    global perf_stats, OCR_BATCH_SIZE
    perf_stats = {
        "ocr_time": 0,
        "frame_processing_time": 0,
        "total_frames": 0,
        "total_ocr_calls": 0,
        "gpu_time": 0,
        "gpu_memory_peak": 0,
        "batch_processing_times": [],
        "last_batch_size": OCR_BATCH_SIZE,
        "last_processing_speed": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "preprocessing_time": 0
    }
    
    # Clear frame cache
    global frame_cache, frame_cache_keys, frame_preprocessing_cache
    with frame_cache_lock:
        frame_cache = {}
        frame_cache_keys = []
        frame_preprocessing_cache = {}
    
    # Start preprocessing workers if enabled
    if PARALLEL_IMAGE_PROCESSING:
        num_preproc_workers = min(4, os.cpu_count() or 4)  # Limit to 4 workers
        start_preprocessing_workers(num_preproc_workers)
    
    # Verify Metal status
    if HAS_METAL:
        device = Metal.MTLCreateSystemDefaultDevice()
        print(f"Using Metal GPU: {device.name()}")
        
        # Log GPU memory if available
        try:
            if hasattr(device, 'currentAllocatedSize'):
                print(f"Initial GPU memory usage: {device.currentAllocatedSize() / (1024*1024):.2f} MB")
        except:
            pass
    
    if not os.path.isfile(video_path):
        print(f"Video file not found: {video_path}")
        return None
        
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None
        
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height}, {fps} FPS, {frame_count} frames")
    
    # Determine if we should resize frames for better memory usage
    # For very high-resolution videos, downscale to save memory
    should_resize = width > 1920 or height > 1080  # Lower threshold to always resize for memory
    resize_scale = 0.5 if should_resize else 1.0  # More aggressive resizing to save memory
    
    # For extremely high-res videos, scale down even further
    if width > 2560 or height > 1440:
        resize_scale = 0.25
        print(f"Very high resolution video detected. Using aggressive scaling: {resize_scale}")
    
    if should_resize:
        print(f"High resolution video detected. Will resize frames to {int(width*resize_scale)}x{int(height*resize_scale)} for processing")
    
    # Determine frames to sample
    frames_to_sample = []
    if frame_ranges:
        # Using specific time periods
        for start_frame, end_frame in frame_ranges:
            print(f"Adding frames from range {start_frame} to {end_frame}")
            frames_to_sample.extend(range(start_frame, end_frame + 1, frame_interval))
    else:
        # Sample frames at regular intervals
        frames_to_sample = list(range(0, frame_count, frame_interval))
    
    if max_frames and len(frames_to_sample) > max_frames:
        # Limit to max frames with evenly distributed sampling
        step = len(frames_to_sample) // max_frames
        frames_to_sample = frames_to_sample[::step][:max_frames]

    print(f"Processing {len(frames_to_sample)} frames")

    # Determine maximum frames to load at once
    if chunk_size is not None:
        max_frames_at_once = min(chunk_size, len(frames_to_sample))
    else:
        max_frames_at_once = min(500, len(frames_to_sample))
        # For high-resolution or dense sampling, reduce further
        if frame_interval <= 2 or should_resize:
            max_frames_at_once = min(100, len(frames_to_sample))

    # Process frames in chunks to manage memory
    all_results = []
    chunk_size = max_frames_at_once
    
    # Record the time for all processing
    overall_start_time = time.time()
    batch_start_time = overall_start_time
    
    for chunk_start in range(0, len(frames_to_sample), chunk_size):
        # Force garbage collection between chunks
        gc.collect()
        clear_gpu_memory()
        
        chunk_end = min(chunk_start + chunk_size, len(frames_to_sample))
        current_frames = frames_to_sample[chunk_start:chunk_end]
        
        print(f"Processing chunk {chunk_start//chunk_size + 1}/{(len(frames_to_sample) + chunk_size - 1)//chunk_size} "
              f"({len(current_frames)} frames)")
        
        # Prepare batches for this chunk
        frame_batches = []
        current_batch = []
        
        # Use the current OCR_BATCH_SIZE which will adapt during processing
        current_batch_size = OCR_BATCH_SIZE
        
        # Force GC before loading new frames
        gc.collect()
    
        # Load frames for this chunk
        loaded_frames = []
        for frame_num in current_frames:
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Error reading frame {frame_num}")
                continue
            
            # Resize frame if needed
            if should_resize:
                frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)
                
            loaded_frames.append((frame_num, frame))
            
            # Periodically clear memory during loading
            if len(loaded_frames) % 50 == 0:
                gc.collect()
        
        # Create batches from loaded frames
        for i in range(0, len(loaded_frames), current_batch_size):
            batch = loaded_frames[i:i+current_batch_size]
            frame_batches.append(batch)
            
            # Clear memory periodically
            if len(frame_batches) % 2 == 0:  # More frequent cleanup
                clear_gpu_memory()
                # Force garbage collection
                gc.collect()
        
        # Free up memory used by loaded_frames
        loaded_frames = None
        gc.collect()
        
        print(f"Created {len(frame_batches)} batches for this chunk")
        
        # Process batches in parallel with controlled execution
        chunk_results = []
        global total_progress
        total_progress = 0
        
        # Make sure we're not exceeding available memory
        max_parallel_batches = min(2, max(1, int(GPU_MEMORY_LIMIT_MB // 1024)))  # Even more conservative
        actual_threads = min(MAX_THREADS, max_parallel_batches)
        
        print(f"Using {actual_threads} threads for parallel processing")
        
        with ThreadPoolExecutor(max_workers=actual_threads) as executor:
            # Process batches with controlled execution to prevent memory issues
            futures_list = []
            completed_batches = 0
            
            # Submit initial batch of tasks (only 1 at a time to start)
            initial_batches = min(1, len(frame_batches))
            for i in range(initial_batches):
                futures_list.append(executor.submit(perform_ocr_on_frame_batch, frame_batches[i]))
            
            # Process remaining batches as tasks complete
            batch_idx = initial_batches
            
            while futures_list:
                # Wait for the next task to complete
                done, not_done = concurrent.futures.wait(
                    futures_list, 
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                # Update our futures list (keeping only unfinished futures)
                futures_list = list(not_done)
                
                # Process completed results
                for future in done:
                    try:
                        results = future.result()
                        chunk_results.extend(results)
                        completed_batches += 1
                        
                        # Report progress
                        progress_pct = (completed_batches / len(frame_batches) * 100) * ((chunk_end - chunk_start) / len(frames_to_sample))
                        if completed_batches % max(1, len(frame_batches) // 10) == 0:  # Even more frequent updates
                            print(f"Completed {completed_batches}/{len(frame_batches)} batches in chunk ({progress_pct:.1f}%)")
                        
                        # Check memory and always clean up after each batch
                        clear_gpu_memory()
                        # Force garbage collection
                        gc.collect()
                        # Small pause to let system resources recover
                        time.sleep(0.2)
                        
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        traceback.print_exc()
                
                # Submit new tasks to replace completed ones (only 1 at a time)
                while batch_idx < len(frame_batches) and len(futures_list) < max(1, actual_threads // 2):
                    futures_list.append(executor.submit(perform_ocr_on_frame_batch, frame_batches[batch_idx]))
                    batch_idx += 1
                    # Pause briefly between submissions to prevent memory spikes
                    time.sleep(0.1)
        
        # Add results from this chunk to overall results
        all_results.extend(chunk_results)
        
        # Force cleanup between chunks
        frame_batches = None
        chunk_results = None
        gc.collect()
        clear_gpu_memory()
        
        # Report progress after each chunk
        print(f"Completed chunk {chunk_start//chunk_size + 1}/{(len(frames_to_sample) + chunk_size - 1)//chunk_size} "
              f"with {len(all_results)} text entries so far")
        
        # Wait a bit longer between chunks to ensure resources are freed
        time.sleep(1.0)
    
    # Release video capture
    cap.release()
    
    # Calculate total elapsed time from overall start
    elapsed_time = time.time() - overall_start_time
    print(f"OCR processing completed in {elapsed_time:.2f} seconds")
    print(f"Extracted {len(all_results)} text entries from {len(frames_to_sample)} frames")
    print(f"Overall performance: {len(frames_to_sample) / max(0.1, elapsed_time):.2f} frames/sec")
    
    # Report final GPU memory usage
    if HAS_METAL:
        try:
            device = Metal.MTLCreateSystemDefaultDevice()
            if hasattr(device, 'currentAllocatedSize'):
                final_memory = device.currentAllocatedSize() / (1024*1024)
                print(f"Final GPU memory usage: {final_memory:.2f} MB (peak: {perf_stats['gpu_memory_peak']:.2f} MB)")
        except:
            pass
    
    # Force final cleanup
    gc.collect()
    clear_gpu_memory()
    
    # Create DataFrame from results
    ocr_df = pd.DataFrame(all_results)
    
    if ocr_df.empty:
        print("No text was detected in any of the sampled frames")
        return None
    
    return ocr_df

def categorize_text_entries(ocr_data, separator_offset=SEPARATOR_OFFSET):
    """Categorize text entries as 'source' or 'target' based on position."""
    if ocr_data is None or ocr_data.empty:
        print("No OCR data to categorize")
        return None
    
    print("Categorizing text entries as source or target...")
    
    # Calculate separator y-coordinate (assuming source text is above target text)
    separator_y = ocr_data['top'].mean() + separator_offset
    
    # Categorize based on position
    ocr_data['category'] = ocr_data.apply(
        lambda row: 'source' if row['center_y'] < separator_y else 'target', 
        axis=1
    )
    
    # Report counts
    source_count = len(ocr_data[ocr_data['category'] == 'source'])
    target_count = len(ocr_data[ocr_data['category'] == 'target'])
    print(f"Categorized {source_count} source texts and {target_count} target texts")
    
    return ocr_data

def calculate_distance_matrix(gaze_points, text_entries):
    """Calculate distance matrix between gaze points and text entries efficiently."""
    if gaze_points.empty or text_entries.empty:
        print("No gaze points or text entries for distance calculation")
        return np.array([])
    
    # Extract coordinates as numpy arrays for vectorized operations
    gaze_coords = gaze_points[['scaled_x', 'scaled_y']].values
    text_coords = text_entries[['center_x', 'center_y']].values
    
    # Calculate distances using broadcasting
    # This creates a matrix of shape (n_gaze_points, n_text_entries)
    diff = gaze_coords[:, np.newaxis, :] - text_coords[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    
    return distances

def map_gaze_to_text_optimized(gaze_data, ocr_data, conf_threshold, distance_threshold, frame_ranges=None, fps=30):
    """Map gaze points to text entries in batches with optimized performance."""
    if gaze_data is None or gaze_data.empty or ocr_data is None or ocr_data.empty:
        print("No gaze data or OCR data for mapping")
        return None
    
    print(f"Mapping gaze points to text (confidence threshold: {conf_threshold}, distance threshold: {distance_threshold})...")
    start_time = time.time()
    
    # Filter OCR data by confidence threshold
    confident_ocr = ocr_data[ocr_data['conf'] >= conf_threshold].copy()
    print(f"Using {len(confident_ocr)} text entries with confidence >= {conf_threshold}%")
    
    if confident_ocr.empty:
        print("No text entries meet the confidence threshold")
        return None
    
    # Filter gaze data by time periods if specified
    if frame_ranges:
        original_count = len(gaze_data)
        gaze_data = gaze_data[gaze_data.apply(lambda row: is_timestamp_in_periods(row['timestampUs'], frame_ranges, fps), axis=1)]
        print(f"Filtered gaze data to time periods: {len(gaze_data)} of {original_count} points kept")
        
        if gaze_data.empty:
            print("No gaze points within specified time periods")
            return None
    
    # Initialize results storage
    mapped_gaze_data = []
    
    # Get unique frame numbers in gaze data
    unique_frames = gaze_data['frame_num'].unique()
    print(f"Processing {len(unique_frames)} unique frames")
    
    # Process frames in batches
    total_mapped = 0
    total_gaze_points = len(gaze_data)
    
    for frame_num in unique_frames:
        # Skip frames outside specified time periods
        if frame_ranges and not is_frame_in_periods(frame_num, frame_ranges):
            continue
            
        # Filter gaze data for this frame
        frame_gaze = gaze_data[gaze_data['frame_num'] == frame_num]
        
        # Filter OCR data for this frame
        frame_ocr = confident_ocr[confident_ocr['frame_num'] == frame_num]
        
        if frame_gaze.empty or frame_ocr.empty:
            continue
        
        # Calculate distances between all gaze points and text entries in this frame
        distances = calculate_distance_matrix(frame_gaze, frame_ocr)
        
        if distances.size == 0:
            continue
        
        # For each gaze point, find the closest text entry
        min_distances = np.min(distances, axis=1)
        min_indices = np.argmin(distances, axis=1)
        
        # Filter by distance threshold
        valid_mappings = min_distances <= distance_threshold
        
        # Create mapping for valid gaze points
        for i, gaze_idx in enumerate(frame_gaze.index[valid_mappings]):
            ocr_idx = frame_ocr.index[min_indices[i]]
            
            gaze_row = gaze_data.loc[gaze_idx]
            ocr_row = confident_ocr.loc[ocr_idx]
            
            mapped_gaze_data.append({
                'timestamp': gaze_row['timestampUs'],
                'frame_num': frame_num,
                'gaze_x': gaze_row['scaled_x'],
                'gaze_y': gaze_row['scaled_y'],
                'text': ocr_row['text'],
                'confidence': ocr_row['conf'],
                'distance': min_distances[i],
                'category': ocr_row['category'] if 'category' in ocr_row else None
            })
        
        total_mapped += sum(valid_mappings)
        
        # Report progress periodically
        if len(unique_frames) > 10 and len(mapped_gaze_data) % (total_gaze_points // 10) < 100:
            elapsed = time.time() - start_time
            progress = total_mapped / total_gaze_points * 100
            print(f"Mapping progress: {progress:.1f}% ({total_mapped}/{total_gaze_points}) in {elapsed:.1f}s")
    
    # Create final DataFrame
    mapped_df = pd.DataFrame(mapped_gaze_data)
    
    # Report mapping statistics
    elapsed_time = time.time() - start_time
    mapping_rate = total_mapped / total_gaze_points * 100
    print(f"Mapping completed in {elapsed_time:.2f} seconds")
    print(f"Mapped {total_mapped} out of {total_gaze_points} gaze points ({mapping_rate:.1f}%)")
    
    return mapped_df

def main():
    """Command-line interface for eye-tracking to text mapping."""
    global USE_FAST_MODE, PREPROCESS_FRAMES, OCR_BATCH_SIZE, ENABLE_ADAPTIVE_BATCH_SIZING, USE_FRAME_CACHE, PARALLEL_IMAGE_PROCESSING
    
    parser = argparse.ArgumentParser(description="Eye-tracking to text mapping using Apple Vision Framework.")
    parser.add_argument("--video", help="Path to video file", default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--csv", help="Path to eye-tracking CSV file", default=DEFAULT_CSV_PATH)
    parser.add_argument("--interval", type=int, help="Frame sampling interval", default=DEFAULT_FRAME_INTERVAL)
    parser.add_argument("--confidence", type=float, help="OCR confidence threshold (0-100)", default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--distance", type=float, help="Distance threshold for matching (pixels)", default=DEFAULT_DISTANCE_THRESHOLD)
    parser.add_argument("--output", help="Output CSV file path", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to process", default=None)
    parser.add_argument("--time-periods", help="Comma-separated list of time periods to process (format: HH:MM:SS-HH:MM:SS)", default=None)
    parser.add_argument("--encoding", help="CSV file encoding", default="utf-8")
    parser.add_argument("--fast-mode", action="store_true", help="Use fast OCR mode instead of accurate mode")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable frame preprocessing")
    parser.add_argument("--batch-size", type=int, help="OCR batch size", default=OCR_BATCH_SIZE)
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive batch sizing")
    parser.add_argument("--no-cache", action="store_true", help="Disable frame result caching")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel image preprocessing")
    parser.add_argument("--chunk-size", type=int, help="Maximum frames to process at once", default=None)
    
    args = parser.parse_args()
    
    # Apply command-line parameters to globals
    USE_FAST_MODE = args.fast_mode
    PREPROCESS_FRAMES = not args.no_preprocess
    OCR_BATCH_SIZE = args.batch_size
    ENABLE_ADAPTIVE_BATCH_SIZING = not args.no_adaptive
    USE_FRAME_CACHE = not args.no_cache
    PARALLEL_IMAGE_PROCESSING = not args.no_parallel
    
    # Show GPU information if available
    if HAS_METAL:
        device = Metal.MTLCreateSystemDefaultDevice()
        print("\n=== GPU Information ===")
        print(f"GPU Name: {device.name()}")
        if hasattr(device, 'registryID'):
            print(f"Registry ID: {device.registryID()}")
        if hasattr(device, 'maxThreadsPerThreadgroup'):
            print(f"Max Threads: {device.maxThreadsPerThreadgroup().width} x {device.maxThreadsPerThreadgroup().height} x {device.maxThreadsPerThreadgroup().depth}")
        if hasattr(device, 'recommendedMaxWorkingSetSize'):
            print(f"Recommended Working Set: {device.recommendedMaxWorkingSetSize() / (1024*1024)} MB")
        if hasattr(device, 'currentAllocatedSize'):
            print(f"Current Allocated: {device.currentAllocatedSize() / (1024*1024):.2f} MB")
        print("=" * 50 + "\n")
    
    # Process with the parsed arguments
    process_with_args(vars(args))
    
    # Final report
    if HAS_METAL:
        try:
            device = Metal.MTLCreateSystemDefaultDevice()
            print("\n=== Final GPU Stats ===")
            if hasattr(device, 'currentAllocatedSize'):
                print(f"Final GPU Memory: {device.currentAllocatedSize() / (1024*1024):.2f} MB")
            print(f"Peak GPU Memory: {perf_stats['gpu_memory_peak']:.2f} MB")
            print(f"Total GPU Time: {perf_stats['gpu_time']:.2f} seconds")
            
            # Print adaptive batch sizing stats if enabled
            if ENABLE_ADAPTIVE_BATCH_SIZING and perf_stats['batch_processing_times']:
                print("\n=== Adaptive Batch Sizing Stats ===")
                print(f"Final batch size: {perf_stats['last_batch_size']}")
                print(f"Final processing speed: {perf_stats['last_processing_speed']:.2f} fps")
                print("Batch size history:")
                for batch_size, fps in perf_stats['batch_processing_times']:
                    print(f"  Batch size {batch_size}: {fps:.2f} fps")
            print("=" * 50 + "\n")
        except:
            pass

def process_with_args(args, progress_callback=None):
    """Process video and eye-tracking data with specified arguments."""
    video_path = args.get('video') or DEFAULT_VIDEO_PATH
    csv_path = args.get('csv') or DEFAULT_CSV_PATH
    frame_interval = args.get('interval') or DEFAULT_FRAME_INTERVAL
    conf_threshold = args.get('confidence') or DEFAULT_CONFIDENCE_THRESHOLD
    distance_threshold = args.get('distance') or DEFAULT_DISTANCE_THRESHOLD
    output_file = args.get('output') or DEFAULT_OUTPUT_FILE
    max_frames = args.get('max_frames')
    time_periods_str = args.get('time_periods')
    encoding = args.get('encoding') or 'utf-8'
    chunk_size = args.get('chunk_size')
    
    try:
        # Update global configuration first - this fixes the "used prior to global declaration" error
        global WORD_LEVEL_RECOGNITION, USE_IN_MEMORY_PROCESSING, OCR_BATCH_SIZE, MAX_THREADS, GPU_MEMORY_LIMIT_MB
        
        # Extract arguments
        word_level = args.get('word_level', True)
        debug = args.get('debug', False)
        
        # Now set the values after the global declaration
        WORD_LEVEL_RECOGNITION = word_level
        USE_IN_MEMORY_PROCESSING = args.get("in_memory", True) and HAS_CORE_GRAPHICS
        OCR_BATCH_SIZE = args.get("batch_size", OCR_BATCH_SIZE)
        MAX_THREADS = args.get("threads", MAX_THREADS)
        
        # If GPU memory limit is provided, update it
        if args.get("memory_limit"):
            GPU_MEMORY_LIMIT_MB = args.get("memory_limit")
        
        # Print configuration
        print("\n=== Eye-Tracking to Text Mapping Configuration ===")
        print(f"Video file: {video_path}")
        print(f"Eye-tracking data: {csv_path}")
        print(f"Frame interval: {frame_interval}")
        print(f"Output file: {output_file}")
        print(f"Confidence threshold: {conf_threshold}%")
        print(f"Distance threshold: {distance_threshold} pixels")
        print(f"Word-level recognition: {'Enabled' if WORD_LEVEL_RECOGNITION else 'Disabled'}")
        print(f"In-memory processing: {'Enabled' if USE_IN_MEMORY_PROCESSING else 'Disabled'}")
        print(f"OCR batch size: {OCR_BATCH_SIZE}")
        print(f"GPU memory limit: {GPU_MEMORY_LIMIT_MB} MB")
        print(f"Chunk size: {chunk_size if chunk_size else 'auto'}")
        if time_periods_str:
            print(f"Time periods: {', '.join(time_periods_str)}")
        print(f"Using {MAX_THREADS} threads for processing")
        print("=" * 50 + "\n")
        
        # Debug information
        if debug:
            print("DEBUG MODE ENABLED")
            print(f"Arguments: {args}")
            
            # Check file existence
            print(f"Video file exists: {os.path.exists(video_path)}")
            print(f"CSV file exists: {os.path.exists(csv_path)}")
            
            # Check file sizes
            if os.path.exists(video_path):
                print(f"Video file size: {os.path.getsize(video_path) / (1024*1024):.1f} MB")
            if os.path.exists(csv_path):
                print(f"CSV file size: {os.path.getsize(csv_path) / 1024:.1f} KB")
            
            if time_periods_str and len(time_periods_str) == 1:
                print(f"Processing single time period: {time_periods_str[0]}")
        
        # Load eye-tracking data
        gaze_data = load_eye_tracking_data(csv_path)
        if gaze_data is None or gaze_data.empty:
            print("No valid eye-tracking data to process")
            return
        
        # Update progress
        if progress_callback:
            if not progress_callback(5, 100):  # 5% progress after loading gaze data
                return
        
        # Open video to get properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        if debug:
            print(f"Video dimensions: {width}x{height}")
            print(f"Video FPS: {fps}")
        
        # Update progress
        if progress_callback:
            if not progress_callback(10, 100):  # 10% progress after getting video properties
                return
        
        # Parse time periods if specified
        frame_ranges = None
        if time_periods_str:
            try:
                frame_ranges = parse_time_periods(time_periods_str, fps)
                if not frame_ranges:
                    print("No valid time periods specified, processing entire video")
                    
                if debug and frame_ranges:
                    for i, (start, end) in enumerate(frame_ranges):
                        print(f"Frame range {i+1}: {start} to {end} (duration: {end-start} frames)")
            except ValueError as e:
                print(f"Error parsing time periods: {e}")
                print("Processing entire video instead")
        
        # Map gaze timestamps to frames
        gaze_data = map_gaze_to_frames(gaze_data, fps)
        
        # Scale gaze coordinates to video dimensions
        gaze_data = scale_gaze_to_video(gaze_data, width, height)
        
        if debug and not gaze_data.empty:
            print("Sample of gaze data after mapping and scaling:")
            print(gaze_data.head())
            
            if frame_ranges:
                # Count gaze points in the specified time ranges
                in_range_count = sum(1 for _, row in gaze_data.iterrows() 
                                   if is_timestamp_in_periods(row['timestampUs'], frame_ranges, fps))
                print(f"Gaze points in specified time periods: {in_range_count} of {len(gaze_data)}")
        
        # Update progress
        if progress_callback:
            if not progress_callback(15, 100):  # 15% progress after preparing gaze data
                return
        
        # Extract text from video frames
        ocr_data = extract_text_from_video(
            video_path,
            frame_interval,
            frame_ranges=frame_ranges,
            chunk_size=chunk_size
        )
        if ocr_data is None or ocr_data.empty:
            print("No text detected in video frames")
            return
        
        if debug and not ocr_data.empty:
            print("Sample of OCR data:")
            print(ocr_data.head())
            print(f"Total OCR text entries: {len(ocr_data)}")
        
        # Update progress
        if progress_callback:
            if not progress_callback(70, 100):  # 70% progress after OCR
                return
        
        # Categorize text entries
        ocr_data = categorize_text_entries(ocr_data)
        
        # Update progress
        if progress_callback:
            if not progress_callback(75, 100):  # 75% progress after categorization
                return
        
        # Map gaze points to text
        mapped_data = map_gaze_to_text_optimized(gaze_data, ocr_data, conf_threshold, distance_threshold, frame_ranges, fps)
        if mapped_data is None or mapped_data.empty:
            print("No gaze points could be mapped to text")
            return
        
        # Update progress
        if progress_callback:
            if not progress_callback(95, 100):  # 95% progress after mapping
                return
        
        # Save mapped data to CSV
        mapped_data.to_csv(output_file, index=False)
        print(f"Saved {len(mapped_data)} mapped gaze points to {output_file}")
        
        # Save OCR results for reference
        ocr_data.to_csv("ocr_results.csv", index=False)
        print(f"Saved {len(ocr_data)} OCR results to ocr_results.csv")
        
        # Update progress
        if progress_callback:
            progress_callback(100, 100)  # 100% progress when done
        
        print("\nMapping process completed successfully")
        
        # Stop preprocessing workers if they were used
        if PARALLEL_IMAGE_PROCESSING:
            stop_preprocessing_workers()
        
        return True
        
    except Exception as e:
        print(f"Error processing video: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 