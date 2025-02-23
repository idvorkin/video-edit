#!/usr/bin/env python3

import cv2
import pytesseract
from PIL import Image
import numpy as np
from datetime import timedelta
import argparse
import sys
import cv_helper
import typer
from icecream import ic
import os
from pathlib import Path

app = typer.Typer()

def timestamp_to_string(timestamp_ms):
    """Convert milliseconds to HH:MM:SS format"""
    td = timedelta(milliseconds=timestamp_ms)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class TextDetector:
    def __init__(self, frames_per_second=1, output_file=None):
        self.last_text = None
        self.frames_per_second = frames_per_second
        self.output_file = output_file
        self.output_fp = None
        # Configure pytesseract path if needed
        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

    def create(self, input_video):
        """Initialize the processor with video properties"""
        self.fps = input_video.get(cv2.CAP_PROP_FPS)
        self.frame_count = 0
        # Calculate how many frames to skip to achieve desired frames_per_second
        self.frame_skip = int(self.fps / self.frames_per_second)
        ic(f"Video FPS: {self.fps}, Processing {self.frames_per_second} FPS, Skipping every {self.frame_skip} frames")
        # Create output directory if it doesn't exist
        if self.output_file:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            # Open output file in write mode
            self.output_fp = open(self.output_file, 'w')

    def destroy(self):
        """Cleanup resources"""
        cv2.destroyAllWindows()
        if self.output_fp:
            self.output_fp.close()

    def normalize_text(self, text):
        """Normalize text for comparison by removing extra spaces and lowercasing"""
        return ' '.join(text.lower().split())

    def frame(self, idx, frame):
        """Process a single frame"""
        self.frame_count = idx

        # Process frames based on desired frames_per_second
        if idx % self.frame_skip == 0:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply inverted thresholding for white text on black background
            thresh = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,  # White text on black background
                11,
                2
            )

            # Convert OpenCV image to PIL Image for Tesseract
            pil_image = Image.fromarray(thresh)

            try:
                # Get text and confidence data
                data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
                
                # Process all detected text blocks
                text_blocks = []
                for i, conf in enumerate(data['conf']):
                    try:
                        conf_val = float(conf)
                        if conf_val > 25:  # Confidence threshold
                            text = data['text'][i].strip()
                            if text and sum(1 for c in text if c.isalpha()) >= 4:
                                text_blocks.append(text)
                    except ValueError:
                        continue  # Skip invalid confidence values
                
                # Combine text blocks
                text = ' '.join(text_blocks).strip()
                
                # Only output if text is found and different from last text
                if text:
                    normalized_text = self.normalize_text(text)
                    if not self.last_text or normalized_text != self.normalize_text(self.last_text):
                        timestamp_ms = (idx / self.fps) * 1000
                        output_line = f"{timestamp_to_string(timestamp_ms)} -> {text}\n"
                        if self.output_fp:
                            self.output_fp.write(output_line)
                            self.output_fp.flush()
                        else:
                            print(output_line, end='')
                        self.last_text = text

            except Exception as e:
                print(f"Error processing frame {idx}: {e}", file=sys.stderr)

@app.command()
def process_video(
    video_path: str = typer.Argument("input.mp4", help="Path to the video file to process"),
    fps: float = typer.Option(1/3, help="Number of frames to process per second"),
    output_file: str = typer.Option(
        os.path.expanduser("~/tmp/timecode.txt"),
        help="Path to output file (default: ~/tmp/timecode.txt)"
    )
):
    """
    Process a video file and extract text with timestamps.
    Outputs timestamps and detected text in the format: HH:MM:SS -> text
    """
    ic(f"Processing video for text: {video_path}")
    ic(f"Output file: {output_file}")
    
    input_video = cv_helper.cv2_video(video_path)
    detector = TextDetector(frames_per_second=fps, output_file=output_file)
    cv_helper.process_video(input_video, detector)

if __name__ == "__main__":
    app() 