#!/usr/bin/env -S uv run --
# /// script
# dependencies = [
#   "ultralytics",
#   "opencv-python",
#   "typer",
#   "icecream",
#   "pydantic",
#   "numpy",
#   "torch",
#   "matplotlib",
#   "Pillow",
#   "imutils",
#   "rich"
# ]
# ///
# gen-unique-video from video with lots of changeless frames


from ultralytics import YOLO

import cv_helper
from icecream import ic
import cv2
import typer
import os.path
import pose_helper
from typing import List
import pickle
from pathlib import Path
from pydantic import BaseModel
import datetime
import os
import json
from rich.console import Console
from rich.prompt import Confirm
import subprocess

console = Console()

app = typer.Typer(no_args_is_help=True)

# YOLO model path constant
YOLO_POSE_MODEL = "yolo11n-pose.pt"


class YoloResult(BaseModel):
    keypoints: List

    @classmethod
    def from_predict(cls, yolo_result):
        return YoloResult(keypoints=yolo_result[0].keypoints)


class YoloFrame(BaseModel):
    frame: int
    yolo_results: YoloResult


class CaptureYoloData:
    def __init__(self, pickle_path: Path):
        self.yolo_frames: List[YoloFrame] = []
        self.pickle_path = pickle_path

    def create(self, input_video):
        # self.yolo = YOLO('yolov8n-seg.pt')  # pretrained YOLOv8n model
        self.yolo = YOLO(YOLO_POSE_MODEL)  # pretrained YOLOv8n model

    def destroy(self):
        # write to gz file
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.yolo_frames, f)

    def frame(self, idx, frame):
        # results don't move so frequently that we need to re-yolo
        # on each frame, so just do every 500ms

        # if idx % 15 != 0:
        # return

        results = self.yolo.predict(frame, verbose=False)
        if not results:
            return
        self.yolo_frames.append(
            YoloFrame(frame=idx, yolo_results=YoloResult.from_predict(results))
        )


class SwingsProcessor:
    def __init__(
        self,
        base_filename,
        yolo_frames_path: Path,
        label: str,
        body_part_display_seconds: int,
    ):
        self.base_filename = base_filename
        self.label = label
        self.body_part_display_seconds = body_part_display_seconds
        self.current_frame = 0
        with open(yolo_frames_path, "rb") as f:
            self.yolo_frames = pickle.load(f)

    def create(self, input_video):
        self.video = input_video
        self.fps = input_video.get(cv2.CAP_PROP_FPS)
        # self.yolo = YOLO('yolov8n-seg.pt')  # pretrained YOLOv8n model
        self.yolo = YOLO(YOLO_POSE_MODEL)  # pretrained YOLOv8n model
        self.yolo_filename = f"{self.base_filename}.mp4"
        self.yolo_writer = cv_helper.LazyVideoWriter(self.yolo_filename, self.fps)
        self.output_video_files = [self.yolo_writer]
        self.rep_counter = pose_helper.SwingRepCounter()

    def destroy(self):
        cv2.destroyAllWindows()
        for f in self.output_video_files:
            f.release()

    def frame(self, idx, frame):
        # Track current frame for body part display logic
        self.current_frame = idx

        # results don't move so frequently that we need to re-yolo
        # on each frame, so just do every 500ms

        # if idx % self.update_freq != 0:
        self.results = self.yolo_frames[idx].yolo_results.keypoints[0]

        if not self.results:
            # no frame to process
            return

        is_jupyter = False
        if is_jupyter:
            # output on jupyter every second
            if idx % self.fps == 0:
                ic(idx)

        base_image = frame  # self.results[0].plot()
        self.rep_counter.frame(
            is_hinge=pose_helper.Body(self.results).spine_vertical() < 45
        )

        # Calculate if body parts should be visible based on frame number and fps
        show_body_parts = True
        if self.body_part_display_seconds > 0:
            show_body_parts = (idx / self.fps) < self.body_part_display_seconds

        base_image = pose_helper.add_pose(
            keypoints=self.results,
            im=base_image,
            frame=idx,
            rep=self.rep_counter.rep,
            label=self.label,
            show_body_parts=show_body_parts,
        )
        self.yolo_writer.write(base_image)


def process_video(video_input_file: str) -> List[YoloFrame]:
    """Process a video file with YOLO and return the frames data."""
    input_video = cv2.VideoCapture(video_input_file)
    if not input_video.isOpened():
        raise Exception(f"Unable to Open {video_input_file}")

    # Get directory and base filename separately
    base_name = os.path.basename(video_input_file)
    base_filename = os.path.splitext(base_name)[0]
    yolo_data_path = f"output/{base_filename}-yolo.pickle.gz"
    path = Path(yolo_data_path)
    capture = CaptureYoloData(path)
    cv_helper.process_video(input_video, capture)
    return capture.yolo_frames

def process_video_with_yolo(
    video_input_file: str,
    output_file: str,
    yolo_data: List[YoloFrame],
    label: bool,
    body_part_seconds: float,
):
    """Process a video file with YOLO data and save the output."""
    input_video = cv2.VideoCapture(video_input_file)
    if not input_video.isOpened():
        raise Exception(f"Unable to Open {video_input_file}")

    # Get directory and base filename separately
    base_name = os.path.basename(video_input_file)
    base_filename = os.path.splitext(base_name)[0]
    yolo_data_path = f"output/{base_filename}-yolo.pickle.gz"
    pickle_path = Path(yolo_data_path)
    
    processor = SwingsProcessor(
        base_filename=output_file.replace('.mp4', ''),
        yolo_frames_path=pickle_path,
        label=str(datetime.datetime.now().strftime("%Y-%m-%d")) if label else "",
        body_part_display_seconds=body_part_seconds,
    )
    cv_helper.process_video(input_video, processor)


@app.command()
def swings(
    video_input_file: str = typer.Argument(
        "in.mp4", help="Input video file to process"
    ),
    force_yolo: bool = typer.Option(
        False, help="Force regeneration of YOLO data"
    ),
    force_video: bool = typer.Option(
        False, help="Force regeneration of video output"
    ),
    label: bool = typer.Option(
        False, help="Label the video with swing counts"
    ),
    should_open: bool = typer.Option(
        True, help="Open the processed video file"
    ),
    prompt: bool = typer.Option(
        False, help="Prompt before generating video output"
    ),
    body_part_seconds: float = typer.Option(
        0.5, help="Seconds to show body part labels"
    ),
):
    """Process a video file to detect and analyze swinging motions."""
    if video_input_file is None:
        typer.echo("Error: Missing argument 'VIDEO_INPUT_FILE'")
        typer.echo("Try 'python yolo.py swings process --help' for more information.")
        raise typer.Exit(1)
        
    console.print(f"[bold]Running YOLO over[/bold] {video_input_file}")
    
    # Get directory and base filename separately
    base_name = os.path.basename(video_input_file)
    base_filename = os.path.splitext(base_name)[0]
    
    console.print(f"Processing File {video_input_file}, with base {base_filename}")

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Check if YOLO data exists
    yolo_data_file = f"output/{base_filename}-yolo.pickle.gz"
    if not force_yolo and os.path.exists(yolo_data_file):
        console.print(f"[green]Using existing YOLO data from[/green] {yolo_data_file}")
        with open(yolo_data_file, "rb") as f:
            yolo_data = pickle.load(f)
    else:
        console.print("[yellow]Generating new YOLO data[/yellow]")
        yolo_data = process_video(video_input_file)
        with open(yolo_data_file, "wb") as f:
            pickle.dump(yolo_data, f)

    # Process the video with YOLO data
    output_file = f"output/{base_filename}-processed.mp4"
    if not force_video and os.path.exists(output_file):
        console.print(f"[green]Using existing video output[/green] {output_file}")
    else:
        if prompt:
            if not Confirm.ask(f"Generate video output {output_file}?"):
                return
        console.print("[yellow]Generating new video output[/yellow]")
        process_video_with_yolo(
            video_input_file,
            output_file,
            yolo_data,
            label=label,
            body_part_seconds=body_part_seconds,
        )

    if should_open:
        console.print("[yellow]Opening video output[/yellow]")
        if os.path.exists(output_file):
            subprocess.run(["open", output_file])
        else:
            console.print(f"[red]Error: Output file not found at {output_file}[/red]")
            # Try to find the file with a similar name
            import glob
            possible_files = glob.glob(f"output/*{base_filename}*.mp4")
            if possible_files:
                console.print(f"[yellow]Found similar files:[/yellow]")
                for f in possible_files:
                    console.print(f"  {f}")
                # Open the first one
                subprocess.run(["open", possible_files[0]])
                console.print(f"[green]Opened {possible_files[0]}[/green]")


if __name__ == "__main__":
    app()
