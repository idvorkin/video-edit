#!python3
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


app = typer.Typer()


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
        self.yolo = YOLO("yolov8n-pose.pt")  # pretrained YOLOv8n model

    def destroy(self):
        # write to gz file
        with open(self.pickle_path.name, "wb") as f:
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
        with open(yolo_frames_path.name, "rb") as f:
            self.yolo_frames = pickle.load(f)

    def create(self, input_video):
        self.video = input_video
        self.fps = input_video.get(cv2.CAP_PROP_FPS)
        # self.yolo = YOLO('yolov8n-seg.pt')  # pretrained YOLOv8n model
        self.yolo = YOLO("yolov8n-pose.pt")  # pretrained YOLOv8n model
        self.yolo_filename = f"{self.base_filename}_yolo.mp4"
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


@app.command()
def yolo(
    video_input_file: str = typer.Argument(
        "in.mp4", help="Input video file to process"
    ),
):
    """
    Process a video file with YOLO pose detection and save the results.

    This command will:
    1. Load the video file
    2. Run YOLO pose detection on the frames
    3. Save the detection results to a pickle file (filename.yolo_frames.pickle.gz)

    The results can then be used by the 'swings' command for further processing.
    """
    ic(f"Running Yolo Over {video_input_file}")
    base_filename = video_input_file.split(".")[0]

    input_video = cv2.VideoCapture(video_input_file)
    if not input_video.isOpened():
        print(f"Unable to Open {video_input_file}")
        return

    ic(f"Processing File {video_input_file}, w/{base_filename}")
    path = Path(f"{base_filename}.yolo_frames.pickle.gz")
    capture = CaptureYoloData(path)

    cv_helper.process_video(input_video, capture)
    return


@app.command()
def swings(
    video_input_file: str = typer.Argument(
        "in.mp4", help="Input video file to process"
    ),
    force: bool = typer.Option(
        False, help="Force processing even if output file exists"
    ),
    label: str = typer.Option(
        str(datetime.datetime.now().strftime("%Y-%m-%d")),
        help="Label to add to the processed video",
    ),
    open: bool = typer.Option(True, help="Open the processed video after completion"),
    body_part_seconds: int = typer.Option(
        2, help="Number of seconds to display body part labels (0 for always)"
    ),
) -> None:
    """
    Process a video file to detect and analyze swinging motions.

    This command will:
    1. Load the pre-processed YOLO detection results
    2. Analyze spine angles and count repetitions
    3. Generate an annotated video with pose detection and rep counting
    4. Save the result as filename_yolo.mp4

    Requirements:
    - The video must have been processed with the 'yolo' command first
    - The .yolo_frames.pickle.gz file must exist
    """
    ic(f"Running Yolo Over {video_input_file}")
    base_filename = video_input_file.split(".")[0]
    unique_filename = f"{base_filename}_unique.mp4"

    if not force and os.path.exists(unique_filename):
        print(f"{unique_filename} exists, skipping")
        return

    input_video = cv2.VideoCapture(video_input_file)
    if not input_video.isOpened():
        print(f"Unable to Open {video_input_file}")
        return

    ic(f"Processing File {video_input_file}, w/{base_filename}")
    yolo = SwingsProcessor(
        base_filename,
        yolo_frames_path=Path(f"{base_filename}.yolo_frames.pickle.gz"),
        label=label,
        body_part_display_seconds=body_part_seconds,
    )

    cv_helper.process_video(input_video, yolo)
    if open:
        # run shell open on the video
        os.system(f"open {yolo.yolo_filename}")

    return


if __name__ == "__main__":
    app()
