#!python3
# gen-unique-video from video with lots of changeless frames


import torch
from plots import Annotator, colors
from PIL import Image

import cv_helper
from icecream import ic
import cv2
import numpy as np
from dataclasses import dataclass
import typer
import os.path

app = typer.Typer()


class YoloProcessor:
    def __init__(self, base_filename):
        self.base_filename = base_filename
        self.in_fps = in_fps
        self.debug_window_refresh_rate = int(
            self.in_fps / 2
        )  # every 0.5 seconds; TODO Compute
        pass

    def create(self, input_video):
        self.video = input_video
        self.video_fps = input_video.get(cv2.CAP_PROP_FPS)
        self.yolo = torch.hub.load(
            "ultralytics/yolov5", "yolov5s"
        )  # or yolov5m, yolov5l, yolov5x, custom
        self.unique_filename = f"{self.base_filename}_unique.mp4"
        self.output_unique = cv_helper.LazyVideoWriter(
            self.unique_filename, self.in_fps
        )
        self.mask_filename = f"{self.base_filename}_mask.mp4"
        self.output_unique_mask = cv_helper.LazyVideoWriter(
            self.mask_filename, self.in_fps
        )
        self.output_video_files = [self.output_unique, self.output_unique_mask]

    def destroy(self):
        cv2.destroyAllWindows()
        for f in self.output_video_files:
            f.release()

    def frame(self, idx, frame):
        results = self.yolo(frame)
        predictions = results.pred[0]

        # PyTorch uses PIL Format
        # I wonder if I can skip some of these switches
        # You may need to convert the color.
        img_pil = np.ascontiguousarray(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        )
        if idx % self.in_fps  == 0:
            ic(idx)

        annotator = Annotator((img_pil))
        for *box, confidence, cls in predictions:
            # ic(cls, confidence, results.names[int(cls)])
            label = f"{results.names[int(cls)]} {confidence:.2f}"
            annotator.box_label(box, label, color=colors(cls))

        # For reversing the operation:
        im_np = np.asarray(annotator.im)
        self.output_unique.write(im_np)
        self.output_unique_mask.write(im_np)


@app.command()
def Yolo(
    video_input_file: str = typer.Argument("in.mp4"), force: bool = typer.Option(False)
) -> None:
    """
    Remove background from Ring Video
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
    yolo = YoloProcessor(base_filename, 30)
    return cv_helper.process_video(input_video,yolo)

if __name__ == "__main__":
    app()
