#!python3
# gen-unique-video from video with lots of changeless frames


import torch
from plots import Annotator, colors

import cv_helper
from icecream import ic
import cv2
import typer
import os.path

app = typer.Typer()


class YoloProcessor:
    def __init__(self, base_filename):
        self.base_filename = base_filename
        pass

    def create(self, input_video):
        self.video = input_video
        self.fps = input_video.get(cv2.CAP_PROP_FPS)
        self.yolo = torch.hub.load(
            "ultralytics/yolov5", "yolov5s"
        )  # or yolov5m, yolov5l, yolov5x, custom
        self.yolo_filename = f"{self.base_filename}_yolo.mp4"
        self.yolo_writer = cv_helper.LazyVideoWriter(
            self.yolo_filename, self.fps
        )
        self.output_video_files = [self.yolo_writer]
        self.results = None # cache this from previous runs
        self.update_freq = int(self.fps/2) # every 0.5 seconds update bounding box

    def destroy(self):
        cv2.destroyAllWindows()
        for f in self.output_video_files:
            f.release()

    def frame(self, idx, frame):

        # results don't move so frequently that we need to re-yolo
        # on each frame, so just do every 500ms

        if idx%self.update_freq == 0:
            self.results = self.yolo(frame)

        predictions = self.results.pred[0]

        is_jupyter = False
        if is_jupyter:
            # output on jupyter every second
            if idx % self.fps == 0:
                ic(idx)

        # pytorch needs to be in PIL format
        frame_PIL = cv_helper.open_cv_to_PIL(frame)
        annotator = Annotator((frame_PIL))
        for *box, confidence, cls in predictions:
            label = f"{self.results.names[int(cls)]} {confidence:.2f}"
            annotator.box_label(box, label, color=colors(cls))

        # PIL to opencv
        self.yolo_writer.write(cv_helper.PIL_to_open_cv(annotator.im))


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
    yolo = YoloProcessor(base_filename)
    return cv_helper.process_video(input_video, yolo)


if __name__ == "__main__":
    app()
