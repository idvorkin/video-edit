#!python3
# gen-unique-video from video with lots of changeless frames


from ultralytics import YOLO

import cv_helper
from icecream import ic
import cv2
import typer
import os.path
import pose_helper

app = typer.Typer()


class YoloProcessor:
    def __init__(
        self,
        base_filename,
        update_fps_ratio=0.0,
        people_only=False,
        trim_no_people=False,
    ):
        self.base_filename = base_filename
        self.update_fps_ratio = update_fps_ratio
        self.people_only = people_only
        self.last_person_frame = 0
        self.seconds_to_lag_people_motion = 5
        self.trim_no_people = trim_no_people

    def create(self, input_video):
        self.video = input_video
        self.fps = input_video.get(cv2.CAP_PROP_FPS)
        # self.yolo = YOLO('yolov8n-seg.pt')  # pretrained YOLOv8n model
        self.yolo = YOLO("yolov8n-pose.pt")  # pretrained YOLOv8n model
        self.yolo_filename = f"{self.base_filename}_yolo.mp4"
        self.yolo_writer = cv_helper.LazyVideoWriter(self.yolo_filename, self.fps)
        self.output_video_files = [self.yolo_writer]
        self.results = None  # cache this from previous runs
        self.update_freq = 1  # default update_frequency is every frame
        if self.update_fps_ratio > 0:
            self.update_freq = int(self.fps * self.update_fps_ratio)

    def destroy(self):
        cv2.destroyAllWindows()
        for f in self.output_video_files:
            f.release()

    def frame(self, idx, frame):
        # results don't move so frequently that we need to re-yolo
        # on each frame, so just do every 500ms

        if idx % self.update_freq == 0:
            self.results = self.yolo(frame)

        is_jupyter = False
        if is_jupyter:
            # output on jupyter every second
            if idx % self.fps == 0:
                ic(idx)

        # pytorch needs to be in PIL format
        write_frame = True

        if self.trim_no_people:
            no_people_motion_for_threshold = (
                idx
                > self.last_person_frame + self.fps * self.seconds_to_lag_people_motion
            )
            if no_people_motion_for_threshold:
                write_frame = False

        if write_frame:
            base_image = frame  # self.results[0].plot()
            base_image = pose_helper.add_pose(base_image, base_image)
            self.yolo_writer.write(base_image)


@app.command()
def Yolo(
    video_input_file: str = typer.Argument("in.mp4"),
    force: bool = typer.Option(False),
    fps_ratio: float = typer.Option(0.5),
    people_only: bool = typer.Option(True),
    trim_no_people: bool = typer.Option(False),
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
    yolo = YoloProcessor(
        base_filename,
        people_only=people_only,
        update_fps_ratio=fps_ratio,
        trim_no_people=trim_no_people,
    )
    return cv_helper.process_video(input_video, yolo)


if __name__ == "__main__":
    app()
