import cv2
from contextlib import contextmanager
from imutils.video import FPS
from icecream import ic
import os
import typer
import numpy as np
from PIL import Image
from typing_extensions import Protocol


def cv2_video(path):
    input_video = cv2.VideoCapture(os.path.expanduser(path))
    if not input_video.isOpened():
        print(f"Unable to Open {path}")
        raise Exception(f"Unable to Open {path}")
    return input_video


# Make a context manager (since it's familiar)
# interface create(input_video),release(), frame(i,frame)
@contextmanager
def process_video_frames_context_manager(frame_processor, input_video):
    frame_processor.create(input_video)
    try:
        yield frame_processor
    finally:
        frame_processor.destroy()

class FrameProcessor(Protocol):
    def create(self, input_video) -> None:
        pass
    def destroy(self) -> None:
        pass
    def frame(self, idx:int, frame) -> None:
        pass



def process_video(input_video, frame_processor:FrameProcessor):
    width = input_video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    video_fps = input_video.get(cv2.CAP_PROP_FPS)
    frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    ic(width, height, video_fps, frame_count)

    # start the FPS timer
    fps = FPS().start()
    with typer.progressbar(
        length=frame_count, label="Processing Video"
    ) as progress_bar, process_video_frames_context_manager(frame_processor, input_video) as process_frame:
        for (i, frame) in enumerate(video_reader(input_video)):

            # Update UX counters
            fps.update()
            process_frame.frame(i, frame)
            progress_bar.update(1)

    # stop the timer and display FPS information
    fps.stop()
    ic(int(fps.fps()),"Elapsed Seconds", int(fps.elapsed()))


def video_reader(input_video):
    while True:
        ret, frame = input_video.read()
        if not ret:
            return
        yield frame


# Use a lazy video writer so don't have to pass in an
# height/width, can read from first frame.
class LazyVideoWriter:
    def __init__(self, name: str, fps: int):
        self.name = name
        self.height, self.width = 0, 0
        self.vw = None
        self.fps = fps

    def create(self, frame):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.width, self.height = int(frame.shape[1]), int(frame.shape[0])
        self.vw = cv2.VideoWriter(
            self.name, fourcc, self.fps, (self.width, self.height)
        )

    def write(self, frame):
        if self.vw is None:
            self.create(frame)
        width, height = int(frame.shape[1]), int(frame.shape[0])
        assert width == self.width
        assert height == self.height
        self.vw.write(frame)

    def release(self):
        if self.vw:
            self.vw.release()

def PIL_to_open_cv(pil_img):
    as_cv = np.asarray(pil_img) # I nee to change color spaces
    cv_fix_color = cv2.cvtColor(as_cv, cv2.COLOR_RGB2BGR)
    return cv_fix_color


def open_cv_to_PIL(frame):
    img_pil = np.ascontiguousarray(
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    )
    return img_pil


# TODO: Would be cool if detected in what system you are (cli,cli/w/term,jupyter)
def display_jupyter(frame):
    from IPython.display import display, Image, clear_output
    _, jpg = cv2.imencode(".jpeg", frame)
    clear_output(True)
    display(Image(data=jpg.tobytes()))
