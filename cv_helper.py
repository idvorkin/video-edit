import cv2
from contextlib import contextmanager
from imutils.video import FPS
from icecream import ic
import os
import typer
import numpy as np
from PIL import Image
from typing_extensions import Protocol
import threading
import queue


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

    def frame(self, idx: int, frame) -> None:
        pass


def process_video(input_video, frame_processor: FrameProcessor):
    width = input_video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    video_fps = input_video.get(cv2.CAP_PROP_FPS)
    frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    ic(width, height, video_fps, frame_count)

    # start the FPS timer
    fps = FPS().start()
    with typer.progressbar(
        length=100, label="Processing Video"
    ) as progress_bar, process_video_frames_context_manager(
        frame_processor, input_video
    ) as process_frame:
        for i, frame in enumerate(video_reader(input_video)):
            # Update UX counters
            fps.update()
            process_frame.frame(i, frame)
            if (i % int(frame_count / 100)) == 0:
                progress_bar.update(1)

    # stop the timer and display FPS information
    fps.stop()
    ic(int(fps.fps()), "Elapsed Seconds", int(fps.elapsed()))


def video_reader(input_video):
    while True:
        ret, frame = input_video.read()
        if not ret:
            return
        yield frame


class LazyVideoWriter:
    def __init__(self, name: str, fps: int):
        self.name = name
        self.height, self.width = 0, 0
        self.vw = None
        self.fps = fps
        self.frame_queue = queue.Queue()
        self.frame_available = (
            threading.Event()
        )  # Event to signal when a frame is added
        self.stop_thread = False
        self.write_thread = threading.Thread(target=self.process_queue)
        self.write_thread.start()

    def create(self, frame):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.width, self.height = int(frame.shape[1]), int(frame.shape[0])
        self.vw = cv2.VideoWriter(
            self.name, fourcc, self.fps, (self.width, self.height)
        )

    def write(self, frame):
        self.frame_queue.put(frame)
        self.frame_available.set()  # Signal that a new frame is available

    def process_queue(self):
        while not self.stop_thread or not self.frame_queue.empty():
            self.frame_available.wait()  # Wait for a frame to become available
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if self.vw is None:
                    self.create(frame)
                width, height = int(frame.shape[1]), int(frame.shape[0])
                assert (
                    width == self.width and height == self.height
                ), "Frame dimensions do not match."
                self.vw.write(frame)
                if self.frame_queue.empty():
                    self.frame_available.clear()  # Reset the event if no more frames are available

    def release(self):
        self.stop_thread = True
        self.frame_available.set()  # Ensure the thread wakes up to process the stop signal
        self.write_thread.join()
        if self.vw:
            self.vw.release()


# Use a lazy video writer so don't have to pass in an
# height/width, can read from first frame.


def PIL_to_open_cv(pil_img):
    as_cv = np.asarray(pil_img)  # I nee to change color spaces
    cv_fix_color = cv2.cvtColor(as_cv, cv2.COLOR_RGB2BGR)
    return cv_fix_color


def open_cv_to_PIL(frame):
    img_pil = np.ascontiguousarray(
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    )
    return img_pil


def write_text(image, text, origin, font_scale=1.0):
    # TODO, shift fonts if canvas is small
    color_white = (255, 255, 255)
    color_black = (0, 0, 0)
    x, y = origin

    count_lines = len(text.split("\n"))
    max_width = max([len(x) for x in text.split("\n")])
    font_height = 40

    rect_top_left = (x, y - 40 * font_scale)
    rect_bottom_right = (
        x + max_width * font_scale * 20,
        y + font_height * font_scale * count_lines,
    )

    # convert to ints
    rect_top_left = tuple(map(int, rect_top_left))
    rect_bottom_right = tuple(map(int, rect_bottom_right))

    # draw box around text
    image = cv2.rectangle(
        image,
        rect_top_left,
        rect_bottom_right,
        color_black,
        -1,
    )

    for line in text.split("\n"):
        image = cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color_white,
            2,
            cv2.LINE_AA,
        )
        y += font_height * font_scale
    return image


def scale_point_to_image(image, point):
    """
    Scales a normalized point (0 to 1) to the image dimensions.

    Args:
        image: The image to which the point will be scaled.
        point: A tuple representing the normalized (x, y) coordinates of the point.

    Returns:e
        A tuple representing the scaled (x, y) coordinates of the point.
    """

    # assert valid param types
    assert isinstance(image, np.ndarray)
    assert len(point) == 2
    # assert points between 0 and 1
    assert 0 <= point[0] <= 1
    assert 0 <= point[1] <= 1

    img_height, img_width = image.shape[:2]
    scaled_x = int(point[0] * img_width)
    scaled_y = int(point[1] * img_height)
    return (scaled_x, scaled_y)


# TODO: Would be cool if detected in what system you are (cli,cli/w/term,jupyter)
def display_jupyter(frame):
    from IPython.display import display, Image, clear_output

    _, jpg = cv2.imencode(".jpeg", frame)
    clear_output(True)
    display(Image(data=jpg.tobytes()))
