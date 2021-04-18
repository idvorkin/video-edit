# gen-unique-video from video with lots of changeless frames
# My use case - I want to look at my ring videos, but skip the parts where nothing changes.
# When ring creates a motion video, it starts 30s before motion, and often ends with 30s
# no motion. E.g <same-20s>-<motion-20s>-<same-20s>
# So, my goal is trim the video  down to <same-2s>-<motion-20s><same-2s>

# Random Libraries
# pose-detectin: https://github.com/CMU-Perceptual-Computing-Lab/openpose
# easier open CV: https://github.com/jrosebr1/imutils
# Sci-Kit Image: https://scikit-image.org/
# Python Motion Detecor: https://www.geeksforgeeks.org/webcam-motion-detector-python/


#######
# Higher level primitive being used - Background substituion
############
#   - https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html

from imutils.video import FileVideoStream, FPS
from imutils import skeletonize
from icecream import ic
from pendulum import duration
import cv2
import numpy as np
from dataclasses import dataclass
import copy
from typing import Optional
import typer
from pathlib import Path

app = typer.Typer()


backSub = cv2.createBackgroundSubtractorMOG2(history=120)
# backSub = cv2.createBackgroundSubtractorKNN()

# Some globals
color_black = 0
color_white = 255
color_grey = 127


@dataclass
class FrameState:
    idx: any
    frame: any
    last_analysis_frame: any
    last_fg_mask: any


analysis_key_frame_rate = 15
output_fps = 1000


def remove_ring_timestamp(frame):
    max_y, max_x = frame.shape[0], frame.shape[1]
    start_point = (0 + int(max_x * 0.7), max_y - 100)
    bottom_right = max_x, max_y
    end_point = bottom_right
    color = (0, 0, 0)
    fill_rectangle_thickness = -1
    thickness = fill_rectangle_thickness

    return cv2.rectangle(frame, start_point, end_point, color, thickness)


def threshold(frame):
    color_threshold = 127
    convert_to_color = color_white

    ## QQ: Is this doing anything if I've already bg subtracted - not so sure!
    ## I guess I can histogram to find out?
    ret, threshed = cv2.threshold(
        frame,
        color_threshold,
        convert_to_color,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    frame = threshed

    # findContours needs image to contour to be white, so invert threshold
    # Should be able to do in cv2.threshold, but feeling lazy.

    return cv2.bitwise_not(frame)


def square_kernel(side):
    return np.ones((side, side), np.uint8)


def to_contours(frame):

    frame = threshold(frame)

    # Remove artifact noise, 10x10 seems like plenty
    erode_kernel = square_kernel(10)
    frame = cv2.erode(frame, square_kernel(10))

    # Without noise, can dialate a fair bit
    frame = cv2.dilate(frame, square_kernel(40))

    contouring_method = cv2.RETR_EXTERNAL  # Only outer edges
    # contouring_method =  cv2.RETR_CCOMP # 2 level outer, then inner
    # contouring_method =  cv2.RETR_FLOODFILL # Fill it in

    contours, hierachy = cv2.findContours(
        frame, contouring_method, cv2.CHAIN_APPROX_NONE
    )

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 100:
            continue
        # ic(i)

        ## (4) Create mask and do bitwise-op
        contour_color = color_grey  # Black
        contour_index = 0  # drawCounters takes an array and index as input
        contour_thickness_fill = -1
        frame = cv2.drawContours(
            frame, [contour], contour_index, contour_color, contour_thickness_fill
        )

    # Dialate again to try and fill holes
    dialate_kernel = square_kernel(200)
    frame = cv2.dilate(frame, dialate_kernel)

    return frame


def analyze(state: FrameState, frame):
    is_first_frame = state.idx < 2
    is_analysis_frame = state.idx % analysis_key_frame_rate == 0
    is_do_analyze = is_analysis_frame or is_first_frame
    if not is_do_analyze:
        # to see input video, return frame when not analysis
        # return frame
        return state.last_fg_mask

    fgMask = backSub.apply(frame)
    fgMask = to_contours(fgMask)
    state.last_fg_mask = fgMask
    return state.last_fg_mask


# Gray Scale Frame
def to_grayscale(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def to_blur(f):
    return cv2.blur(f, (20, 20))


def process_frame(state: FrameState, frame):

    transforms = [remove_ring_timestamp, to_grayscale]
    for t in transforms:
        frame = t(frame)

    frame = analyze(state, frame)

    # Running 60 FPS input
    # Only skeletonize every second

    return frame


def main(input_file):
    if input_file == None:
        input_file = "in.mp4"

    # out_file = "unique.mp4"
    fvs = FileVideoStream(input_file)
    # start the FPS timer
    fps = FPS().start()
    ic(input_file)
    ic(fps)
    ic(fvs)
    stream = cv2.VideoCapture(input_file)
    ic(stream.isOpened())
    state = FrameState(0, 0, 0, 0)

    name = "output.mov"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(name, fourcc, 20.0, (1920, 1200))

    while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)
        ret, in_frame = stream.read()
        state.idx += 1
        state.frame = np.copy(in_frame)
        if not ret:
            break

        frame = process_frame(state, in_frame)

        cv2.putText(
            in_frame,
            f"{state.idx}:{state.last_fg_mask}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"{state.idx}:{state.last_fg_mask}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        def resize(src):
            if not isinstance(src, np.ndarray):
                return src

            # calculate the 50 percent of original dimensions
            scale_percent = 50
            width = int(src.shape[1] * scale_percent / 100)
            height = int(src.shape[0] * scale_percent / 100)
            dsize = (width, height)
            return cv2.resize(src, dsize)

        cv2.imshow("Input", resize(in_frame))
        cv2.imshow("Mask", resize(frame))
        cv2.imshow(
            "MaskedInput", resize(cv2.bitwise_and(in_frame, in_frame, mask=frame))
        )
        output.write(in_frame)
        display_duration_ms = int(
            duration(seconds=1 / output_fps).total_seconds() * 1000
        )
        cv2.waitKey(display_duration_ms)
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    ic(fps.elapsed())
    ic(fps.fps())
    ic(state.idx)

    # do a bit of cleanup
    cv2.destroyAllWindows()
    output.release()

@app.command()
def RemoveBackground(video_input_file: Optional[Path] = typer.Argument(None)) -> None:
    """
    Remove background from Ring Video
    """
    return main(video_input_file)

if __name__ == "__main__":
    app()
