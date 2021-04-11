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

from imutils.video import FileVideoStream, FPS
from imutils import skeletonize
from icecream import ic
from pendulum import duration
import cv2
import numpy as np
from dataclasses import dataclass
import copy


@dataclass
class FrameState:
    idx: any
    frame: any
    last_analysis_frame: any
    last_analysis_diff: any


analysis_fps = 10
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


def analyze(state: FrameState, frame):
    is_analysis_frame = state.idx % analysis_fps == 0
    if not is_analysis_frame:
        # to see input video, return frame when not analysis
        # return frame
        return state.last_analysis_diff

    is_first_frame = state.idx == 0
    initialize_state = is_first_frame
    if initialize_state:
        state.last_analysis_frame = frame
        return frame

    blur  = cv2.blur(frame, (5, 5))
    tmp = cv2.absdiff(blur, state.last_analysis_frame)
    state.last_analysis_frame =  blur


    # tmp = cv2.subtract(skeleton, state.last_analysis_frame)
    # tmp = np.clip(tmp, 0, None)  # bound array between 0 and None

    # Threshold
    tmp = cv2.adaptiveThreshold(
        tmp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
        # tmp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    tmp = cv2.dilate(tmp, None, iterations = 1)
    kernel = np.ones((200,200),np.uint8)
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, kernel)


    # Mask out the motion
    mask = cv2.bitwise_not(tmp)
    tmp = cv2.bitwise_and(state.frame,state.frame,mask=mask)

    state.last_analysis_diff = tmp
    return tmp


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


def main():
    print("Hello")
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
    state = FrameState(np.ones(1),0, 0, 0)

    name = 'output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(name, fourcc, 20.0, (1920,1200))


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
            f"{state.idx}:{state.last_analysis_diff}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"{state.idx}:{state.last_analysis_diff}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

        def resize(src):
            if not isinstance(src, np.ndarray):
                return src

            #calculate the 50 percent of original dimensions
            scale_percent=50
            width = int(src.shape[1] * scale_percent / 100)
            height = int(src.shape[0] * scale_percent / 100)
            dsize = (width, height)
            return cv2.resize(src, dsize)

        cv2.imshow("Input", resize(in_frame))
        cv2.imshow("Frame", resize(frame))
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


main()
