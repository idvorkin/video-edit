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


# TODOs
# There are a few inter frame artifacts to clean up, consider 2 pass
# 1 pass build the masks
# 2 clean up things we can detect inter frame
# E.g. 10s motion -> 2s no_motion -> 10s motion, should assume motion in between


from cv_helper import *
from imutils.video import FPS
from icecream import ic
import cv2
import numpy as np
from dataclasses import dataclass
import typer
import os.path

app = typer.Typer()

backSub = cv2.createBackgroundSubtractorMOG2(history=120)
# backSub = cv2.createBackgroundSubtractorKNN()

# Some globals
color_black = 0
color_white = 255
color_grey = 127
fill_rectangle_thickness = -1


@dataclass
class FrameState:
    idx: any
    last_fg_mask: any


# Input is 20FPS, analyze every 1/2 second
# TBD compute this.

analysis_key_frame_rate = 10


def obsolete_handled_by_bg_remover_remove_ring_timestamp(frame):
    max_y, max_x = frame.shape[0], frame.shape[1]
    start_point = (0 + int(max_x * 0.7), max_y - 100)
    bottom_right = max_x, max_y
    end_point = bottom_right
    color = (0, 0, 0)
    thickness = fill_rectangle_thickness
    return cv2.rectangle(frame, start_point, end_point, color, thickness)


def to_black_and_white(frame):
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

    frame = to_black_and_white(frame)

    # Remove artifact noise, 10x10 seems like plenty
    frame = cv2.erode(frame, square_kernel(10))

    # Without noise, can dialate (fill in) with a decent kernel size.
    frame = cv2.dilate(frame, square_kernel(40))

    contouring_method = cv2.RETR_EXTERNAL  # Only outer edges
    # contouring_method =  cv2.RETR_CCOMP # 2 level outer, then inner
    # contouring_method =  cv2.RETR_FLOODFILL # Fill it in

    contours, hierachy = cv2.findContours(
        frame, contouring_method, cv2.CHAIN_APPROX_NONE
    )

    # Draw in the found contours
    draw_all_counters = -1
    contour_color = color_grey  # Black
    contour_thickness_fill = -1
    good_contours = [c for c in contours if cv2.contourArea(c) > 100]
    frame = cv2.drawContours(
        frame, good_contours, draw_all_counters, contour_color, contour_thickness_fill
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
        return state.last_fg_mask

    fgMask = backSub.apply(frame)
    fgMask = to_contours(fgMask)
    state.last_fg_mask = fgMask
    return state.last_fg_mask

def create_analyze_debug_frame(frame, motion_mask):
    masked_input = cv2.bitwise_and(frame, frame, mask=motion_mask)
    mask_3c = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
    top_row = np.concatenate((frame, mask_3c), axis=1)
    bottom_row = np.concatenate((mask_3c, masked_input), axis=1)
    merge_image = np.concatenate((top_row, bottom_row), axis=0)
    return merge_image


# Gray Scale Frame
def to_grayscale(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def to_blur(f):
    return cv2.blur(f, (20, 20))


def process_frame(state: FrameState, frame):

    transforms = [to_grayscale]
    for t in transforms:
        frame = t(frame)

    return analyze(state, frame)


def shrink_image_half(src):
    if not isinstance(src, np.ndarray):
        return src

    # calculate the 50 percent of original dimensions
    scale_percent = 50
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dsize = (width, height)
    return cv2.resize(src, dsize)


def burn_in_debug_info(frame, idx, in_fps):
    cv2.rectangle(frame, (0,0), (650,70), color_black, fill_rectangle_thickness)
    cv2.putText(
        frame,
        f"{int(idx/in_fps)}:{idx}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        2,
    )


def is_frame_black(frame):
    # Even mostly black images have some noise, set a threshold
    percent_image_non_zero_still_black = 0.1
    total_pixels = frame.shape[0] * frame.shape[1]
    non_zero_pixels_in_black_image = int(
        0.01 * percent_image_non_zero_still_black * total_pixels
    )
    count_non_zero = np.count_nonzero(frame)
    return count_non_zero < non_zero_pixels_in_black_image


def process_video(base_filename: str, input_video):

    state = FrameState(0, 0)

    width = input_video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    in_fps = input_video.get(cv2.CAP_PROP_FPS)  # float `height`
    frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    ic(width, height, in_fps, frame_count)

    unique_filename = f"{base_filename}_unique.mp4"
    output_unique = LazyVideoWriter(unique_filename, in_fps)
    mask_filename = f"{base_filename}_mask.mp4"
    output_unique_mask = LazyVideoWriter(mask_filename, in_fps)
    output_video_files = [output_unique, output_unique_mask]
    debug_window_refresh_rate = 10

    # start the FPS timer
    fps = FPS().start()
    with typer.progressbar(length=frame_count, label="Processing Video") as progress:
        for (idx, original_frame) in enumerate(video_reader(input_video)):

            # Update UX counters
            fps.update()
            progress.update(1)
            state.idx = idx

            # PERF: Processing at 1/4 size boosts FPS by TK%
            in_frame = shrink_image_half(original_frame)
            motion_mask = process_frame(state, in_frame)

            # only write output frame if frame is not black
            if is_frame_black(motion_mask):
                continue

            # PERF - show_debug_window at lower rate boosts FPS by TK%
            if idx % debug_window_refresh_rate == 0:
                debug_frame = create_analyze_debug_frame(in_frame, motion_mask)
                burn_in_debug_info(debug_frame, idx, in_fps)
                cv2.imshow(f"{base_filename} Input", shrink_image_half(debug_frame))
                cv2.waitKey(1)

            output_unique.write(original_frame)

            # masked_input = cv2.bitwise_and(in_frame, in_frame, mask=motion_mask)
            # output_unique_mask.write(masked_input)

    # stop the timer and display FPS information
    fps.stop()
    ic(int(fps.fps()), int(fps.elapsed()))

    cv2.destroyAllWindows()
    for f in output_video_files:
        f.release()


@app.command()
def RemoveBackground(
    video_input_file: str = typer.Argument("in.mp4"), force:bool=typer.Option(False)
) -> None:
    """
    Remove background from Ring Video
    """
    base_filename = video_input_file.split(".")[0]
    # if file exists, skip it.
    unique_filename = f"{base_filename}_unique.mp4"

    if not force and os.path.exists(unique_filename):
        print(f"{unique_filename} exists, skipping")
        return

    input_video = cv2.VideoCapture(video_input_file)
    if not input_video.isOpened():
        return f"Unable to Open {video_input_file} "

    return process_video(base_filename, input_video)


if __name__ == "__main__":
    app()