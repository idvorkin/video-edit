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


from imutils.video import FPS
from icecream import ic
from pendulum import duration
import cv2
import numpy as np
from dataclasses import dataclass
import typer

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

    transforms = [to_grayscale]
    for t in transforms:
        frame = t(frame)

    frame = analyze(state, frame)

    # Running 60 FPS input
    # Only skeletonize every second

    return frame


def shrink_image_half(src):
    if not isinstance(src, np.ndarray):
        return src

    # calculate the 50 percent of original dimensions
    scale_percent = 50
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dsize = (width, height)
    return cv2.resize(src, dsize)


def burn_in_debug_info(frame, state, count_non_zero, in_fps):
    cv2.putText(
        frame,
        f"{int(state.idx/in_fps)}:{state.idx}:{count_non_zero}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def main(input_file):
    # start the FPS timer
    fps = FPS().start()
    input_video = cv2.VideoCapture(input_file)
    ic(input_video.isOpened())
    state = FrameState(0, 0, 0, 0)

    width = input_video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    in_fps = input_video.get(cv2.CAP_PROP_FPS)  # float `height`
    frame_count = input_video.get(cv2.CAP_PROP_FRAME_COUNT)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_unique = cv2.VideoWriter(
        "output_unique.mp4", fourcc, in_fps, (int(width), int(height))
    )
    output_unique_mask = cv2.VideoWriter(
        "output_unique_masked.mp4", fourcc, in_fps, (int(width), int(height))
    )

    # Even black images have some noise
    # lets say
    percent_image_non_zero_still_blank = 0.1
    non_zero_pixels_in_black_image = int(
        0.01 * percent_image_non_zero_still_blank * width * height
    )
    ic(non_zero_pixels_in_black_image)

    with typer.progressbar(range(int(frame_count)), label="Processing Video") as frames:
        for frame in frames:  # Hack,
            fps.update()  # update FPS first so can continue early.

            ret, in_frame = input_video.read()
            state.idx = frame
            state.frame = np.copy(in_frame)
            assert ret

            frame = process_frame(state, in_frame)

            # only write output frame if mask is non_zero
            count_non_zero = np.count_nonzero(frame)

            if count_non_zero < non_zero_pixels_in_black_image:
                continue

            burn_in_debug_info(frame, state, count_non_zero, in_fps)
            burn_in_debug_info(in_frame, state, count_non_zero, in_fps)

            cv2.imshow("Input", shrink_image_half(in_frame))
            cv2.imshow("Mask", shrink_image_half(frame))

            masked_input = cv2.bitwise_and(in_frame, in_frame, mask=frame)
            cv2.imshow("MaskedInput", shrink_image_half(masked_input))
            cv2.waitKey(1)

            # ic (state.idx,count_non_zero)
            output_unique_mask.write(masked_input)
            output_unique.write(in_frame)

    # stop the timer and display FPS information
    fps.stop()
    ic(fps.elapsed())
    ic(fps.fps())
    ic(state.idx)

    # do a bit of cleanup
    cv2.destroyAllWindows()
    output_unique_mask.release()
    output_unique.release()


@app.command()
def RemoveBackground(video_input_file: str = typer.Argument("in.mp4")) -> None:
    """
    Remove background from Ring Video
    """
    ic(video_input_file)
    return main(video_input_file)


if __name__ == "__main__":
    app()
