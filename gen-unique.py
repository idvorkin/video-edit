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
from icecream import ic
import cv2
import time


def main():
    print("Hello")
    input_file = "in.mp4"
    out_file = "unique.mp4"
    fvs = FileVideoStream(input_file)
    # start the FPS timer
    fps = FPS().start()
    ic(input_file)
    ic(fps)
    ic(fvs)
    stream = cv2.VideoCapture(input_file)
    ic(stream.isOpened())
    count_frames = 0

    # loop over frames from the video file stream
    while True:
        count_frames += 1
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)
        ret, frame = stream.read()
        if not ret:
            break

        # show the frame and update the FPS counter
        cv2.imshow("Frame", frame)

        # Relocated filtering into producer thread with transform=filterFrame
        #  Python 2.7: FPS 92.11 -> 131.36
        #  Python 3.7: FPS 41.44 -> 50.11
        # frame = filterFrame(frame)

        # display the size of the queue on the frame
        cv2.putText(
            frame,
            "Queue Size: {}".format(fvs.Q.qsize()),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    ic(fps.elapsed())
    ic(fps.fps())
    ic(count_frames)

    # do a bit of cleanup
    cv2.destroyAllWindows()

main()
