import cv2

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
        if self.vw == None:
            self.create(frame)
        width, height = int(frame.shape[1]), int(frame.shape[0])
        assert width == self.width
        assert height == self.height
        self.vw.write(frame)

    def release(self):
        if self.vw:
            self.vw.release()
