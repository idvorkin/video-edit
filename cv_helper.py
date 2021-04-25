def video_reader(input_video):
    while True:
        ret, frame = input_video.read()
        if not ret:
            return
        yield frame
