import cv2 as cv
import os


class DataSaver:
    def __init__(self):
        self.output_path = "output_videos"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.writer = None

    def initialize_writer(self, frame_size, fps):
        """Initialize video writer."""
        self.writer = cv.VideoWriter(
            os.path.join(self.output_path, "output.mp4"),
            cv.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size
        )

    def write_frame(self, frame):
        """Write a frame to the video."""
        if self.writer:
            self.writer.write(frame)

    def finalize_writer(self):
        """Release the video writer."""
        if self.writer:
            self.writer.release()
            self.writer = None
