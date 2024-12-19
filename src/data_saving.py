import cv2 as cv


class DataSaver:
    def __init__(self):
        self.writer = None

    def initialize_writer(self, frame_size, fps, output_path):
        """Initialize the video writer with a specific output path."""
        self.writer = cv.VideoWriter(
            output_path,
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
