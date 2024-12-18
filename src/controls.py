import threading
import cv2 as cv
from PIL import Image, ImageTk


class AppControls:
    def __init__(self):
        self.running = False
        self.eyes_visible = False
        self.thread = None
        self.processor = None
        self.saver = None
        self.canvas = None
        self.image_on_canvas = None

    def set_dependencies(self, processor, saver):
        """Set dependencies for the controls."""
        self.processor = processor
        self.saver = saver

    def start_processing(self, canvas, image_on_canvas):
        """Start video processing thread."""
        self.running = True
        self.canvas = canvas
        self.image_on_canvas = image_on_canvas
        self.thread = threading.Thread(target=self._process_video, daemon=True)
        self.thread.start()

    def stop_processing(self):
        """Stop video processing thread."""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=1)
            self.thread = None

    def _process_video(self):
        """Video processing logic for real-time frame processing."""
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            self.running = False
            return

        # Initialize video writer for saving processed output
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS) or 30  # Default to 30 FPS if unavailable
        self.saver.initialize_writer((frame_width, frame_height), fps)

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                # Process the frame
                processed_frame = self.processor.process_frame(frame)

                # Save the processed frame
                self.saver.write_frame(processed_frame)

                # Resize the frame to fit the canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                processed_frame = self._resize_frame_to_canvas(processed_frame, canvas_width, canvas_height)

                # Convert the resized frame to RGB
                processed_frame = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)

                # Convert to PhotoImage and update the canvas
                image = Image.fromarray(processed_frame)
                self.photo = ImageTk.PhotoImage(image=image)

                # Display the image in the Tkinter canvas
                self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
                self.canvas.photo = self.photo  # Keep a reference to prevent garbage collection

        except RuntimeError as e:
            print(f"Runtime error: {e}")
        finally:
            # Release resources
            cap.release()
            self.saver.finalize_writer()
            self.canvas.delete(self.image_on_canvas)
            cv.destroyAllWindows()

    def _resize_frame_to_canvas(self, frame, canvas_width, canvas_height):
        """Resize the frame to fill the canvas while maintaining aspect ratio."""
        frame_height, frame_width = frame.shape[:2]
        scale = max(canvas_width / frame_width, canvas_height / frame_height)

        resized_width = int(frame_width * scale)
        resized_height = int(frame_height * scale)
        resized_frame = cv.resize(frame, (resized_width, resized_height), interpolation=cv.INTER_LINEAR)

        # Crop the center of the resized frame to fit the canvas exactly
        crop_x = (resized_width - canvas_width) // 2
        crop_y = (resized_height - canvas_height) // 2
        cropped_frame = resized_frame[crop_y:crop_y + canvas_height, crop_x:crop_x + canvas_width]

        return cropped_frame
