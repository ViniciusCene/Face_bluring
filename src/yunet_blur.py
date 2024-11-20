import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import cv2 as cv
import threading
import os
import time


class YuNetBlurGUI:
    def __init__(self, root):
        """
        Initializes the GUI and its components.

        Parameters
        ----------
        root : ttk.Window
            The root window of the ttkbootstrap GUI.
        """
        self.root = root
        self.root.title("YuNet Automatic Face Blurring")

        # Set the window to full-screen mode
        self.root.attributes('-fullscreen', True)

        # Video variables
        self.cap = None
        self.model = None
        self.running = False
        self.conf_threshold = 0.45  # Default confidence threshold
        self.video_writer = None

        # Get screen size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Reserve space for the controls (buttons, slider, etc.)
        control_height = 100  # Adjust this value if needed
        self.canvas_width = int(screen_width * 0.9)
        self.canvas_height = int((screen_height - control_height) * 0.9)

        # GUI Components
        # Frame for canvas with padding and borders
        self.canvas_frame = ttk.Frame(
            root, padding=10, bootstyle="secondary"  # Add padding and style
        )
        self.canvas_frame.pack(fill="both", expand=False, pady=10)

        # Canvas for video display
        self.canvas = ttk.Canvas(
            self.canvas_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            relief="solid",  # Add border to the canvas
            borderwidth=2,
        )
        self.canvas.pack()

        # Start/Stop Button with enhanced styling
        self.start_button = ttk.Button(
            root, text="Start", command=self.toggle_video_processing,
            bootstyle="success-outline", padding=10, width=20
        )
        self.start_button.pack(pady=10)

        # Confidence Threshold Slider
        self.threshold_label = ttk.Label(root, text="Confidence Threshold:")
        self.threshold_label.pack(pady=5)
        self.threshold_slider = ttk.Scale(
            root, from_=0.1, to=1.0, value=self.conf_threshold, length=400,
            command=self.update_threshold, orient=HORIZONTAL, bootstyle="info"
        )
        self.threshold_slider.pack(pady=5)

        # Real-time numeric display for threshold value
        self.threshold_value_label = ttk.Label(
            root, text=f"Current Value: {self.conf_threshold:.2f}", bootstyle="info"
        )
        self.threshold_value_label.pack(pady=5)

        self.video_thread = None

        # Bind Esc key to exit full-screen mode
        self.root.bind("<Escape>", self.exit_fullscreen)

    def toggle_video_processing(self):
        """
        Starts or stops the video processing based on the current state.
        """
        if self.running:
            self.running = False
            self.start_button.config(text="Start", bootstyle="success-outline")
            if self.video_writer:
                self.video_writer.release()
        else:
            self.running = True
            self.start_button.config(text="Stop", bootstyle="danger-outline")
            self.video_thread = threading.Thread(target=self.video_processing)
            self.video_thread.start()

    def update_threshold(self, value):
        """
        Updates the confidence threshold for the YuNet model dynamically.

        Parameters
        ----------
        value : str
            The new threshold value as a string from the slider.
        """
        self.conf_threshold = float(value)
        self.threshold_value_label.config(text=f"Current Threshold Value: {self.conf_threshold:.2f}")
        if self.model:
            self.model._confThreshold = self.conf_threshold

    def show_error(self, message):
        """
        Displays an error message in a popup window.

        Parameters
        ----------
        message : str
            The error message to display.
        """
        error_window = ttk.Window(themename="flatly")
        error_window.title("Error")
        error_label = ttk.Label(error_window, text=message, bootstyle="danger")
        error_label.pack(pady=10)
        ttk.Button(error_window, text="OK", command=error_window.destroy).pack(pady=5)
        error_window.mainloop()

    def video_processing(self):
        """
        Handles real-time video processing and updates the GUI.
        """
        self.cap = cv.VideoCapture(0)

        # Measure frame processing times
        frame_times = []

        # Placeholder for video writer and dynamic FPS
        fourcc = cv.VideoWriter_fourcc(*"XVID")
        self.video_writer = None
        actual_fps = None

        while self.running:
            start_time = time.time()  # Start time of frame processing

            ret, frame = self.cap.read()
            if not ret:
                break

            # Dynamically reinitialize the YuNet model with the updated threshold
            self.model = self.load_yunet_model()

            # Resize frame to match the canvas dimensions
            resized_frame = cv.resize(frame, (self.canvas_width, self.canvas_height))

            # Run inference
            results = self.model.infer(cv.resize(frame, (320, 320)))

            # Resize results back to the resized frame dimensions
            scale_x = resized_frame.shape[1] / 320
            scale_y = resized_frame.shape[0] / 320
            if results is not None and results.size > 0:
                results[:, :4] *= [scale_x, scale_y, scale_x, scale_y]

            # Visualize results
            processed_frame = self.visualize(resized_frame, results)

            # Initialize the video writer with the actual FPS (after a few frames)
            if self.video_writer is None and len(frame_times) > 10:
                actual_fps = 1 / (sum(frame_times) / len(frame_times))  # Average FPS
                actual_fps = min(max(int(actual_fps), 1), 30)  # Clamp FPS
                self.video_writer = cv.VideoWriter(
                    "output_blurred.avi", fourcc, actual_fps,
                    (self.canvas_width, self.canvas_height)
                )

            # Write processed frame to the video file
            if self.video_writer:
                self.video_writer.write(processed_frame)

            # Convert BGR to RGB for proper color display
            frame_rgb = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)

            # Convert to PIL Image and then to ImageTk
            image = Image.fromarray(frame_rgb)
            image_tk = ImageTk.PhotoImage(image=image)

            # Display the image in the Tkinter canvas
            self.canvas.create_image(0, 0, anchor="nw", image=image_tk)
            self.canvas.image_tk = image_tk  # Keep a reference to prevent garbage collection

            # Calculate frame processing time
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            if len(frame_times) > 100:  # Keep the last 100 frame times
                frame_times.pop(0)

        self.cap.release()
        if self.video_writer:
            self.video_writer.release()  # Ensure the writer is properly closed

    def load_yunet_model(self):
        """
        Loads the YuNet model with the current confidence threshold.

        Returns
        -------
        YuNet
            The YuNet model instance.
        """
        model_path = os.path.join(
            os.path.dirname(__file__),
            "trained_models/yunet/face_detection_yunet_2023mar.onnx"
        )
        from yunet import YuNet
        return YuNet(
            modelPath=model_path, inputSize=[320, 320], confThreshold=self.conf_threshold,
            nmsThreshold=0.3, topK=5000
        )

    def visualize(self, image, results):
        """
        Visualizes the detected faces with blurring on the image.

        Parameters
        ----------
        image : np.array
            The input image.
        results : list
            The face detection results.

        Returns
        -------
        np.array
            The processed image with blurred faces.
        """
        if results is None or results.size == 0:
            return image

        height, width = image.shape[:2]

        for det in results:
            # Extract bounding box and scale it
            roi_x0, roi_y0, roi_w, roi_h = det[:4].astype(int)
            x, y, w, h = self.scale_roi(roi_x0, roi_y0, roi_w, roi_h, ratio=1.5)

            # Ensure ROI is within frame bounds
            x = max(0, x)
            y = max(0, y)
            w = min(width - x, w)
            h = min(height - y, h)

            if w <= 0 or h <= 0:
                # Skip invalid or out-of-bounds ROI
                continue

            # Apply Gaussian blur to the valid ROI
            image[y:y+h, x:x+w] = cv.GaussianBlur(image[y:y+h, x:x+w], (99, 99), 30)

        return image

    def scale_roi(self, x, y, w, h, ratio=1.0):
        """
        Scales the ROI for blurring.

        Parameters
        ----------
        x, y : int
            The top-left corner of the bounding box.
        w, h : int
            The width and height of the bounding box.
        ratio : float
            The scaling ratio.

        Returns
        -------
        tuple
            The scaled bounding box as (x, y, w, h).
        """
        center_x, center_y = x + w // 2, y + h // 2
        new_w, new_h = int(w * ratio), int(h * ratio)
        new_x, new_y = center_x - new_w // 2, center_y - new_h // 2
        return new_x, new_y, new_w, new_h

    def exit_fullscreen(self, event=None):
        """
        Exits full-screen mode.
        """
        self.root.attributes('-fullscreen', False)


def main():
    """
    Main function to start the GUI.
    """
    root = ttk.Window(themename="flatly")  # Choose a clean theme
    app = YuNetBlurGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
