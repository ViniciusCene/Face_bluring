import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import cv2 as cv
import threading
import os


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

        # Video variables
        self.cap = None
        self.model = None
        self.running = False
        self.conf_threshold = 0.45  # Default confidence threshold
        self.video_writer = None

        # GUI Components
        # Canvas for video display
        self.canvas = ttk.Canvas(root, width=640, height=480)
        self.canvas.pack(fill="both", expand=True)

        # Start/Stop Button with enhanced styling
        self.start_button = ttk.Button(
            root, text="Start", command=self.toggle_video_processing,
            bootstyle="success-outline", padding=10, width=20
        )
        self.start_button.pack(pady=10)

        # Confidence Threshold Input
        self.threshold_label = ttk.Label(root, text="Confidence Threshold (0.1 - 1):")
        self.threshold_label.pack(pady=5)
        self.threshold_input = ttk.Entry(root, bootstyle="info", width=10)
        self.threshold_input.insert(0, str(self.conf_threshold))  # Set default value
        self.threshold_input.pack(pady=5)

        # Validate threshold input
        self.threshold_input.bind("<Return>", self.update_threshold)

        self.video_thread = None

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

    def update_threshold(self, event=None):
        """
        Updates the confidence threshold for the YuNet model from user input.

        Parameters
        ----------
        event : Event, optional
            The event that triggered this function (e.g., pressing Enter).
        """
        try:
            value = float(self.threshold_input.get())
            if 0.1 <= value <= 1.0:
                self.conf_threshold = value
                if self.model:
                    self.model.setConfThreshold(self.conf_threshold)
            else:
                self.show_error("Value must be between 0.1 and 1.")
        except ValueError:
            self.show_error("Invalid input. Enter a number between 0.1 and 1.")

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

        # Initialize the YuNet model
        self.model = self.load_yunet_model()

        # Configure video writer for saving output
        fourcc = cv.VideoWriter_fourcc(*"XVID")
        frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.video_writer = cv.VideoWriter(
            "output_blurred.avi", fourcc, 20.0, (frame_width, frame_height)
        )

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Resize frame to match model input size
            input_size = (320, 320)
            resized_frame = cv.resize(frame, input_size)

            # Run inference
            results = self.model.infer(resized_frame)

            # Resize results back to original frame dimensions
            scale_x = frame.shape[1] / input_size[0]
            scale_y = frame.shape[0] / input_size[1]
            if results is not None and results.size > 0:
                results[:, :4] *= [scale_x, scale_y, scale_x, scale_y]

            # Visualize results
            processed_frame = self.visualize(frame, results)

            # Write processed frame to the video file
            self.video_writer.write(processed_frame)

            # Convert BGR to RGB for proper color display
            frame_rgb = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)

            # Convert to PIL Image and then to ImageTk
            image = Image.fromarray(frame_rgb)
            image_tk = ImageTk.PhotoImage(image=image)

            # Display the image in the Tkinter canvas
            self.canvas.create_image(0, 0, anchor="nw", image=image_tk)
            self.canvas.image_tk = image_tk  # Keep a reference to prevent garbage collection
            self.root.update_idletasks()

        self.cap.release()

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


def main():
    """
    Main function to start the GUI.
    """
    root = ttk.Window(themename="flatly")  # Choose a clean theme
    app = YuNetBlurGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
