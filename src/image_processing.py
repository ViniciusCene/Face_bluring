import os
import cv2 as cv
import numpy as np


class ImageProcessor:
    def __init__(self, controls):
        self.controls = controls
        self.confidence_threshold = 0.45  # Default confidence threshold
        self.blur_intensity = 10         # Default blur intensity
        self.blur_area = 120             # Default blur area (percentage)
        self.face_detector = self._load_model()

    def _load_model(self):
        """Load the YuNet face detection model."""
        model_path = os.path.join(
            os.path.dirname(__file__),
            "trained_models/yunet/face_detection_yunet_2023mar.onnx"
        )
        return cv.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=self.confidence_threshold,
            nms_threshold=0.3,
            top_k=5000
        )

    def update_confidence_threshold(self, value):
        """
        Update the confidence threshold dynamically.
        Parameters
        ----------
        value : float
            The new confidence threshold.
        """
        self.confidence_threshold = float(value)
        self.face_detector.setScoreThreshold(self.confidence_threshold)

    def process_frame(self, frame):
        """
        Apply face detection and optionally blur or unblur regions of interest.

        Parameters
        ----------
        frame : np.ndarray
            The input video frame.

        Returns
        -------
        np.ndarray
            The processed video frame.
        """
        input_size = (320, 320)
        resized_frame = cv.resize(frame, input_size)
        self.face_detector.setInputSize(input_size)

        results = self.face_detector.detect(resized_frame)
        if results is None or results[1] is None:
            return frame

        # Scale factors for mapping detected coordinates back to the original frame
        scale_x = frame.shape[1] / input_size[0]
        scale_y = frame.shape[0] / input_size[1]

        for det in results[1]:
            x, y, w, h = map(int, det[:4])
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)

            # Scale the ROI based on blur_area percentage
            scale_factor = self.blur_area / 100.0
            scaled_w = int(w * scale_factor)
            scaled_h = int(h * scale_factor)
            scaled_x = max(0, x - (scaled_w - w) // 2)
            scaled_y = max(0, y - (scaled_h - h) // 2)

            scaled_x = min(frame.shape[1] - 1, scaled_x)
            scaled_y = min(frame.shape[0] - 1, scaled_y)
            scaled_w = min(frame.shape[1] - scaled_x, scaled_w)
            scaled_h = min(frame.shape[0] - scaled_y, scaled_h)

            roi = frame[scaled_y:scaled_y + scaled_h, scaled_x:scaled_x + scaled_w]

            if roi.size > 0:
                kernel_size = max(1, self.blur_intensity * 2 + 1)
                blurred = cv.GaussianBlur(roi, (kernel_size, kernel_size), 999)

                if self.controls.eyes_visible:
                    # Handle unblurring of eye regions
                    landmarks = det[4:14].reshape(5, 2).astype(int)
                    right_eye, left_eye = landmarks[0], landmarks[1]

                    # Copy the original ROI to avoid multiple modifications
                    processed_roi = blurred.copy()

                    for eye in [right_eye, left_eye]:
                        ex, ey = eye
                        ex = int(ex * scale_x)
                        ey = int(ey * scale_y)
                        eye_x0 = max(0, ex - 15)
                        eye_y0 = max(0, ey - 15)
                        eye_x1 = min(frame.shape[1], ex + 15)
                        eye_y1 = min(frame.shape[0], ey + 15)

                        # Replace the blurred region with the unblurred eye region
                        if eye_x1 - eye_x0 > 0 and eye_y1 - eye_y0 > 0:
                            processed_roi[eye_y0-scaled_y:eye_y1-scaled_y, eye_x0-scaled_x:eye_x1-scaled_x] = \
                                frame[eye_y0:eye_y1, eye_x0:eye_x1]

                    # Apply the updated ROI to the frame
                    frame[scaled_y:scaled_y + scaled_h, scaled_x:scaled_x + scaled_w] = processed_roi
                else:
                    # Default: Blur the entire face ROI
                    frame[scaled_y:scaled_y + scaled_h, scaled_x:scaled_x + scaled_w] = blurred

        return frame
