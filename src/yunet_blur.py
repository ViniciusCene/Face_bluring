# Imports
import os
import numpy as np
import cv2 as cv
from yunet import YuNet

# Check OpenCV version
assert tuple(map(int, cv.__version__.split("."))) >= (4, 10, 0), (
    "Please install the latest version of opencv-python (>= 4.10.0)."
)


def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), 
              fps=None):
    """
    Draws bounding boxes, landmarks, and applies blur to detected faces 
    in the image.

    Parameters
    ----------
    image : np.array
        Input image frame in which faces are detected.
    results : list or np.array
        List of face detection results with bounding boxes and landmarks.
    box_color : tuple, optional
        Color of the bounding box in (B, G, R) format, by default (0, 255, 0).
    text_color : tuple, optional
        Color of the text displaying FPS, by default (0, 0, 255).
    fps : float, optional
        Frames per second to be displayed, by default None.

    Returns
    -------
    np.array
        Image with annotations and blurred faces.
    """
    if results is None or results.size == 0:
        return image

    frame = image.copy()
    landmarks_color = [
        (255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255)
    ]

    if fps:
        cv.putText(frame, f"FPS: {fps:.2f}", (0, 15), cv.FONT_HERSHEY_SIMPLEX, 
                   0.5, text_color)

    for det in results:
        roi_x0, roi_y0, roi_w, roi_h = det[:4].astype(int)
        scaled_coord, scaled_size = scale_roi(
            roi_x0, roi_y0, roi_w, roi_h, ratio=1.5
        )
        score = det[-1]

        # Draw bounding box and blur the face
        cv.rectangle(
            frame, tuple(scaled_coord), tuple(scaled_coord + scaled_size), 
            box_color, 2
        )
        frame = face_blur(frame, scaled_coord, scaled_size)

        # Display detection score
        cv.putText(
            frame, f"{score:.4f}", (roi_x0, roi_y0 + 12), cv.FONT_HERSHEY_SIMPLEX, 
            0.5, text_color
        )

        # Draw landmarks
        for idx, (x, y) in enumerate(det[4:14].reshape(5, 2).astype(int)):
            cv.circle(frame, (x, y), 2, landmarks_color[idx], 2)

    return frame


def scale_roi(x0, y0, width, height, ratio=1.0):
    """
    Scales the region of interest (ROI) for blurring based on the given ratio.

    Parameters
    ----------
    x0 : int
        X-coordinate of the top-left corner of the bounding box.
    y0 : int
        Y-coordinate of the top-left corner of the bounding box.
    width : int
        Width of the bounding box.
    height : int
        Height of the bounding box.
    ratio : float, optional
        Scaling ratio for the ROI, by default 1.0.

    Returns
    -------
    tuple of np.array
        Scaled top-left corner coordinates and dimensions (width, height) 
        of the ROI.
    """
    center_x, center_y = x0 + width // 2, y0 + height // 2
    new_width, new_height = int(width * ratio), int(height * ratio)
    new_x0, new_y0 = center_x - new_width // 2, center_y - new_height // 2
    return np.array([new_x0, new_y0]), np.array([new_width, new_height])


def face_blur(frame, coord, size):
    """
    Applies Gaussian blur to the specified face region in the frame.

    Parameters
    ----------
    frame : np.array
        Input image frame where blurring is applied.
    coord : np.array
        Top-left corner coordinates of the region to blur.
    size : np.array
        Dimensions (width, height) of the region to blur.

    Returns
    -------
    np.array
        Image frame with the specified region blurred.
    """
    x, y = coord
    w, h = size
    if y < 0 or x < 0 or y + h > frame.shape[0] or x + w > frame.shape[1]:
        print("Warning: ROI out of bounds, skipping blur.")
        return frame
    frame[y:y+h, x:x+w] = cv.GaussianBlur(frame[y:y+h, x:x+w], (99, 99), 30)
    return frame


def load_yunet_model():
    """
    Loads the YuNet model from the trained models directory.

    Returns
    -------
    YuNet
        Configured YuNet model for face detection.
    """
    model_path = os.path.join(
        os.path.dirname(__file__), 
        "trained_models/yunet/face_detection_yunet_2023mar.onnx"
    )
    return YuNet(
        modelPath=model_path, inputSize=[320, 320], confThreshold=0.85, 
        nmsThreshold=0.3, topK=5000
    )


def setup_video_input():
    """
    Sets up video input from the default webcam.

    Returns
    -------
    tuple
        Video capture object and resolution (width, height).
    """
    cap = cv.VideoCapture(0)
    resolution = (
        int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), 
        int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    )
    return cap, resolution


def setup_video_output(resolution):
    """
    Sets up the video output file for saving processed video.

    Parameters
    ----------
    resolution : tuple
        Resolution (width, height) of the video output.

    Returns
    -------
    cv.VideoWriter
        VideoWriter object to save the processed video.
    """
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    return cv.VideoWriter("blurred_video.avi", fourcc, 40.0, resolution)


def main():
    """
    Main function to run the face blurring application.
    Sets up video input, loads model, processes each frame, and displays 
    the output.

    Returns
    -------
    None
    """
    cap, resolution = setup_video_input()
    model = load_yunet_model()
    model.setInputSize(resolution)
    output_file = setup_video_output(resolution)
    timer = cv.TickMeter()

    while cv.waitKey(1) < 0:
        has_frame, frame = cap.read()
        if not has_frame:
            print("No frames grabbed!")
            break

        timer.start()
        results = model.infer(frame)
        timer.stop()
        frame = visualize(frame, results, fps=timer.getFPS())
        output_file.write(frame)
        cv.imshow("YuNet Face Blurring", frame)
        timer.reset()

    cap.release()
    output_file.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
