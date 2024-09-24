# This method is an automatic face blurring algorithm based on YuNet hosted in the OpenCV Zoo project.
# It is subject to the MIT License terms in the LICENSE file found in the same directory.
#
# The YuNet is a Copyright (C) model from 2021, from Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.
# More details can be found at https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md

# Basic imports
import numpy as np
import cv2 as cv

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version(
    "4.10.0"
), "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

# Importing the model
from yunet import YuNet


def visualize(image, results, box_color=(0, 255, 0), text_font=cv.FONT_HERSHEY_SIMPLEX, text_color=(0, 0, 255), fps=None):
    """
    visualize _summary_

    _extended_summary_

    Parameters
    ----------
    image : _type_
        _description_
    results : _type_
        _description_
    box_color : tuple, optional
        _description_, by default (0, 255, 0)
    text_color : tuple, optional
        _description_, by default (0, 0, 255)
    fps : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    frame = image.copy()
    landmark_color = [
        (255, 0, 0),  # right eye
        (0, 0, 255),  # left eye
        (0, 255, 0),  # nose tip
        (255, 0, 255),  # right mouth corner
        (0, 255, 255),  # left mouth corner
    ]

    if fps is not None:
        cv.putText(
            frame, "FPS: {:.2f}".format(fps), (0, 15), text_font, 0.5, text_color
        )

    ratio=1.5

    for det in results:
        [roi_x0, roi_y0, roi_width, roi_height] = det[0:4].astype(np.int32)
        scaled_coord, scaled_ratios = scale_roi(roi_x0, roi_y0, roi_width, roi_height, ratio)

        cv.rectangle(
            frame,
            (scaled_coord[0], scaled_coord[1]),
            (scaled_coord[0] + scaled_ratios[0], scaled_coord[1] + scaled_ratios[1]),
            box_color,
            2,
        )

        frame = face_blur(frame, scaled_coord, scaled_ratios)

        score = det[-1]
        cv.putText(
            frame,
            "{:.4f}".format(score),
            (roi_x0, roi_y0 + 12),
            text_font,
            0.5,
            text_color,
        )

        landmarks = det[4:14].astype(np.int32).reshape((5, 2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(frame, landmark, 2, landmark_color[idx], 2)

    return frame


def scale_roi(roi_x0=None, roi_y0=None, roi_width=None, roi_height=None, ratio=1):
    """
    scale_roi _summary_

    _extended_summary_

    Parameters
    ----------
    roi_x0 : _type_, optional
        _description_, by default None
    roi_y0 : _type_, optional
        _description_, by default None
    roi_width : _type_, optional
        _description_, by default None
    roi_height : _type_, optional
        _description_, by default None
    ratio : int, optional
        _description_, by default 1

    Returns
    -------
    _type_
        _description_
    """
    # calculate the proportional bluring ratio surrounding the original ROI central point
   # center_ratios = [int(roi_width/2), int(roi_height/2)] # central points for w and h
    center_width = int(roi_width/2)
    center_height = int(roi_height/2)

    #central_coord = [roi_x0 + center_ratios[0], roi_y0 + center_ratios[1]]
    roi_center_x0 = roi_x0 + center_width
    roi_center_y0 = roi_y0 + center_height

    #scaled_ratios = [int(central_coord[0] * ratio), int(central_coord[1] * ratio)]
    scaled_height = int(roi_height * ratio)
    scaled_width = int(roi_width * ratio)
   
    #scaled_coord = [int(central_coord[0] - scaled_ratios[0]/2), int(central_coord[1] - scaled_ratios[1]/2)]
    scaled_x0 = int(roi_center_x0 - scaled_width/2)
    scaled_y0 = int(roi_center_y0 - scaled_height/2)

    scaled_coord = [scaled_x0, scaled_y0]
    scaled_ratios = [scaled_width, scaled_height]

    return scaled_coord, scaled_ratios


def face_blur(frame=None, scaled_coord=None, scaled_ratios=None):
    """
    face_blur _summary_

    _extended_summary_

    Parameters
    ----------
    frame : _type_, optional
        _description_, by default None
    scaled_coord : _type_, optional
        _description_, by default None
    scaled_ratios : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    scaled_roi = frame[
        scaled_coord[1]:scaled_coord[1] + scaled_ratios[1],
        scaled_coord[0]:scaled_coord[0] + scaled_ratios[0],
    ]

    # Apply Gaussian blur to the face ROI
    blurred_face = cv.GaussianBlur(scaled_roi, (99, 99), 30)

    # Place the blurred face back into the original frame
    frame[
        scaled_coord[1]:scaled_coord[1] + scaled_ratios[1],
        scaled_coord[0]:scaled_coord[0] + scaled_ratios[0],
    ] = blurred_face

    return frame


def yunet_config():
    """
    yunet_config _summary_

    _extended_summary_

    Returns
    -------
    _type_
        _description_
    """

    # Instantiate YuNet
    model = YuNet(

        # Path of model weights
        modelPath="trained_models/yunet/face_detection_yunet_2023mar.onnx",

        # Standard image input dimension (do not change that)!
        inputSize=[320, 320],

        # Threshold to identify faces (lower is more permissive but also prone to false detection)
        confThreshold=0.85,

        # Threshold to suppress bounding boxes of IoU >= nms_threshold. Default = 0.3
        nmsThreshold=0.3,

        # Keep top_k bounding boxes of face detection before nmsThreshold
        topK=5000,

    )

    return model


def config_video_input():
    """
    config_video_input _summary_

    _extended_summary_

    Returns
    -------
    _type_
        _description_
    """
    cap = cv.VideoCapture(0)
    img_resolution = [
        int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    ]

    return cap, img_resolution


def config_output_file(img_resolution=[680, 480]):
    """
    config_output_file _summary_

    _extended_summary_

    Parameters
    ----------
    img_resolution : list, optional
        _description_, by default [680, 480]

    Returns
    -------
    _type_
        _description_
    """
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    outup_file = cv.VideoWriter(
        "blurred_video2.avi", fourcc, 40.0, (img_resolution)
    )

    return outup_file


def main():
    """
    main _summary_

    _extended_summary_
    """

    # Configure video input
    cap, img_resolution = config_video_input()

    # Configure model
    model = yunet_config()
    model.setInputSize(img_resolution)

    # Configure video output
    outup_file = config_output_file(img_resolution)

    # tm will be used to calculate fps
    tm = cv.TickMeter()

    # Runs until any key is pressed
    while cv.waitKey(1) < 0:

        # Capture frame from video input
        has_frame, frame = cap.read()
        if not has_frame:
            print("No frames grabbed!")
            break

        # Model inference using frame
        tm.start()
        results = model.infer(frame)  # results is a tuple
        tm.stop()

        # Visualize results with boundarie box draw, bluring, and face detection score
        frame = visualize(frame, results, fps=tm.getFPS())

        # Add the processed frame to the output processed file
        outup_file.write(frame)

        # Visualize results on-the-fly
        cv.imshow("YuNet Blurring", frame)

        tm.reset()

    # Release the video capture and close the window
    cap.release()
    outup_file.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
