# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import cv2 as cv


class YuNet:
    def __init__(self, model_path):
        """
        Initialize the YuNet face detection model with optional GPU acceleration.
        """
        self.model_path = model_path
        self._model = self._load_model()

    def _load_model(self):
        """
        Load the YuNet model with GPU acceleration if available.
        """
        # Load the ONNX model
        net = cv.dnn.readNet(self.model_path)

        # Check for GPU availability
        if cv.cuda.getCudaEnabledDeviceCount() > 0:
            print("[INFO] Using GPU acceleration for YuNet.")
            net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        else:
            print("[INFO] GPU not available. Using CPU.")
            net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        # Configure the model
        face_detector = cv.FaceDetectorYN_create(
            model=self.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000,
        )
        face_detector.setNet(net)
        return face_detector

    def detect(self, frame):
        """
        Perform face detection on the given frame.

        Parameters:
            frame (numpy.ndarray): The input frame.

        Returns:
            numpy.ndarray: Detection results (bounding boxes, landmarks, scores).
        """
        results = self._model.detect(frame)
        return results
