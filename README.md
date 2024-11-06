
# YuNet Automatic Face Blurring

YuNet Automatic Face Blurring is a real-time face-blurring application leveraging the YuNet model from the OpenCV Zoo. This tool detects faces in a video feed and applies a Gaussian blur to conceal them, making it useful for privacy-preserving applications such as video processing, surveillance, and content creation.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Folder Structure](#folder-structure)
- [License](#license)

## Features

- **Real-Time Detection**: Detects faces in real-time from the video feed using YuNet.
- **Automatic Blurring**: Applies Gaussian blur to detected faces.
- **Customizable Parameters**: Allows adjustment of detection and blurring parameters.
- **Live FPS Display**: Shows frames-per-second (FPS) in the video output.

## Installation

### Prerequisites

Ensure you have Python 3.7 or higher installed. This project relies on the OpenCV library, which should be version 4.10.0 or higher for optimal compatibility.

### Steps

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/YuNet-Automatic-Face-Blurring.git
    cd YuNet-Automatic-Face-Blurring
    ```

2. Install the required Python dependencies:

    ```bash
    python3 -m pip install --upgrade opencv-python
    ```

3. Ensure the YuNet model weights are in the `trained_models` directory within the `src` folder.

## Usage

To run the real-time face blurring application, execute the `yunet_blur.py` file:

```bash
python3 src/yunet_blur.py
```

This script will activate the webcam, apply face detection, and blur detected faces in real time. Press any key to stop the video feed and close the application.

## Dependencies

This project relies on the following dependencies:

- **OpenCV** (4.10.0 or higher): Provides tools for image processing and computer vision.
- **NumPy**: Used for numerical operations and array manipulation.

To install these dependencies, run:

```bash
python3 -m pip install -r requirements.txt
```

*Note:* Ensure you have an internet connection for downloading the model weights from the OpenCV Zoo if not pre-downloaded.

## Folder Structure

The project is organized as follows:

```
YuNet-Automatic-Face-Blurring/
├── LICENSE
├── README.md
└── src/
    ├── yunet.py           # YuNet model configuration file
    ├── yunet_blur.py      # Main script for running real-time face blurring
    └── trained_models/
        └── face_detection_yunet_2023mar.onnx  # Pre-trained model weights for YuNet
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
