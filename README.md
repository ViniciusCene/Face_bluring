# YuNet Automatic Face Blurring

This project provides real-time **automatic face blurring** using the **YuNet** face detection model hosted in the **OpenCV Zoo**. It features a user-friendly GUI for adjusting blurring parameters and managing video acquisition settings.


## Features

### Real-Time Face Blurring
- Detects faces using the **YuNet** model and applies a Gaussian blur to the detected regions.
- Adjustable **confidence threshold** for face detection.
- Scalable region of interest (ROI) for blurring.
- Configurable blur intensity.

### Interactive GUI
The GUI offers the following controls:

#### Buttons:
1. **Start/Stop**: 
   - **Start**: Begins video acquisition and processing.
   - **Stop**: Halts video processing and releases resources.
2. **Full Screen**:
   - **Disable Full Screen** (default): The GUI opens in full-screen mode by default. Clicking this switches to a windowed mode.
   - **Enable Full Screen**: Switches back to full-screen mode.
3. **Eyes Visible**:
   - Toggles the visibility of eyes in the blurred regions.
   - **ON**: Eyes remain unblurred while the rest of the face is blurred.
   - **OFF**: Entire face is blurred.
4. **Offline Processing**:
   - Allows processing of multiple video files in a selected directory.
   - Saves blurred videos in a subfolder named `blurred files` in the same directory.
   - Replaces previously processed videos if they exist.
   - Supports various formats, including `.avi`, `.mp4`, `.mov`, and `.mkv`.   
5. **Exit Program**:
   - Stops video processing (if running) and exits the application.

#### Sliders:
1. **Confidence Threshold**:
   - Adjusts the detection threshold for faces.
   - Range: `0.1` to `1.0`.
2. **Blur Intensity**:
   - Configures the intensity of the Gaussian blur.
   - Range: `1` to `20`.
3. **Blurring Area**:
   - Scales the ROI for blurring.
   - Range: `100%` to `200%`.

### Output
- Displays real-time processed video with blurred faces.
- Saves the video to a file (`output_blurred.mp4`) upon exiting.
- Offline processing outputs are saved to a folder named `blurred files`.

## Project Structure

```
YuNet_Automatic_Face_Blurring/
├── src/
│   ├── yunet.py               # Implements YuNet face detection logic.
│   ├── yunet_blur.py          # Main GUI application and video processing logic.
│   └── trained_models/
│       └── face_detection_yunet_2023mar.onnx  # Pre-trained YuNet model file.
├── README.md                  # Project documentation.
├── LICENSE                    # License file.
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/YuNet_Automatic_Face_Blurring.git
   cd YuNet_Automatic_Face_Blurring
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python opencv-python-headless ttkbootstrap numpy
   ```

3. Ensure the **YuNet** model file is present in the `trained_models` directory.

## Usage

1. Navigate to the `src` directory:
   ```bash
   cd src
   ```

2. Run the application:
   ```bash
   python yunet_blur.py
   ```

3. Interact with the GUI:
   - Adjust sliders to configure the face detection and blurring parameters.
   - Use the buttons to start/stop processing, toggle full-screen mode, enable/disable eye visibility, perform offline batch video processing. and exit the program.

4. Offline Video Processing:
   - Click the Offline Processing button to select a directory containing video files.
   - All supported videos (`.avi`, `.mp4`, `.mov`, and `.mkv`.) will be processed.
   - Processed videos are saved in a subfolder named blurred files.

## How It Works

- The **YuNet** model detects faces in real-time using the OpenCV `dnn` module.
- Detected face regions are processed with a Gaussian blur to anonymize them.
- The GUI allows dynamic adjustment of detection and blurring parameters without restarting the application.

## Screenshots

*To be added here*

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- **YuNet Model**: Shenzhen Institute of Artificial Intelligence and Robotics for Society.
- **OpenCV Zoo**: Hosting and integration of pre-trained models.
