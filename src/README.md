# Modular Face Blurring Application

This pull request introduces a modular and scalable Python-based application for real-time and offline video face blurring using the YuNet model from OpenCV. The solution is structured into the following modules, each encapsulating specific functionalities for clarity and maintainability.

---

## **Project Structure**

### **1. `main.py`**
The entry point of the application. It initializes and orchestrates all components:
- **Modules Imported**: 
  - `gui_module`
  - `controls`
  - `image_processing`
  - `data_saving`
- **Primary Functionality**:
  - Initializes the GUI (`YuNetBlurGUI`) and connects it to the `AppControls`, `ImageProcessor`, and `DataSaver`.

---

### **2. `gui_module.py`**
Defines the graphical user interface (GUI) for the application using `Tkinter`.
- **Key Features**:
  - **Buttons**:
    - **Start/Stop**: Toggles video processing.
    - **Full Screen**: Toggles between full-screen and windowed modes.
    - **Eyes Visible**: Toggles visibility of the eye regions in the blurred video.
    - **Offline Processing**: Processes all video files in a selected directory.
    - **Exit**: Exits the application.
  - **Sliders**:
    - **Confidence Threshold**: Adjusts the face detection confidence level.
    - **Blur Intensity**: Adjusts the level of blurring applied.
    - **Blur Area**: Controls the proportional area to blur around the detected face.
  - **Video Display**:
    - Integrates a `Tkinter` canvas for displaying processed frames in real time.

---

### **3. `controls.py`**
Handles the core application logic, such as video capture, processing, and GUI interaction.
- **Primary Class**: `AppControls`
- **Key Functions**:
  - `start_processing(canvas, image_on_canvas)`: Starts the video processing thread.
  - `stop_processing()`: Stops the video processing thread and cleans up resources.
  - `_process_video()`: Handles frame-by-frame processing for real-time display and saving.
  - `set_dependencies(processor, saver)`: Links the processor and saver modules to the controls.

---

### **4. `image_processing.py`**
Handles the face detection and image processing logic using the YuNet model.
- **Primary Class**: `ImageProcessor`
- **Key Functions**:
  - `__init__(controls)`: Initializes the processor and links it to the controls.
  - `_load_model()`: Loads the YuNet face detection model.
  - `process_frame(frame)`: Processes each video frame to:
    - Detect faces.
    - Apply Gaussian blur to detected regions.
    - Optionally make the eye regions visible if toggled via the GUI.

---

### **5. `data_saving.py`**
Manages video saving functionality for both real-time and offline processing.
- **Primary Class**: `DataSaver`
- **Key Functions**:
  - `initialize_writer(frame_size, fps)`: Initializes the video writer.
  - `write_frame(frame)`: Writes processed frames to the output video.
  - `finalize_writer()`: Releases the video writer and cleans up resources.

---

### **6. `yunet.py`**
Encapsulates the YuNet model logic, adapting it for face detection.
- **Primary Class**: `YuNet`
- **Key Functions**:
  - `__init__(model_path)`: Loads the ONNX model and sets parameters.
  - `detect(frame)`: Detects faces in the provided frame.

---

## **Key Features**
1. **Real-Time Face Blurring**:
   - Captures video from a webcam.
   - Detects faces and applies blurring based on user-defined settings.
   - Displays the processed video in the GUI.

2. **Offline Batch Video Processing**:
   - Allows users to select a directory containing video files.
   - Processes each video sequentially, saving the results in an organized structure.

3. **Eye Region Visibility**:
   - Toggles between fully blurred faces and visible eye regions.

4. **Dynamic GUI Controls**:
   - Sliders to adjust confidence threshold, blur intensity, and blur area dynamically.
   - Buttons for start/stop, full screen, offline processing, and exit.

5. **Processed Video Saving**:
   - Saves real-time and offline processed videos with user-adjustable parameters.

---

## **How to Test**
1. **Real-Time Processing**:
   - Run `main.py`.
   - Click "Start" to begin real-time video processing.
   - Use sliders and buttons to adjust settings dynamically.

2. **Offline Processing**:
   - Click "Offline Processing".
   - Select a directory with video files.
   - Processed videos are saved in a subdirectory (`blurred_files`).

3. **Feature Toggling**:
   - Test the "Eyes Visible" button to toggle between blurred and visible eye regions.

4. **GUI Responsiveness**:
   - Resize the window and test full-screen mode for dynamic adjustments.

---

## **Future Enhancements**
- **Performance Optimization**:
  - Add GPU acceleration for the YuNet model.
- **Robust Error Handling**:
  - Handle unsupported video formats gracefully.
- **Improved User Experience**:
  - Add a loading indicator during offline processing.
- **Custom Output Directory**:
  - Allow users to specify the output folder for processed videos.
- **Logging**:
  - Add runtime logging for easier debugging.

---

This modular design improves code maintainability, scalability, and testing, setting a solid foundation for future enhancements.
