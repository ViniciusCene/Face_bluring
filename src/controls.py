import os
import concurrent.futures
import cv2 as cv
from PIL import Image, ImageTk
from image_processing import ImageProcessor
from data_saving import DataSaver


class AppControls:
    def __init__(self):
        self.running = False
        self.eyes_visible = False 
        self.processor = None
        self.saver = None

    def set_dependencies(self, processor, saver):
        """Set dependencies for the controls."""
        self.processor = processor
        self.saver = saver

    def toggle_eyes_visible(self):
        """Toggle the eyes_visible attribute."""
        self.eyes_visible = not self.eyes_visible        

    def process_offline(self, directory):
        """Process all video files in the selected directory."""
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} does not exist.")
            return

        video_files = [f for f in os.listdir(directory) if f.endswith(('.mp4', '.avi', '.mkv'))]
        if not video_files:
            print(f"No video files found in {directory}.")
            return

        output_dir = os.path.join(directory, "blurred_files")
        os.makedirs(output_dir, exist_ok=True)

        def process_video(video_file):
            input_path = os.path.join(directory, video_file)
            output_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}_blurred.mp4")
            try:
                print(f"Processing: {video_file}")

                # Check if video file is readable
                test_cap = cv.VideoCapture(input_path)
                if not test_cap.isOpened():
                    raise IOError(f"Could not open video file: {video_file}")
                test_cap.release()

                # Get video properties
                cap = cv.VideoCapture(input_path)
                frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv.CAP_PROP_FPS) or 30

                # Use independent instances of ImageProcessor and DataSaver
                local_processor = ImageProcessor(self)  # Create a new instance
                local_saver = DataSaver()
                local_saver.initialize_writer((frame_width, frame_height), fps, output_path)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Process the frame
                    processed_frame = local_processor.process_frame(frame)

                    # Write processed frame
                    local_saver.write_frame(processed_frame)

                    # Write processed frame
                    ## self.saver.write_frame(processed_frame)

                # Release video resources
                cap.release()
                self.saver.finalize_writer()
                print(f"Processed: {video_file}")

            except Exception as e:
                print(f"Error processing {video_file}: {e}")

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_video, video_file) for video_file in video_files]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Raise exception if the task failed
                except Exception as e:
                    print(f"Error in thread: {e}")

        print("Offline processing completed.")

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
        try:
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                raise IOError("Could not open video source.")

            # Initialize video writer for saving processed output
            frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv.CAP_PROP_FPS) or 30  # Default to 30 FPS if unavailable
            
            # Set default output path for real-time processing
            output_path = os.path.join("output_videos", "realtime_output.mp4")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.saver.initialize_writer((frame_width, frame_height), fps, output_path)

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    raise IOError("Failed to capture frame.")

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

            cap.release()
            self.saver.finalize_writer()
            self.canvas.delete(self.image_on_canvas)
            cv.destroyAllWindows()

        except Exception as e:
            print(f"Real-time processing error: {e}")
            self.running = False

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
