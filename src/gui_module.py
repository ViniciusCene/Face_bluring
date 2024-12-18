import tkinter as tk
from tkinter import ttk, filedialog


class YuNetBlurGUI:
    def __init__(self, root, controls, processor, saver):
        self.root = root
        self.controls = controls
        self.processor = processor
        self.saver = saver

        # Configure the root window
        self.root.title("YuNet Face Blurring")
        self.root.geometry("1200x800")
        self.create_widgets()

    def create_widgets(self):
        """Initialize all GUI widgets."""

        # Frame for sliders
        slider_frame = tk.Frame(self.root)
        slider_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Sliders for controls (as before)
        self.threshold_label = tk.Label(slider_frame, text="Confidence Threshold", font=("Arial", 10))
        self.threshold_label.grid(row=0, column=0, padx=5, sticky="w")
        self.threshold_slider = ttk.Scale(slider_frame, from_=0.1, to=1.0, length=200, command=self.update_threshold)
        self.threshold_slider.set(0.5)
        self.threshold_slider.grid(row=1, column=0, padx=5)

        self.blur_label = tk.Label(slider_frame, text="Blur Intensity", font=("Arial", 10))
        self.blur_label.grid(row=0, column=1, padx=5, sticky="w")
        self.blur_slider = ttk.Scale(slider_frame, from_=1, to=20, length=200, command=self.update_blur_intensity)
        self.blur_slider.set(10)
        self.blur_slider.grid(row=1, column=1, padx=5)

        self.blur_area_label = tk.Label(slider_frame, text="Blur Area", font=("Arial", 10))
        self.blur_area_label.grid(row=0, column=2, padx=5, sticky="w")
        self.blur_area_slider = ttk.Scale(slider_frame, from_=100, to=200, length=200, command=self.update_blur_area)
        self.blur_area_slider.set(150)
        self.blur_area_slider.grid(row=1, column=2, padx=5)

        # Frame for buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.start_button = tk.Button(
            button_frame, text="Start", command=self.start_stop, width=15, font=("Arial", 12, "bold")
        )
        self.start_button.grid(row=0, column=0, padx=5)

        self.full_screen_button = tk.Button(
            button_frame, text="Enable Full Screen", command=self.toggle_full_screen, width=15, font=("Arial", 12, "bold")
        )
        self.full_screen_button.grid(row=0, column=1, padx=5)

        self.eyes_visible_button = tk.Button(
            button_frame, text="Eyes Visible: OFF", command=self.toggle_eyes_visible, width=15, font=("Arial", 12, "bold")
        )
        self.eyes_visible_button.grid(row=0, column=2, padx=5)

        self.offline_button = tk.Button(
            button_frame, text="Offline Processing", command=self.offline_processing, width=20, font=("Arial", 12, "bold")
        )
        self.offline_button.grid(row=0, column=3, padx=5)

        self.exit_button = tk.Button(
            button_frame, text="Exit", command=self.exit_program, width=15, font=("Arial", 12, "bold")
        )
        self.exit_button.grid(row=0, column=4, padx=5)

        # Canvas for video display
        self.canvas = tk.Canvas(self.root, bg="black", width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create an image placeholder on the canvas
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW)

    def start_stop(self):
        """Start or stop video processing."""
        if not self.controls.running:
            self.controls.start_processing(self.canvas, self.image_on_canvas)
            self.update_gui()  # Periodic GUI updates
            self.start_button.config(text="Stop")
        else:
            self.controls.stop_processing()
            self.start_button.config(text="Start")

    def update_gui(self):
        """Ensure smooth GUI updates."""
        if self.controls.running:
            self.root.update_idletasks()
            self.root.update()
            self.root.after(10, self.update_gui)  # Refresh every 10ms

    def toggle_full_screen(self):
        """Toggle between full screen and windowed mode."""
        if self.root.attributes("-fullscreen"):
            self.root.attributes("-fullscreen", False)
            self.full_screen_button.config(text="Enable Full Screen")
        else:
            self.root.attributes("-fullscreen", True)
            self.full_screen_button.config(text="Disable Full Screen")

    def toggle_eyes_visible(self):
        """Toggle eyes visible feature."""
        self.controls.eyes_visible = not self.controls.eyes_visible
        status = "ON" if self.controls.eyes_visible else "OFF"
        self.eyes_visible_button.config(text=f"Eyes Visible: {status}")

    def offline_processing(self):
        """Handle offline processing."""
        directory = filedialog.askdirectory(title="Select Directory for Offline Processing")
        if directory:
            self.controls.process_offline(directory)

    def exit_program(self):
        """Exit the program."""
        self.controls.stop_processing()
        self.root.destroy()

    def update_threshold(self, value):
        """Update the confidence threshold."""
        self.processor.update_confidence_threshold(float(value))

    def update_blur_intensity(self, value):
        """Update the blur intensity."""
        self.processor.blur_intensity = int(float(value))

    def update_blur_area(self, value):
        """Update the blur area."""
        self.processor.blur_area = int(float(value))
