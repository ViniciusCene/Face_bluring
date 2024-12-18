# Directory structure and modularization

# src/
# ├── main.py                # Entry point of the application
# ├── gui_module.py          # GUI components and event bindings
# ├── image_processing.py    # Face detection and image processing
# ├── controls.py            # Application state and logic
# ├── data_saving.py         # File saving and directory management

# --- main.py ---
# Entry point script to instantiate and coordinate the modules.


import tkinter as tk
from gui_module import YuNetBlurGUI
from controls import AppControls
from image_processing import ImageProcessor
from data_saving import DataSaver


def main():
    # Initialize application components
    root = tk.Tk()
    controls = AppControls()
    processor = ImageProcessor(controls)
    saver = DataSaver()

    # Set dependencies for controls
    controls.set_dependencies(processor, saver)

    # Pass components to the GUI
    app = YuNetBlurGUI(root, controls, processor, saver)

    # Start the application
    root.mainloop()


if __name__ == "__main__":
    main()

