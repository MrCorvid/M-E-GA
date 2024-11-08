import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import sys
import re


def select_folder():
    """
    Opens a dialog to select a folder and returns the selected folder path.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory(title='Select Folder Containing Images')
    return folder_selected


def natural_sort_key(s):
    """
    Generates a key for natural sorting of strings containing numbers.
    Example: 'image2.png' comes before 'image10.png'.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('(\d+)', s)]


def get_image_files(folder):
    """
    Retrieves and sorts image files from the specified folder.

    Args:
        folder (str): Path to the folder containing images.

    Returns:
        list: Sorted list of image filenames.
    """
    # Supported image extensions
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']

    # List all files in the folder with supported extensions
    files = [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
           and os.path.splitext(f.lower())[1] in image_extensions
    ]

    if not files:
        return []

    # Sort files naturally (e.g., image1.png, image2.png, ..., image10.png)
    files.sort(key=natural_sort_key)
    return files


def create_gif(folder, image_files, output_path, duration=500):
    """
    Creates a GIF from a list of image files with a consistent global palette.

    Args:
        folder (str): Path to the folder containing images.
        image_files (list): List of image filenames.
        output_path (str): Path where the output GIF will be saved.
        duration (int, optional): Duration of each frame in milliseconds. Defaults to 500.

    Returns:
        bool: True if GIF creation was successful, False otherwise.
    """
    images = []
    base_size = None  # To ensure all images are the same size

    try:
        # Open all images and convert them to RGB
        for file_name in image_files:
            file_path = os.path.join(folder, file_name)
            img = Image.open(file_path).convert('RGB')

            # Resize images to a common size if necessary
            if base_size is None:
                base_size = img.size
            else:
                if img.size != base_size:
                    img = img.resize(base_size, Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS

            images.append(img)

        if not images:
            print("No images to create GIF.")
            return False

        # Create a global palette from all images
        # Merge all images into one to generate a representative palette
        palette_image = Image.new('RGB', (base_size[0], base_size[1] * len(images)))
        for idx, img in enumerate(images):
            palette_image.paste(img, (0, idx * base_size[1]))

        # Quantize the palette image to get a global palette
        palette_image = palette_image.quantize(colors=256, method=Image.MEDIANCUT)
        global_palette = palette_image.getpalette()

        # Apply the global palette to all images
        quantized_images = []
        for img in images:
            quantized = img.quantize(palette=palette_image, method=Image.FASTOCTREE)
            quantized_images.append(quantized)

        # Save the GIF with the global palette
        quantized_images[0].save(
            output_path,
            save_all=True,
            append_images=quantized_images[1:],
            duration=duration,
            loop=0,
            disposal=2,
            optimize=False  # Avoid optimization to prevent palette issues
        )
        return True

    except Exception as e:
        print(f"Error saving GIF: {e}")
        return False


def delete_files(folder, image_files):
    """
    Deletes a list of files from the specified folder.

    Args:
        folder (str): Path to the folder containing files.
        image_files (list): List of filenames to delete.
    """
    for file_name in image_files:
        file_path = os.path.join(folder, file_name)
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


def main():
    """
    Main function to orchestrate GIF creation and file deletion.
    """
    folder = select_folder()
    if not folder:
        print("No folder selected. Exiting.")
        sys.exit()

    image_files = get_image_files(folder)
    if not image_files:
        messagebox.showerror("No Images Found", "No image files were found in the selected folder.")
        sys.exit()

    # Ask user for confirmation before proceeding
    proceed = messagebox.askyesno(
        "Create GIF",
        f"Found {len(image_files)} image(s) in the folder.\nDo you want to create a GIF and delete the source images?"
    )
    if not proceed:
        print("Operation cancelled by user.")
        sys.exit()

    # Define output GIF path
    output_gif = os.path.join(folder, "output.gif")

    # Check if output GIF already exists
    if os.path.exists(output_gif):
        overwrite = messagebox.askyesno(
            "Overwrite GIF",
            f"The file {output_gif} already exists.\nDo you want to overwrite it?"
        )
        if not overwrite:
            messagebox.showinfo("Operation Cancelled", "GIF creation cancelled to prevent overwriting existing file.")
            sys.exit()

    # Create GIF
    success = create_gif(folder, image_files, output_gif)
    if success:
        # Delete source images
        delete_files(folder, image_files)
        messagebox.showinfo(
            "Success",
            f"GIF created successfully at:\n{output_gif}\nSource images have been deleted."
        )
    else:
        messagebox.showerror("Error", "Failed to create GIF.")


if __name__ == "__main__":
    main()
