from colorama import Fore, Style
from matplotlib import pyplot as plt
from PIL import Image, UnidentifiedImageError
from plantcv import plantcv as pcv

import argparse
import cv2
import numpy as np
import os


def folder(p_src: str, p_dst: str, p_type: str):
    """
    Collects all JPG/JPEG images from a source directory and ensures the
    destination directory exists.

    :param p_src: Source directory containing images.
    :param p_dst: Destination directory for saving transformations.
    :param p_type: Type of processing (used for logging or future use).
    :return: List of absolute file paths to JPG images in the source directory.
    """
    if not os.path.isdir(p_dst):
        try:
            os.makedirs(p_dst)
        except Exception as e:
            print(f"{Fore.RED}Error: destination folder can't be "
                  f"created: {e}{Style.RESET_ALL}")
            return []

    jpg_files = [
        os.path.abspath(os.path.join(p_src, f))
        for f in os.listdir(p_src)
        if is_jpg(os.path.join(p_src, f))
        and os.path.isfile(os.path.join(p_src, f))
    ]

    return jpg_files


def is_jpg(path: str) -> bool:
    """
    Checks if the given file is a valid JPG/JPEG image.

    :param path: Path to the image file.
    :return: True if the file exists and is a JPEG image, False otherwise.
    """
    try:
        with Image.open(path) as img:
            return img.format == "JPEG"
    except (UnidentifiedImageError, OSError):
        return False


def parse_input():
    """
    Parses command-line arguments, including input path, destination folder,
    processing types, and visual flag.

    :return: Tuple containing:
             - v_path (str): Path to input image or directory.
             - v_args (argparse.Namespace): Full namespace with all options.
    """
    parser = argparse.ArgumentParser(prog='Augmentation')

    parser.add_argument('image_path', nargs='?',
                        help='Direct path to a single image file')

    parser.add_argument(
        '-dst', '--destination',
        required=True,
        help='Destination directory for saving transformations'
    )

    parser.add_argument('-v', '--visual', action='store_true',
                        help='Enable visual rendering')

    v_args = parser.parse_args()
    v_path = v_args.image_path

    return v_path, v_args


class Transformation:
    """
    Class representing a single image transformation,
    with optional visual output.
    """
    def __init__(self, p_path: str, p_visual: bool):
        """
        Constructor: Loads the image and initializes placeholders
        for transformation outputs.

        :param p_path: Path to the image file.
        :param p_visual: Boolean flag to display images visually
                         during processing.
        """
        self.img, self.path, self.filename = pcv.readimage(p_path)
        self.visual = p_visual

        self.img_roi = None
        self.img_gauss = None
        self.img_masked = None
        self.img_analyzed = None
        self.img_pseudolandmarks = None

    def __del__(self):
        """
        Destructor for Transformation.
        """
        pass

    def image(self):
        """
        Executes the transformation corresponding to the provided type.
        """
        try:
            self.gauss()
            self.roi()
            self.mask()
            # TODO: implement other cases
            return
        except Exception as e:
            print(f"{Fore.RED}Error: Processing failed for file: "
                  f"{self.path}: {e}{Style.RESET_ALL}")
            return

    def gauss(self):
        """
        Applies Gaussian blur to the image.
        """
        self.img_gauss = pcv.gaussian_blur(img=self.img, ksize=(21, 21),
                                           sigma_x=0, sigma_y=0)

        if self.visual:
            plt.imshow(self.img_gauss)
            plt.title("Gaussian Blur")
            plt.show()

    def mask(self):
        """
        Applies a binary mask based on the saturation channel of the image.
        """
        v_hsv = pcv.rgb2gray_hsv(rgb_img=self.img, channel='s')
        v_mask_binary = pcv.threshold.binary(
            gray_img=v_hsv, threshold=85, object_type='light')
        self.img_masked = pcv.apply_mask(
            img=self.img, mask=v_mask_binary, mask_color='white')

        if self.visual:
            plt.imshow(self.img_masked)
            plt.title("Mask Applied")
            plt.show()

    def roi(self):
        """
        Detects Regions of Interest (ROI) using the 'a' channel
        of LAB colorspace. Draws contours on the image.
        """
        v_gray = pcv.rgb2gray_lab(rgb_img=self.img, channel='a')
        v_mask = pcv.threshold.binary(
            gray_img=v_gray, threshold=100, object_type='light')
        v_roi = pcv.roi.from_binary_image(img=self.img, bin_img=v_mask)

        self.img_roi = self.img.copy()

        for contour in v_roi.contours:
            if isinstance(contour, tuple):
                contour = list(contour)
            cv2.drawContours(self.img_roi, contour, -1, (255, 0, 0), 3)

        if self.visual:
            plt.imshow(cv2.cvtColor(self.img_roi, cv2.COLOR_BGR2RGB))
            plt.title('Automatic ROI Detection')
            plt.show()

    def show_all(self):
        """
        Displays all available transformed images in a single figure
        with subplots. Click on an image to open it in a separate
        interactive window.
        """
        # Collect all images and titles
        images = [
            (self.img, "Original"),
            (self.img_gauss, "Gaussian Blur"),
            (self.img_masked, "Mask Applied"),
            (self.img_roi, "ROI Detection"),
            (self.img_analyzed, "Analyzed Objects"),
            (self.img_pseudolandmarks, "Pseudolandmarks"),
        ]

        # Keep only valid images
        images = [(img, title) for img, title in images if img is not None]

        n = len(images)

        if n == 0:
            print("No images to display")
            return

        # Create subplots
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        # Display each image in a subplot
        for ax, (img, title) in zip(axes, images):
            if isinstance(img, np.ndarray):
                img_disp = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            if img.ndim == 3 and img.shape[-1] == 3
                            else img)
            else:
                img_disp = img
            ax.imshow(img_disp)
            ax.set_title(title)
            ax.axis('off')

        # Function to open image in a new window on click
        def onclick(event):
            for i, ax in enumerate(axes):
                if ax == event.inaxes:
                    img, title = images[i]
                    plt.figure(figsize=(6, 6))
                    if isinstance(img, np.ndarray):
                        img_disp = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    if img.ndim == 3 and img.shape[-1] == 3
                                    else img)
                    else:
                        img_disp = img
                    plt.imshow(img_disp)
                    plt.title(title)
                    plt.axis('off')
                    plt.show()
                    break

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()


def main():
    """
    Main entry point of the script.
    Parses input arguments, iterates over images, applies requested
    transformations, and displays all results.
    """
    v_path, v_args = parse_input()

    if os.path.isdir(v_path):
        # Collect all JPG images in the directory
        v_list = folder(v_path, v_args.destination, p_type=None)
        if not v_list:
            print(f"{Fore.RED}Error: empty directory{Style.RESET_ALL}")
            return

        # Process each image
        for element in v_list:
            v_transformation = Transformation(element, v_args.visual)
            v_transformation.image()
            v_transformation.show_all()

    elif os.path.isfile(v_path):
        if not is_jpg(v_path):
            print(f"{Fore.RED}Error: argument needs to be a "
                  f"jpg/jpeg{Style.RESET_ALL}")
            return

        # Process single image
        v_transformation = Transformation(v_path, v_args.visual)
        v_transformation.image()
        v_transformation.show_all()

    else:
        print(f"{Fore.RED}Error: Provided path does not exist"
              f"{Style.RESET_ALL}")


if __name__ == '__main__':
    main()
