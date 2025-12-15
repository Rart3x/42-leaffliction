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
        self.img_color_histogram = None

    def __del__(self):
        """
        Destructor for Transformation.
        """
        pass

    def image(self):
        """
        Executes the transformation corresponding to the provided type.
        """
        self.gauss()
        self.roi()
        self.mask()
        self.analyze()
        self.pseudo_landmarks()
        self.color_histogram()
        try:
            pass
        except Exception as e:
            print(f"{Fore.RED}Error: Processing failed for file: "
                  f"{self.path}: {e}{Style.RESET_ALL}")

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

    def analyze(self):
        v_hsv = pcv.rgb2gray_hsv(rgb_img=self.img, channel='s')
        v_mask_binary = pcv.threshold.binary(
            gray_img=v_hsv, threshold=85, object_type='light')
        shape_image = (pcv.analyze.size(
            img=self.img,
            labeled_mask=v_mask_binary,
            label=""))
        self.img_analyzed = shape_image.copy()

    def pseudo_landmarks(self):
        """
        Detects and draws pseudo-landmarks
            on the image using homology analysis.
        """
        v_hsv = pcv.rgb2gray_hsv(rgb_img=self.img, channel='s')
        v_mask_binary = pcv.threshold.binary(
            gray_img=v_hsv,
            threshold=85,
            object_type='light')
        left, right, center_h = pcv.homology.y_axis_pseudolandmarks(
            img=self.img, mask=v_mask_binary)

        self.img_pseudolandmarks = self.img.copy()

        for point in left:
            pt = tuple(map(int, point[0]))
            cv2.circle(self.img_pseudolandmarks, pt, 5, (255, 0, 0), -1)

        for point in right:
            pt = tuple(map(int, point[0]))
            cv2.circle(self.img_pseudolandmarks, pt, 5, (0, 255, 0), -1)

        for point in center_h:
            pt = tuple(map(int, point[0]))
            cv2.circle(self.img_pseudolandmarks, pt, 5, (0, 0, 255), -1)

        if self.visual:
            plt.imshow(cv2.cvtColor(self.img_pseudolandmarks,
                                    cv2.COLOR_BGR2RGB))
            plt.title('Pseudo-landmarks'
                      '(Blue: Left, Green: Right, Red: Center)')
            plt.show()

    def color_histogram(self):
        """
        Analyzes and displays color histograms for RGB, LAB and HSV
        in a single combined plot.
        """
        hsv_gray = pcv.rgb2gray_hsv(
            rgb_img=self.img,
            channel="s",
        )

        mask = pcv.threshold.binary(
            gray_img=hsv_gray,
            threshold=85,
            object_type="light",
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        # --- RGB ---
        rgb_colors = ("b", "g", "r")
        rgb_labels = ("R", "G", "B")

        for i, (color, label) in enumerate(zip(rgb_colors, rgb_labels)):
            hist = cv2.calcHist(
                [self.img],
                [i],
                mask,
                [256],
                [0, 256],
            )
            ax.plot(
                hist,
                color=color,
                linestyle="-",
                label=f"RGB-{label}",
            )

        # --- LAB ---
        lab_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        lab_colors = ("black", "darkgreen", "darkred")
        lab_labels = ("L", "A", "B")

        for i, (color, label) in enumerate(zip(lab_colors, lab_labels)):
            hist = cv2.calcHist(
                [lab_img],
                [i],
                mask,
                [256],
                [0, 256],
            )
            ax.plot(
                hist,
                color=color,
                linestyle="--",
                label=f"LAB-{label}",
            )

        # --- HSV ---
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        hsv_colors = ("orange", "purple", "cyan")
        hsv_labels = ("H", "S", "V")

        for i, (color, label) in enumerate(zip(hsv_colors, hsv_labels)):
            hist = cv2.calcHist(
                [hsv_img],
                [i],
                mask,
                [256],
                [0, 256],
            )
            ax.plot(
                hist,
                color=color,
                linestyle=":",
                label=f"HSV-{label}",
            )

        ax.set_title("Color Histograms (RGB / LAB / HSV)")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.legend(ncol=3)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        # Convert figure to image
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(height, width, 4)

        self.img_color_histogram = cv2.cvtColor(
            buf,
            cv2.COLOR_RGBA2BGR,
        )

        if self.visual:
            plt.show()
        else:
            plt.close(fig)

    def save(self, p_dst: str):
        """
        Saves all transformed images to the destination directory.

        :param p_dst: Destination directory path where images will be saved.
        """
        images = [
            (self.img, "_original"),
            (self.img_gauss, "_gaussian_blur"),
            (self.img_masked, "_mask_applied"),
            (self.img_roi, "_ROI_detection"),
            (self.img_analyzed, "_analyzed_objects"),
            (self.img_pseudolandmarks, "_pseudolandmarks"),
            (self.img_color_histogram, "_color_histogram"),
        ]

        base_name = os.path.splitext(self.filename)[0]

        for img, suffix in images:
            if img is not None:
                output_path = os.path.join(p_dst, f"{base_name}{suffix}.jpg")
                try:
                    cv2.imwrite(output_path, img)
                    print(f"{Fore.GREEN}Saved: {output_path}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}"
                          f"Error saving {output_path}: {e}"
                          f"{Style.RESET_ALL}")

    def show_all(self):
        """
        Displays all available transformed images.
        Main images are displayed on the first row,
        color histogram is displayed below in full width.
        Clicking on any image opens it in a separate window.
        """
        main_images = [
            (self.img, "Original"),
            (self.img_gauss, "Gaussian Blur"),
            (self.img_masked, "Mask Applied"),
            (self.img_roi, "ROI Detection"),
            (self.img_analyzed, "Analyzed Objects"),
            (self.img_pseudolandmarks, "Pseudolandmarks"),
        ]

        main_images = [
            (img, title) for img, title in main_images if img is not None
        ]

        hist_image = self.img_color_histogram

        if not main_images and hist_image is None:
            print("No images to display")
            return

        n_cols = len(main_images)
        n_rows = 2 if hist_image is not None else 1

        fig = plt.figure(figsize=(5 * n_cols, 10 if n_rows == 2 else 5))
        gs = fig.add_gridspec(n_rows, n_cols)

        axes = []

        # --- First row: main images ---
        for col, (img, title) in enumerate(main_images):
            ax = fig.add_subplot(gs[0, col])
            img_disp = (
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if isinstance(img, np.ndarray)
                and img.ndim == 3 and img.shape[-1] == 3
                else img
            )
            ax.imshow(img_disp)
            ax.set_title(title)
            ax.axis("off")
            axes.append((ax, img, title))

        # --- Second row: histogram (full width) ---
        if hist_image is not None:
            hist_ax = fig.add_subplot(gs[1, :])
            hist_disp = cv2.cvtColor(
                hist_image,
                cv2.COLOR_BGR2RGB,
            )
            hist_ax.imshow(hist_disp)
            hist_ax.set_title("Color Histogram")
            hist_ax.axis("off")
            axes.append((hist_ax, hist_image, "Color Histogram"))

        # --- Click handling ---
        def onclick(event):
            for ax, img, title in axes:
                if event.inaxes == ax:
                    plt.figure(figsize=(8, 6))
                    img_disp = (
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if isinstance(img, np.ndarray)
                        and img.ndim == 3
                        and img.shape[-1] == 3
                        else img
                    )
                    plt.imshow(img_disp)
                    plt.title(title)
                    plt.axis("off")
                    plt.show()
                    break

        fig.canvas.mpl_connect("button_press_event", onclick)
        plt.tight_layout()
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
        if not v_args.destination:
            print(f"{Fore.RED}"
                  f"Error: provide destination when it's a folder"
                  f"{Style.RESET_ALL}")
            return
        v_list = folder(v_path, v_args.destination, p_type=None)
        if not v_list:
            print(f"{Fore.RED}Error: empty directory{Style.RESET_ALL}")
            return

        # Process each image
        for element in v_list:
            v_transformation = Transformation(element, v_args.visual)
            v_transformation.image()
            v_transformation.save(v_args.destination)

    elif os.path.isfile(v_path):
        if v_args.destination:
            print(f"{Fore.YELLOW}"
                  f"WARNING: destination path won't be used"
                  f"{Style.RESET_ALL}")
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
