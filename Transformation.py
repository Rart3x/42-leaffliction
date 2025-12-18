from colorama import Fore, Style
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv
from utils import is_jpg

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


def parse_input():
    """
    Parses command-line arguments, including input path, destination folder,
    processing types, and visual flag.

    :return: Tuple containing:
             - v_path (str): Path to input image or directory.
             - v_args (argparse.Namespace): Full namespace with all options.
    """
    parser = argparse.ArgumentParser(prog='Augmentation')

    parser.add_argument('image_path',
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


# Helper functions to replace plantcv functionality with cv2
def _extract_hsv_channel(img, channel):
    """
    Extract a specific channel from HSV color space.

    :param img: Input BGR image
    :param channel: Channel to extract ('h', 's', 'v')
    :return: Single channel grayscale image
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    channel_map = {'h': 0, 's': 1, 'v': 2}
    return hsv[:, :, channel_map[channel.lower()]]


def _extract_lab_channel(img, channel):
    """
    Extract a specific channel from LAB color space.

    :param img: Input BGR image
    :param channel: Channel to extract ('l', 'a', 'b')
    :return: Single channel grayscale image
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    channel_map = {'l': 0, 'a': 1, 'b': 2}
    return lab[:, :, channel_map[channel.lower()]]


def _binary_threshold(gray_img, threshold, object_type='light'):
    """
    Apply binary thresholding to a grayscale image.

    :param gray_img: Input grayscale image
    :param threshold: Threshold value
    :param object_type: 'light' for objects brighter than threshold,
                        'dark' for darker
    :return: Binary mask image
    """
    if object_type == 'light':
        _, binary = cv2.threshold(
            gray_img, threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(
            gray_img, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


def _apply_mask(img, mask, mask_color='white'):
    """
    Apply a binary mask to an image.

    :param img: Input image
    :param mask: Binary mask (0 or 255)
    :param mask_color: Background color when mask is 0 ('white' or 'black')
    :return: Masked image
    """
    if mask_color == 'white':
        background = np.ones_like(img) * 255
    else:
        background = np.zeros_like(img)

    mask_3channel = cv2.merge([mask, mask, mask]) / 255.0
    result = img * mask_3channel + background * (1 - mask_3channel)
    return result.astype(np.uint8)


class _ROI:
    """Simple class to hold ROI contours, mimicking plantcv's ROI object."""
    def __init__(self, contours):
        self.contours = contours


def _find_roi_contours(bin_img):
    """
    Find contours from a binary image and return ROI-like object.

    :param bin_img: Binary image
    :return: ROI object with contours attribute
    """
    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _ROI(contours)


def _analyze_size(img, labeled_mask, label=""):
    """
    Analyze size/shape of objects in mask and draw on image.

    :param img: Input image
    :param labeled_mask: Binary mask
    :param label: Optional label (not used)
    :return: Image with contours and bounding boxes drawn
    """
    result_img = img.copy()
    contours, _ = cv2.findContours(
        labeled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Draw contour
        cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 2)
        # Draw bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(
            result_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return result_img


def _y_axis_pseudolandmarks(img, mask):
    """
    Compute pseudo-landmarks along the y-axis of the mask.

    :param img: Input image (not used but kept for compatibility)
    :param mask: Binary mask
    :return: Tuple of (left_points, right_points, center_points)
    """
    height, width = mask.shape
    left_points = []
    right_points = []
    center_points = []

    # Find contours to get the object region
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return left_points, right_points, center_points

    # Combine all contours into a single mask region
    combined_mask = np.zeros_like(mask)
    cv2.drawContours(combined_mask, contours, -1, 255, -1)

    # For each y-coordinate, find leftmost, rightmost, and center x-coords
    for y in range(height):
        row = combined_mask[y, :]
        white_pixels = np.where(row == 255)[0]

        if len(white_pixels) > 0:
            left_x = white_pixels[0]
            right_x = white_pixels[-1]
            center_x = (left_x + right_x) // 2

            # Store as arrays with [0] access pattern like plantcv
            left_points.append(np.array([[left_x, y]]))
            right_points.append(np.array([[right_x, y]]))
            center_points.append(np.array([[center_x, y]]))

    return left_points, right_points, center_points


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
        self.path = os.path.abspath(p_path)
        self.filename = os.path.basename(p_path)
        self.img = cv2.imread(self.path)
        if self.img is None:
            raise ValueError(f"Could not load image from {p_path}")
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
        try:
            self.gauss()
            self.roi()
            self.mask()
            self.analyze()
            self.pseudo_landmarks()
            self.color_histogram()
        except Exception as e:
            print(f"{Fore.RED}Error: Processing failed for file: "
                  f"{self.path}: {e}{Style.RESET_ALL}")

    def gauss(self):
        """
        Applies Gaussian blur to the image.
        """
        self.img_gauss = cv2.GaussianBlur(self.img, (21, 21), 0)

        if self.visual:
            plt.imshow(self.img_gauss)
            plt.title("Gaussian Blur")
            plt.show()

    def mask(self):
        """
        Applies a binary mask based on the saturation channel of the image.
        """
        v_hsv = _extract_hsv_channel(self.img, 's')
        v_mask_binary = _binary_threshold(
            v_hsv, threshold=60, object_type='light')
        self.img_masked = _apply_mask(
            self.img, v_mask_binary, mask_color='white')

        if self.visual:
            plt.imshow(self.img_masked)
            plt.title("Mask Applied")
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

    def roi(self):
        v_hsv = _extract_hsv_channel(self.img, 's')
        v_mask_binary = _binary_threshold(
            v_hsv, threshold=85, object_type='light')
        shape_image = _analyze_size(self.img, v_mask_binary, label="")
        self.img_roi = shape_image.copy()

    def pseudo_landmarks(self):
        """
        Detects and draws pseudo-landmarks
            on the image using homology analysis.
        """
        v_hsv = _extract_hsv_channel(self.img, 's')
        v_mask_binary = _binary_threshold(
            v_hsv, threshold=85, object_type='light')
        left, right, center_h = _y_axis_pseudolandmarks(
            self.img, v_mask_binary)

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
        hsv_gray = _extract_hsv_channel(self.img, "s")

        mask = _binary_threshold(hsv_gray, threshold=85, object_type="light")

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
                linestyle="solid",
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
                linestyle="solid",
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
                linestyle="solid",
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
            (self.img_gauss, "_gaussian_blur"),
            (self.img_masked, "_mask_applied"),
            (self.img_roi, "_ROI_detection"),
            (self.img_analyzed, "_analyzed_objects"),
            (self.img_pseudolandmarks, "_pseudolandmarks"),
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
