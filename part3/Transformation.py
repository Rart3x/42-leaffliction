from colorama import Fore, Style
from matplotlib import pyplot as plt
from PIL import Image, UnidentifiedImageError
from plantcv import plantcv as pcv

import argparse
import cv2
import numpy as np
import os


def is_jpg(path: str) -> bool:
    """
    Checks if the given file is a valid JPG/JPEG image.
    """
    try:
        with Image.open(path) as img:
            return img.format == "JPEG"
    except (UnidentifiedImageError, OSError):
        return False


def folder(src: str, dst: str) -> list[str]:
    """
    Collects all JPG/JPEG images from a source directory and ensures
    the destination directory exists.
    """
    if not os.path.isdir(dst):
        try:
            os.makedirs(dst)
        except OSError as exc:
            print(
                f"{Fore.RED}Error: destination folder can't be created: "
                f"{exc}{Style.RESET_ALL}"
            )
            return []

    return [
        os.path.abspath(os.path.join(src, file))
        for file in os.listdir(src)
        if os.path.isfile(os.path.join(src, file))
        and is_jpg(os.path.join(src, file))
    ]


def parse_input():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(prog="Augmentation")

    parser.add_argument(
        "image_path",
        nargs="?",
        help="Path to a single image file or a directory",
    )

    parser.add_argument(
        "-dst",
        "--destination",
        help="Destination directory for saving transformations",
    )

    parser.add_argument(
        "-v",
        "--visual",
        action="store_true",
        help="Enable visual rendering",
    )

    args = parser.parse_args()
    return args.image_path, args


class Transformation:
    """
    Represents an image transformation pipeline.
    """

    def __init__(self, path: str, visual: bool):
        self.img, self.path, self.filename = pcv.readimage(path)
        self.visual = visual

        self.img_gauss = None
        self.img_roi = None
        self.img_masked = None
        self.img_analyzed = None
        self.img_pseudolandmarks = None
        self.img_color_histogram = None

    def image(self):
        """
        Runs all transformations.
        """
        self.gauss()
        self.roi()
        self.mask()
        self.analyze()
        self.pseudo_landmarks()
        self.color_histogram()

    def gauss(self):
        self.img_gauss = pcv.gaussian_blur(
            img=self.img,
            ksize=(21, 21),
            sigma_x=0,
            sigma_y=0,
        )

        if self.visual:
            plt.imshow(self.img_gauss)
            plt.title("Gaussian Blur")
            plt.show()

    def mask(self):
        hsv = pcv.rgb2gray_hsv(self.img, channel="s")
        mask = pcv.threshold.binary(
            gray_img=hsv,
            threshold=85,
            object_type="light",
        )

        self.img_masked = pcv.apply_mask(
            img=self.img,
            mask=mask,
            mask_color="white",
        )

        if self.visual:
            plt.imshow(self.img_masked)
            plt.title("Mask Applied")
            plt.show()

    def roi(self):
        gray = pcv.rgb2gray_lab(self.img, channel="a")
        mask = pcv.threshold.binary(
            gray_img=gray,
            threshold=100,
            object_type="light",
        )

        roi = pcv.roi.from_binary_image(
            img=self.img,
            bin_img=mask,
        )

        self.img_roi = self.img.copy()

        for contour in roi.contours:
            cv2.drawContours(
                self.img_roi,
                contour,
                -1,
                (255, 0, 0),
                3,
            )

        if self.visual:
            plt.imshow(
                cv2.cvtColor(self.img_roi, cv2.COLOR_BGR2RGB)
            )
            plt.title("Automatic ROI Detection")
            plt.show()

    def analyze(self):
        hsv = pcv.rgb2gray_hsv(self.img, channel="s")
        mask = pcv.threshold.binary(
            gray_img=hsv,
            threshold=85,
            object_type="light",
        )

        analyzed = pcv.analyze.size(
            img=self.img,
            labeled_mask=mask,
            label="",
        )

        self.img_analyzed = analyzed.copy()

    def pseudo_landmarks(self):
        hsv = pcv.rgb2gray_hsv(self.img, channel="s")
        mask = pcv.threshold.binary(
            gray_img=hsv,
            threshold=85,
            object_type="light",
        )

        left, right, center = pcv.homology.y_axis_pseudolandmarks(
            img=self.img,
            mask=mask,
        )

        self.img_pseudolandmarks = self.img.copy()

        for group, color in (
            (left, (255, 0, 0)),
            (right, (0, 255, 0)),
            (center, (0, 0, 255)),
        ):
            for point in group:
                x, y = map(int, point[0])
                cv2.circle(
                    self.img_pseudolandmarks,
                    (x, y),
                    5,
                    color,
                    -1,
                )

        if self.visual:
            plt.imshow(
                cv2.cvtColor(
                    self.img_pseudolandmarks,
                    cv2.COLOR_BGR2RGB,
                )
            )
            plt.title(
                "Pseudo-landmarks (Blue: Left, Green: Right, Red: Center)"
            )
            plt.show()

    def color_histogram(self):
        hsv = pcv.rgb2gray_hsv(self.img, channel="s")
        mask = pcv.threshold.binary(
            gray_img=hsv,
            threshold=85,
            object_type="light",
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, color in enumerate(("b", "g", "r")):
            hist = cv2.calcHist(
                [self.img],
                [i],
                mask,
                [256],
                [0, 256],
            )
            axes[0].plot(hist, color=color)

        axes[0].set_title("RGB Histogram")

        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        for i, label in enumerate(("L", "A", "B")):
            hist = cv2.calcHist(
                [lab],
                [i],
                mask,
                [256],
                [0, 256],
            )
            axes[1].plot(hist, label=label)

        axes[1].set_title("LAB Histogram")
        axes[1].legend()

        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        for i, label in enumerate(("H", "S", "V")):
            hist = cv2.calcHist(
                [hsv_img],
                [i],
                mask,
                [256],
                [0, 256],
            )
            axes[2].plot(hist, label=label)

        axes[2].set_title("HSV Histogram")
        axes[2].legend()

        plt.tight_layout()
        fig.canvas.draw()

        buf = np.frombuffer(
            fig.canvas.buffer_rgba(),
            dtype=np.uint8,
        )

        width, height = fig.canvas.get_width_height()
        buf = buf.reshape(height, width, 4)

        self.img_color_histogram = cv2.cvtColor(
            buf,
            cv2.COLOR_RGBA2BGR,
        )

        if self.visual:
            plt.show()
        else:
            plt.close(fig)

    def save(self, dst: str):
        images = [
            (self.img, "_original"),
            (self.img_gauss, "_gaussian_blur"),
            (self.img_masked, "_mask"),
            (self.img_roi, "_roi"),
            (self.img_analyzed, "_analyzed"),
            (self.img_pseudolandmarks, "_pseudolandmarks"),
            (self.img_color_histogram, "_histogram"),
        ]

        base = os.path.splitext(self.filename)[0]

        for img, suffix in images:
            if img is None:
                continue

            path = os.path.join(dst, f"{base}{suffix}.jpg")
            cv2.imwrite(path, img)
            print(f"{Fore.GREEN}Saved: {path}{Style.RESET_ALL}")


def main():
    path, args = parse_input()

    if not path:
        print(f"{Fore.RED}Error: no input path provided{Style.RESET_ALL}")
        return

    if os.path.isdir(path):
        if not args.destination:
            print(
                f"{Fore.RED}Error: destination required for folders"
                f"{Style.RESET_ALL}"
            )
            return

        files = folder(path, args.destination)
        if not files:
            print(f"{Fore.RED}Error: empty directory{Style.RESET_ALL}")
            return

        for file in files:
            transform = Transformation(file, args.visual)
            transform.image()
            transform.save(args.destination)

    elif os.path.isfile(path):
        if not is_jpg(path):
            print(
                f"{Fore.RED}Error: input must be JPG/JPEG{Style.RESET_ALL}"
            )
            return

        transform = Transformation(path, args.visual)
        transform.image()
        transform.color_histogram()

    else:
        print(f"{Fore.RED}Error: invalid path{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
