from colorama import Fore, Style
from matplotlib import pyplot as plt
from PIL import Image, UnidentifiedImageError
from plantcv import plantcv as pcv

import os
import sys


def is_jpg(path: str) -> bool:
    """
    Check if a file is a JPG/JPEG image

    :param path: The image path

    :return: bool
    """
    try:
        with Image.open(path) as img:
            return img.format == "JPEG"
    except (UnidentifiedImageError, OSError):
        return False


class Transformation:
    def __init__(self, path: str):
        self.img, self.path, self.filename = pcv.readimage(path)

    def __del__(self):
        pass

    def show(self):
        plt.imshow(self.img)
        plt.show()


def main():
    """
    Main function
    """
    if len(sys.argv) != 2:
        print(f"{Fore.RED}"
              f"Usage: python augmentation.py file_path"
              f"{Style.RESET_ALL}")
        return

    if not os.path.isfile(sys.argv[1]):
        print(f"{Fore.RED}"
              f"Error: argument must be a file path"
              f"{Style.RESET_ALL}")
        return

    if not is_jpg(sys.argv[1]):
        print(f"{Fore.RED}"
              f"Error: file must be a JPG/JPEG image"
              f"{Style.RESET_ALL}")
        return

    v_path = sys.argv[1]

    transformation = Transformation(v_path)

    transformation.show()


if __name__ == '__main__':
    main()
