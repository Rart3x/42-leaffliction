from colorama import Fore, Style
from PIL import Image, UnidentifiedImageError

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
        pass

    def __del__(self):
        pass


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
        print(f"{Fore.RED}Error: file must be a JPG/JPEG image{Style.RESET_ALL}")
        return

    v_path = sys.argv[1]

    transformation = Transformation(v_path)


if __name__ == '__main__':
    main()
