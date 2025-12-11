from colorama import Fore, Style
from matplotlib import pyplot as plt
from PIL import Image, UnidentifiedImageError
from plantcv import plantcv as pcv

import argparse
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


def process_image(p_path: str):
    if not os.path.isfile(p_path):
        print("Need to be a file")
        sys.exit(1)
    if not is_jpg(p_path):
        print("Needs to be a jpeg/jpg file")
        sys.exit(1)
    v_transformation = Transformation(p_path)
    v_transformation.show()
    return


def process_folder(p_src: str, p_dst: str):
    if not os.path.isdir(p_src):
        print("src existe pas")
        sys.exit(1)
    if not os.path.isdir(p_dst):
        # trying create dir before process all images
        try:
            os.makedirs(p_dst)
        except Exception as e:
            print("Destination folder can't be made", e)
            sys.exit(1)
    
    jpg_files = [f for f in os.listdir(p_src) 
                if is_jpg(os.path.join(p_src, f)) and 
                os.path.isfile(os.path.join(p_src, f))]


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
                        prog='Transformation',
                        description='Apply image transformations using PlantCV')
    
    parser.add_argument('image_path', nargs='?', 
                       help='Direct path to a single image file')
    parser.add_argument('-src', '--source', 
                       help='Source directory containing images')
    parser.add_argument('-dst', '--destination', 
                       help='Destination directory for saving transformations')
    args = parser.parse_args()
    v_is_folder = args.source or args.destination
    v_is_image = args.image_path
    if v_is_image and v_is_folder:
        print("Choisis l'un des deux en faites")
        sys.exit(1)
    if v_is_image:
        process_image(args.image_path)
    elif v_is_folder:
        process_folder(args.source, args.destination)


if __name__ == '__main__':
    main()
