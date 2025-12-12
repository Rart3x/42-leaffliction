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

    def gauss(self):
        v_gauss = pcv.gaussian_blur(img=self.img, ksize=(21,21), sigma_x=0, sigma_y=0)
        plt.imshow(v_gauss)
        plt.show()
    
    def mask(self):
        v_hsv = pcv.rgb2gray_hsv(rgb_img=self.img, channel='s')
        v_mask_binary = pcv.threshold.binary(gray_img=v_hsv, threshold=85, 
                                            object_type='light')
        v_masked = pcv.apply_mask(img=self.img, mask=v_mask_binary, mask_color='white')
        
        plt.imshow(v_masked)
        plt.show()

    def roi(self):
        v_roi = pcv.roi.custom(img=self.img, vertices=[[1190,490], [1470,830], 
                                        [1565,1460], [1130,1620], 
                                        [920,1430], [890,950]])
        plt.imshow(v_roi)
        plt.show()

    def show(self):
        plt.imshow(self.img)
        plt.show()


def process_image(p_path: str, p_type: str):
    if not os.path.isfile(p_path):
        print("Need to be a file")
        sys.exit(1)
    if not is_jpg(p_path):
        print("Needs to be a jpeg/jpg file")
        sys.exit(1)
    v_transformation = Transformation(p_path)
    v_transformation.show()
    return


def process_folder(p_src: str, p_dst: str, p_type: str):
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
    parser.add_argument('-t', '--type', 
                        choices=['gauss', 'mask', 'roi', 'analyze_obj'
                                 , 'pseudolandmarks', 'color_histogram'])
    args = parser.parse_args()
    v_is_folder = args.source or args.destination
    v_is_image = args.image_path
    v_type = args.type
    if not v_type:
        print("choisis un type")
        sys.exit(1)
    if v_is_image and v_is_folder:
        print("Choisis l'un des deux en faites")
        sys.exit(1)
    if v_is_image:
        process_image(args.image_path, v_type)
    elif v_is_folder and args.source and args.destination:
        process_folder(args.source, args.destination, v_type)


if __name__ == '__main__':
    main()
