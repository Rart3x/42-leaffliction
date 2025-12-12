import argparse
import glob
import os

from colorama import Fore, Style
from PIL import Image, ImageEnhance, ImageFilter

g_suffixes = {
    "_blurred",
    "_scaled",
    "_contrasted",
    "_illuminated",
    "_rotated",
    "_projected"
}


def is_jpg(file_path: str) -> bool:
    """
    Check if the given file is a JPEG image (.jpg or .jpeg), case-insensitive.

    :param file_path: Path to the file to check.
    :return: True if the file exists
        and ends with .jpg or .jpeg, False otherwise.
    """
    return (os.path.isfile(file_path)
            and file_path.lower().endswith(
                ('.jpg', '.jpeg')))


def zoom(img, zoom_factor):
    """
    Zoom into the image without increasing the final output size.
    The zoom is performed by cropping the center region of the image
    and resizing it back to the original dimensions.

    :param img: The input PIL Image to zoom into.
    :param zoom_factor: The zoom intensity.
                        A value > 1 zooms in (e.g., 1.5 = 150% zoom).
    :return: A new PIL Image representingthe zoomed version of the input image.
    """
    width, height = img.size

    # Compute the target crop size based on the zoom factor
    new_w = int(width / zoom_factor)
    new_h = int(height / zoom_factor)

    # Compute a centered crop box
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    # Crop the center of the image and resize back to the original resolution
    cropped_img = img.crop((left, top, right, bottom))

    return cropped_img.resize((width, height), Image.LANCZOS)


class Augmentation:
    """
    Augmentation class that applies several transformations to an image
    such as rotation, blur, contrast enhancement, illumination, scaling,
    and projective transformation.
    """
    def __init__(self, p_path: str, p_visual_mode: bool = False):
        """
        Constructor. Loads the image and applies all augmentations.

        :param p_path: Path to the input image.
        """
        # Perspective coefficients
        v_coeffs = [
            1, 0.2, 0,
            0, 1, 0,
            0.004, 0.004
        ]

        self.visual_mode = p_visual_mode
        self.path = p_path
        self.path_without_extension = os.path.splitext(p_path)[0]
        self.img = Image.open(self.path)

        # Enhancers
        v_enhancer_brightness = ImageEnhance.Brightness(self.img)
        v_enhancer_contrast = ImageEnhance.Contrast(self.img)

        # Image transformations
        self.img_rotated = self.img.rotate(180)
        self.img_blurred = self.img.filter(ImageFilter.GaussianBlur(radius=5))
        self.img_contrasted = v_enhancer_contrast.enhance(4)
        self.img_illuminated = v_enhancer_brightness.enhance(3)
        self.img_scaled = zoom(self.img, 2)
        self.img_projected = self.img.transform(
            self.img.size,
            Image.PERSPECTIVE,
            v_coeffs,
            Image.BICUBIC
        )

    def __del__(self):
        """
        Destructor.
        """
        pass

    def rotation(self):
        """
        Save the rotated image.
        """
        self.img_rotated.save(self.path_without_extension + "_rotated.JPG")
        print(f"{Fore.GREEN}Image : "
              f"{self.path_without_extension}_rotated.JPG "
              f"successfully created"
              f"{Style.RESET_ALL}")

    def blur(self):
        """
        Save the blurred image.
        """
        self.img_blurred.save(self.path_without_extension + "_blured.JPG")
        print(f"{Fore.GREEN}Image : "
              f"{self.path_without_extension}_blured.JPG "
              f"successfully created"
              f"{Style.RESET_ALL}")

    def contrast(self):
        """
        Save the contrasted image.
        """
        self.img_contrasted.save(self.path_without_extension +
                                 "_contrasted.JPG")
        print(f"{Fore.GREEN}Image : "
              f"{self.path_without_extension}_contrasted.JPG "
              f"successfully created"
              f"{Style.RESET_ALL}")

    def scaling(self):
        """
        Save the scaled image.
        """
        self.img_scaled.save(self.path_without_extension + "_scaled.JPG")
        print(f"{Fore.GREEN}Image : "
              f"{self.path_without_extension}_scaled.JPG "
              f"successfully created"
              f"{Style.RESET_ALL}")

    def illumination(self):
        """
        Save the illuminated image.
        """
        self.img_illuminated.save(self.path_without_extension +
                                  "_illuminated.JPG")
        print(f"{Fore.GREEN}Image : "
              f"{self.path_without_extension}_illuminated.JPG "
              f"successfully created"
              f"{Style.RESET_ALL}")

    def projective(self):
        """
        Save the projectively transformed image.
        """
        self.img_projected.save(self.path_without_extension + "_projected.JPG")
        print(f"{Fore.GREEN}Image : "
              f"{self.path_without_extension}_projected.JPG "
              f"successfully created"
              f"{Style.RESET_ALL}")

    def show_all(self):
        """
        Display all augmented images side by side in a single window.
        Also saves the final collage as a single image.
        """
        # List of all images
        images = [
            self.img,
            self.img_rotated,
            self.img_blurred,
            self.img_contrasted,
            self.img_illuminated,
            self.img_scaled,
            self.img_projected
        ]

        # Resize all images to match the original image size
        base_w, base_h = self.img.size
        images = [img.resize((base_w, base_h)) for img in images]

        # Compute the size of the final collage
        total_width = base_w * len(images)
        max_height = base_h

        # Create a blank canvas
        collage = Image.new("RGB", (total_width, max_height))

        # Paste all images horizontally
        x_offset = 0
        for img in images:
            collage.paste(img, (x_offset, 0))
            x_offset += base_w

        self.rotation()
        self.blur()
        self.contrast()
        self.scaling()
        self.illumination()
        self.projective()

        # Display the final collage
        if self.visual_mode:
            collage.show()


def main():
    parser = argparse.ArgumentParser(prog='Augmentation')

    parser.add_argument(
        '-v', '--visual',
        action='store_true',
        help='Enable the rendering of the augmented images'
    )

    v_args, v_remaining = parser.parse_known_args()

    file_paths = []

    for arg in v_remaining:
        file_paths.extend(glob.glob(arg))

    if not file_paths:
        print(f"{Fore.RED}No valid file paths provided.{Style.RESET_ALL}")
        return

    for path in file_paths:
        if not os.path.isfile(path):
            print(f"{Fore.RED}Error: {path} is not a file{Style.RESET_ALL}")
            continue

        if not is_jpg(path):
            print(f"{Fore.RED}Error {path}: not a JPG file{Style.RESET_ALL}")
            continue

        base_no_ext = os.path.splitext(path)[0]

        # Skip already augmented images
        if any(base_no_ext.endswith(sfx) for sfx in g_suffixes):
            continue

        aug = Augmentation(path, v_args.visual)
        aug.show_all()


if __name__ == '__main__':
    main()
