import os

from colorama import Fore, Style
from PIL import Image, ImageEnhance, ImageFilter


class Augmentation:
    """
    Augmentation class
    """
    def __init__(self, p_path: str):
        """
        Augmentation constructor

        :param p_path: path to the image

        :return: None
        """
        v_coeffs = [
            1, 0.2, 0,
            0, 1, 0,
            0.004, 0.004
        ]

        self.path = p_path
        self.path_without_extension = os.path.splitext(p_path)[0]
        self.img = Image.open(self.path)

        v_enhancer_brightness = ImageEnhance.Brightness(self.img)
        v_enhancer_contrast = ImageEnhance.Contrast(self.img)

        self.img_rotated = self.img.rotate(180)
        self.img_blured = self.img.filter(ImageFilter.GaussianBlur(radius=5))
        self.img_contrasted = v_enhancer_contrast.enhance(4)
        self.img_illuminated = v_enhancer_brightness.enhance(3)
        self.img_scaled = self.img.resize(
            (
                self.img.width * 2,
                self.img.height * 2),
            Image.LANCZOS
        )
        self.img_projected = self.img.transform(
            self.img.size,
            Image.PERSPECTIVE,
            v_coeffs,
            Image.BICUBIC)

    def __del__(self):
        """
        Augmentation constructor
        """
        pass

    def original(self):
        """"""
        self.img.show()

    def rotation(self):
        """"""
        self.img_rotated.show()
        self.img_rotated.save(self.path_without_extension + "_rotated.JPG")
        print(f"{Fore.GREEN}Image :",
              self.path_without_extension +
              f"_rotated.JPG successfully created"
              f"{Style.RESET_ALL}")

    def blur(self):
        """"""
        self.img_blured.show()
        self.img_blured.save(self.path_without_extension + "_blured.JPG")
        print(f"{Fore.GREEN}Image :",
              self.path_without_extension +
              f"_blured.JPG successfully created"
              f"{Style.RESET_ALL}")

    def contrast(self):
        """"""
        self.img_contrasted.show()
        self.img_contrasted.save(self.path_without_extension +
                                 "_contrasted.JPG")
        print(f"{Fore.GREEN}Image :",
              self.path_without_extension +
              f"_contrasted.JPG successfully created"
              f"{Style.RESET_ALL}")

    def scaling(self):
        """"""
        self.img_scaled.show()
        self.img_scaled.save(self.path_without_extension +
                             "_scaled.JPG")
        print(f"{Fore.GREEN}Image :",
              self.path_without_extension +
              f"_scaled.JPG successfully created"
              f"{Style.RESET_ALL}")

    def illumination(self):
        """"""
        self.img_illuminated.show()
        self.img_illuminated.save(self.path_without_extension +
                                  "_illuminated.JPG")
        print(f"{Fore.GREEN}Image :",
              self.path_without_extension +
              f"_illuminated.JPG successfully created"
              f"{Style.RESET_ALL}")

    def projective(self):
        """"""
        self.img_projected.show()
        self.img_projected.save(self.path_without_extension +
                                "_projected.JPG")
        print(f"{Fore.GREEN}Image :",
              self.path_without_extension +
              f"_projected.JPG successfully created"
              f"{Style.RESET_ALL}")
