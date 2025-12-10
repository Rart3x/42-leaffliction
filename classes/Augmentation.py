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
        self.path = p_path
        self.img = Image.open(self.path)

        enhancer = ImageEnhance.Contrast(self.img)

        self.img_rotated = self.img.rotate(180)
        self.img_blured = self.img.filter(ImageFilter.GaussianBlur(radius=5))
        self.img_contrasted = enhancer.enhance(4)
        self.img_scaled = self.img.resize((self.img.width * 2, self.img.height * 2), Image.LANCZOS)

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

    def blur(self):
        """"""
        self.img_blured.show()

    def contrast(self):
        """"""
        self.img_contrasted.show()

    def scaling(self):
        """"""
        self.img_scaled.show()

    def illumination(self):
        """"""
        pass

    def projective(self):
        """"""
        pass
