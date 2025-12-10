from PIL import Image, ImageFilter


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
        self.img_rotated = self.img.rotate(180)
        self.img_blured = self.img.filter(ImageFilter.GaussianBlur(radius=5))


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
        pass

    def scaling(self):
        """"""
        pass

    def illumination(self):
        """"""
        pass

    def projective(self):
        """"""
        pass
