from PIL import Image


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
        self.image_rotated = self.img.rotate(180)

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
        self.image_rotated.show()

    def blur(self):
        """"""
        pass

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
