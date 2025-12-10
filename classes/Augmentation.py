from colorama import Fore, Style
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

    def __del__(self):
        """
        Augmentation constructor
        """
        pass

    def original(self):
        """"""
        try:
            img = Image.open(self.path)
            img.show()
        except Exception as e:
            print(f"{Fore.RED}"
                  f"Error: {e}"
                  f"{Style.RESET_ALL}")

    def rotation(self):
        """"""
        pass

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
