import os


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
