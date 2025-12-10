import os
import pathlib
import sys
import matplotlib.pyplot as plt


def get_images(p_directory: pathlib.Path):
    """
    Return a list of path of images of
    a directory.

    :param p_directory: Path of the targeted directory
    :type p_directory: pathlib.Path
    """
    return [f for f in os.listdir(p_directory)]


def get_subdirectories(p_directory: pathlib.Path):
    """
    Return a list of subdirectories absolute
    path of the targeted directory.

    :param p_directory: Initial path of directory
    :type p_directory: pathlib.Path
    """
    return [x for x in p_directory.iterdir() if x.is_dir()]


def main():
    if len(sys.argv) != 2 or sys.argv[1] == "":
        print("usage: python3 ./Distribution.py <directory_path>")
        return 1
    v_path = pathlib.Path(sys.argv[1])
    v_sub_directories = get_subdirectories(v_path)
    v_images_count = []
    for sub in v_sub_directories:
        v_images_count.append(len(get_images(sub)))
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.pie(v_images_count, labels=[sub.name for sub in v_sub_directories])
    ax2.bar([sub.name for sub in v_sub_directories], v_images_count)
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("Error: ", e)
