import pathlib
import sys
import matplotlib.pyplot as plt


def get_images(p_directory: pathlib.Path) -> list[pathlib.Path]:
    """
    Return a list of image file paths in the given directory.

    :param p_directory: Path of the targeted directory
    :type p_directory: pathlib.Path
    :return: List of pathlib.Path objects for image files in the directory
    :rtype: list[pathlib.Path]
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp',
                        '.gif', '.tiff', '.webp'}
    return [
        f for f in p_directory.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]


def get_subdirectories(p_directory: pathlib.Path) -> list[pathlib.Path]:
    """
    Return a list of subdirectories absolute
    path of the targeted directory.

    :param p_directory: Initial path of directory
    :type p_directory: pathlib.Path
    :return: List of Path objects representing subdirectories
    :rtype: list[pathlib.Path]
    """
    return [x for x in p_directory.iterdir() if x.is_dir()]


def main():
    if len(sys.argv) != 2 or sys.argv[1] == "":
        print("usage: python3 ./Distribution.py <directory_path>")
        sys.exit(1)

    v_path = pathlib.Path(sys.argv[1])

    if not v_path.exists():
        print(f"Error: Directory '{v_path}' does not exist")
        sys.exit(1)
    if not v_path.is_dir():
        print(f"Error: '{v_path}' is not a directory")
        sys.exit(1)

    v_sub_directories = get_subdirectories(v_path)

    if not v_sub_directories:
        print("Error: No subdirectories found in the specified directory")
        sys.exit(1)
    v_images_count = []
    for sub in v_sub_directories:
        v_images_count.append(len(get_images(sub)))

    if sum(v_images_count) == 0:
        print("Error: No images found in any subdirectory")
        sys.exit(1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Image Distribution Across Directories',
                 fontsize=16,
                 fontweight='bold'
                 )
    plt.subplots_adjust(wspace=0.3)

    sub_names = [sub.name for sub in v_sub_directories]
    ax1.pie(v_images_count,
            labels=sub_names,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10},
            pctdistance=0.85,
            labeldistance=1.05
            )
    ax1.set_title('Pie Chart', fontsize=12, fontweight='bold', pad=20)

    bars = ax2.bar(sub_names,
                   v_images_count,
                   width=0.7,
                   )
    ax2.set_title('Bar Chart', fontsize=12, fontweight='bold', pad=20)
    ax2.set_xlabel('Directory', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=9)
    ax2.tick_params(axis='y', labelsize=9)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center',
                 va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print("Error:", e)
