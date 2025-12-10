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
    except KeyboardInterrupt as e:
        print(e)
    except Exception as e:
        print("Error: ", e)
