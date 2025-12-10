from colorama import Fore, Style

import os
import sys


def main():
    """
    Main function
    """
    if len(sys.argv) != 2:
        print(f"{Fore.RED}"
              f"Usage: python augmentation.py file_path"
              f"{Style.RESET_ALL}")
        return

    if not os.path.isfile(sys.argv[1]):
        print(f"{Fore.RED}"
              f"Error: argument need to be a file path"
              f"{Style.RESET_ALL}")
        return


if __name__ == '__main__':
    main()
