from colorama import Fore, Style

import os
import sys


def main():
    """
    Main function
    """
    if len(sys.argv) != 2:
        print(f"{Fore.RED}Usage: python augmentation.py file_path{Style.RESET_ALL}")
        return

    if not os.path.isfile(sys.argv[1]):
        print(f"{Fore.RED}Error: argument need to be a file path{Style.RESET_ALL}")
        return



if __name__ == '__main__':
    main()