from colorama import Fore, Style

import sys


def main():
    """
    Main function
    """
    if len(sys.argv) != 2:
        print(f"{Fore.RED}Usage: python augmentation.py path{Style.RESET_ALL}")


if __name__ == '__main__':
    main()