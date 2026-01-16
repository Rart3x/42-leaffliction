#!/usr/bin/env python3
"""
Split training data into train and test sets for model evaluation.
This script creates a test set by copying
a percentage of images from each class
to a separate test directory, maintaining class balance.
"""

import argparse
import os
import random
import shutil

from pathlib import Path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        prog='create_test_split',
        description='Create test set from training data by splitting images'
    )

    parser.add_argument('source_dir',
                        help='Source directory containing '
                             'training images (e.g., leaves/images/)')
    parser.add_argument('test_dir',
                        help='Destination directory for test images')
    parser.add_argument('--test-size', type=float, default=0.15,
                        help='Fraction of images to use for '
                             'testing (default: 0.15)')
    parser.add_argument('--min-test-per-class', type=int, default=15,
                        help='Minimum number of test images '
                             'per class (default: 15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files instead of moving '
                             'them (keeps original data intact)')

    args = parser.parse_args()
    return args


def get_image_files(directory):
    """Get all image files from a directory"""
    valid_extensions = {'.jpg', '.jpeg', '.png',
                        '.bmp', '.gif', '.JPG', '.JPEG', '.PNG'}
    return [f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
            and Path(f).suffix in valid_extensions]


def create_test_split(source_dir,
                      test_dir,
                      test_size=0.15,
                      min_test_per_class=15,
                      seed=42,
                      copy_mode=False):
    """
    Create test set by splitting images from source directory

    :param source_dir: Source directory with class subdirectories
    :param test_dir: Destination test directory
    :param test_size: Fraction of images to use for testing
    :param min_test_per_class: Minimum number of test images per class
    :param seed: Random seed for reproducibility
    :param copy_mode: If True, copy files; if False, move files
    """
    random.seed(seed)

    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    os.makedirs(test_dir, exist_ok=True)

    class_dirs = [d for d in os.listdir(source_dir)
                  if os.path.isdir(os.path.join(source_dir, d))]

    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {source_dir}")

    print("\n" + "="*80)
    print("CREATE TEST SET SPLIT")
    print("="*80 + "\n")
    print(f"Source directory: {source_dir}")
    print(f"Test directory: {test_dir}")
    print(f"Test size: {test_size*100:.1f}%")
    print(f"Minimum test per class: {min_test_per_class}")
    print(f"Mode: {'COPY' if copy_mode else 'MOVE'}")
    print(f"Random seed: {seed}\n")

    total_train = 0
    total_test = 0
    class_stats = []

    for class_name in sorted(class_dirs):
        source_class_dir = os.path.join(source_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        os.makedirs(test_class_dir, exist_ok=True)

        images = get_image_files(source_class_dir)

        if not images:
            print(f"⚠ Warning: No images found in {class_name}, skipping...")
            continue

        num_test = max(min_test_per_class, int(len(images) * test_size))

        num_test = min(num_test, len(images))

        random.shuffle(images)
        test_images = images[:num_test]

        operation = shutil.copy2 if copy_mode else shutil.move

        for img in test_images:
            source_path = os.path.join(source_class_dir, img)
            dest_path = os.path.join(test_class_dir, img)
            operation(source_path, dest_path)

        num_train = len(images) - num_test
        total_train += num_train
        total_test += num_test

        class_stats.append({
            'class': class_name,
            'total': len(images),
            'train': num_train,
            'test': num_test,
            'test_pct': (num_test / len(images)) * 100
        })

        print(f"✓ {class_name:25s}: {num_test:4d} test / {num_train:4d} train "
              f"({num_test/len(images)*100:5.1f}%)")

    print("\n" + "-"*80)
    total = total_train + total_test
    if total > 0:
        total_test_pct = f"{(total_test / total) * 100:5.1f}%"
    else:
        total_test_pct = "  0.0%"
    print(f"{'TOTAL':25s}: {total_test:4d} test / {total_train:4d} train "
          f"({total_test_pct})")
    print("-"*80 + "\n")

    if total_test < 100:
        print(f"WARNING: Test set has only {total_test} "
              f"images (minimum requirement: 100)")
        print("Consider using a larger test_size "
              "or ensuring enough images in source directory.\n")
    else:
        print(f"Test set has {total_test} "
              f"images (meets minimum requirement of 100)\n")

    print("="*80)
    print("TEST SET CREATION COMPLETE")
    print("="*80 + "\n")

    return total_test, total_train


def main():
    """Main function"""
    args = parse_args()

    try:
        create_test_split(
            source_dir=args.source_dir,
            test_dir=args.test_dir,
            test_size=args.test_size,
            min_test_per_class=args.min_test_per_class,
            seed=args.seed,
            copy_mode=args.copy
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
