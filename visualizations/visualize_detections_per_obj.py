#!/usr/bin/env python3
"""
Script to copy images from obj_<id>/rgb/<image> structure to all_images/obj_<id>_<image>
and create a histogram of image counts per object
"""

import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict


def copy_images_to_flat_structure(source_dir, target_dir, dry_run: bool = True):
    """
    Copy images from obj_<id>/rgb/<image> structure to all_images/obj_<id>_<image>

    Args:
        source_dir (Path): Directory containing the obj_<id> folders (default: current directory)
        target_dir (Path): Target directory name (default: 'all_images')

    Returns:
        dict: Dictionary with obj_id as keys and image counts as values
    """
    # Convert to Path objects if strings are passed
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    # Common image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']

    # Find all obj_* directories
    obj_dirs = list(source_dir.glob('obj_*'))

    if not obj_dirs:
        print(f"No obj_* directories found in {source_dir}")
        return {}

    copied_count = 0
    image_counts = defaultdict(int)  # Track image count per object

    for obj_dir in obj_dirs:
        # Extract the obj_id from directory name
        obj_id = obj_dir.name

        # Check if rgb subdirectory exists
        rgb_dir = obj_dir / 'rgb'
        if not rgb_dir.exists():
            print(f"Warning: {rgb_dir} does not exist, skipping...")
            continue

        print(f"Processing {obj_id}...")

        # Find all image files in the rgb directory
        for extension in image_extensions:
            image_files = list(rgb_dir.glob(extension))

            for image_file in image_files:
                # Get just the image filename
                image_name = image_file.name

                # Create new filename: obj_<id>_<image>
                new_filename = f"{obj_id}_{image_name}"
                target_path = target_dir / new_filename

                try:
                    # Copy the file
                    if not dry_run:
                        shutil.copy2(image_file, target_path)
                        print(f"  Copied: {image_file} -> {target_path}")
                    copied_count += 1
                    image_counts[obj_id] += 1
                except Exception as e:
                    print(f"  Error copying {image_file}: {e}")

    print(f"\nCompleted! Copied {copied_count} images to {target_dir}/")
    return dict(image_counts)


def create_histogram(image_counts, dataset, split, output_file):
    """
    Create and save a histogram showing image count per object

    Args:
        image_counts (dict): Dictionary with obj_id as keys and image counts as values
        output_file (Path): Output filename for the histogram (default: 'histogram.png')
    """
    if not image_counts:
        print("No image counts to plot.")
        return

    # Convert to Path object if string is passed
    output_file = Path(output_file)

    # Sort objects by ID for consistent ordering
    sorted_objects = sorted(image_counts.keys())
    counts = [image_counts[obj_id] for obj_id in sorted_objects]

    # Create the histogram
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(sorted_objects)), counts, alpha=0.7, color='skyblue', edgecolor='navy')

    # Customize the plot
    plt.xlabel('Object ID')
    plt.ylabel('Number of Templates')
    plt.title(f'Templates per object: {dataset}-{split}')
    plt.grid(axis='y', alpha=0.3)

    # Set x-axis labels (rotate if there are many objects)
    if len(sorted_objects) <= 20:
        plt.xticks(range(len(sorted_objects)), sorted_objects, rotation=45, ha='right')
    else:
        # For many objects, show every nth label
        step = max(1, len(sorted_objects) // 20)
        indices = range(0, len(sorted_objects), step)
        plt.xticks(indices, [sorted_objects[i] for i in indices], rotation=45, ha='right')

    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=8)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Histogram saved as: {output_file}")

    # Print summary statistics
    total_images = sum(counts)
    avg_images = total_images / len(counts) if counts else 0
    min_images = min(counts) if counts else 0
    max_images = max(counts) if counts else 0

    print(f"\nStatistics:")
    print(f"  Total objects: {len(sorted_objects)}")
    print(f"  Total images: {total_images}")
    print(f"  Average images per object: {avg_images:.1f}")
    print(f"  Min images per object: {min_images}")
    print(f"  Max images per object: {max_images}")


def main():
    """
    Main function - you can modify these parameters as needed
    """

    methods = [
        'hart',
        'hart_symmetric',
        'hart_imblearn',
        'hart_imblearn_adapted'
    ]
    for _method in methods:
        experiment = f'1nn-{_method}'
        dataset_sequences = [
            ('hope', 'onboarding_static'),
            ('hope', 'onboarding_dynamic'),
            ('handal', 'onboarding_static'),
            ('handal', 'onboarding_dynamic'),
            ('icbin', 'train'),
            ('lmo', 'train'),
            ('tless', 'train_primesense'),
        ]

        for dataset, split in dataset_sequences:
            relative_path = Path(dataset) / split
            SOURCE_DIRECTORY = Path('/mnt/personal/jelint19/cache/detections_templates_cache/') / experiment / relative_path
            TARGET_DIRECTORY = Path('/mnt/personal/jelint19/results/condensation') / experiment / relative_path
            HISTOGRAM_FILE = TARGET_DIRECTORY.parent.parent / Path(f'histogram_{dataset}-{split}.png')
            print("Image Directory Flattening Script with Histogram")
            print("=" * 50)
            print(f"Source directory: {SOURCE_DIRECTORY.resolve()}")
            print(f"Target directory: {TARGET_DIRECTORY.resolve()}")
            print(f"Histogram file: {HISTOGRAM_FILE.resolve()}")
            print()

            # Copy images and get counts
            image_counts = copy_images_to_flat_structure(SOURCE_DIRECTORY, TARGET_DIRECTORY, dry_run=True)

            # Create histogram
            create_histogram(image_counts, dataset, split, HISTOGRAM_FILE)


if __name__ == "__main__":
    main()