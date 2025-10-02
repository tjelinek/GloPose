"""
Script to copy images from obj_<id>/rgb/<image> structure to all_images/obj_<id>_<image>
and create a histogram of image counts per object with percentage overlay
"""

import shutil
from itertools import product
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict


def copy_images_to_flat_structure(source_dir, target_dir, dry_run: bool = True):

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
    obj_dirs = list(source_dir.glob('obj_*'))

    if not obj_dirs:
        print(f"No obj_* directories found in {source_dir}")
        return {}

    copied_count = 0
    image_counts = defaultdict(int)

    for obj_dir in obj_dirs:
        obj_id = obj_dir.name
        rgb_dir = obj_dir / 'rgb'

        if not rgb_dir.exists():
            print(f"Warning: {rgb_dir} does not exist, skipping...")
            continue

        print(f"Processing {obj_id}...")

        for extension in image_extensions:
            for image_file in rgb_dir.glob(extension):
                image_name = image_file.name
                new_filename = f"{obj_id}_{image_name}"
                target_path = target_dir / new_filename

                try:
                    if not dry_run:
                        shutil.copy2(image_file, target_path)
                        print(f"  Copied: {image_file} -> {target_path}")
                    copied_count += 1
                    image_counts[obj_id] += 1
                except Exception as e:
                    print(f"  Error copying {image_file}: {e}")

    print(f"\nCompleted! Copied {copied_count} images to {target_dir}/")
    return dict(image_counts)


def get_total_counts(original_dataset_path, dataset, split):
    path_to_split = original_dataset_path / dataset / split
    total_counts = defaultdict(int)

    sequences = sorted([d for d in path_to_split.iterdir() if d.is_dir()])

    for sequence in sequences:
        rgb_folder = sequence / 'rgb'
        if not rgb_folder.exists():
            continue

        rgb_files = list(rgb_folder.glob('*'))

        # Extract object ID from sequence (assumes one object per sequence)
        # This matches the logic in the condensation code
        obj_id = f"{sequence.name}"  # Simplified - adjust if needed
        if 'obj_' not in obj_id:
            obj_id = f'obj_{obj_id}'
        total_counts[obj_id] += len(rgb_files)

    if split == 'onboarding_static':
        all_objs = {sequence.name.replace('_up', '').replace('_down', '') for sequence in sequences}
        for obj in all_objs:
            up_total = total_counts.get(f'{obj}_up', 0)
            down_total = total_counts.get(f'{obj}_down', 0)

            total_counts.pop(f'{obj}_up', None)
            total_counts.pop(f'{obj}_down', None)

            total_counts[obj] = up_total + down_total

    return dict(total_counts)


def create_histogram(image_counts, dataset, split, output_file, experiment_name, total_counts=None):
    if not image_counts:
        print("No image counts to plot.")
        return

    output_file = Path(output_file)

    # Sort objects by ID for consistent ordering
    sorted_objects = sorted(image_counts.keys())
    selected_counts = [image_counts[obj_id] for obj_id in sorted_objects]

    # Calculate percentages if total_counts provided
    if total_counts:
        totals = [total_counts.get(obj_id, selected_counts[i]) for i, obj_id in enumerate(sorted_objects)]
        percentages = [100 * sel / tot if tot > 0 else 0 for sel, tot in zip(selected_counts, totals)]
    else:
        totals = selected_counts
        percentages = [100] * len(selected_counts)

    # Create the figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot bars on primary axis
    bars = ax1.bar(range(len(sorted_objects)), selected_counts, alpha=0.7,
                   color='skyblue', edgecolor='navy', label='Selected templates')

    ax1.set_xlabel('Object ID')
    ax1.set_ylabel('Number of Templates', color='navy')
    ax1.tick_params(axis='y', labelcolor='navy')
    ax1.grid(axis='y', alpha=0.3)

    # Create secondary y-axis for percentages
    ax2 = ax1.twinx()
    ax2.set_ylabel('Percentage (%)', color='red')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelcolor='red')

    # Plot percentages as crosses
    ax2.plot(range(len(sorted_objects)), percentages, 'rx',
             markersize=8, markeredgewidth=2, label='Percentage')

    # Set title
    ax1.set_title(f'Templates per object: {dataset}-{split}-{experiment_name}')

    # Set x-axis labels with object IDs
    if len(sorted_objects) <= 20:
        ax1.set_xticks(range(len(sorted_objects)))
        ax1.set_xticklabels(sorted_objects, rotation=45, ha='right')
    else:
        # For many objects, show every nth label
        step = max(1, len(sorted_objects) // 20)
        indices = list(range(0, len(sorted_objects), step))
        ax1.set_xticks(indices)
        ax1.set_xticklabels([sorted_objects[i] for i in indices], rotation=45, ha='right')

    # Add <selected>/<total> labels above bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        label = f'{selected_counts[i]}/{totals[i]}'
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 label,
                 ha='center', va='bottom', fontsize=8)

    # Adjust layout and save
    fig.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Histogram saved as: {output_file}")

    # Print summary statistics
    total_selected = sum(selected_counts)
    total_all = sum(totals)
    avg_selected = total_selected / len(selected_counts) if selected_counts else 0
    avg_percentage = 100 * total_selected / total_all if total_all > 0 else 0

    print(f"\nStatistics:")
    print(f"  Total objects: {len(sorted_objects)}")
    print(f"  Selected images: {total_selected}")
    print(f"  Total images: {total_all}")
    print(f"  Overall percentage: {avg_percentage:.1f}%")
    print(f"  Average selected per object: {avg_selected:.1f}")


def main():
    """
    Main function - processes multiple experiments and datasets
    """
    methods = [
        'hart',
        'hart_symmetric',
        'hart_imblearn',
        'hart_imblearn_adapted'
    ]
    descriptors = ['dinov2', 'dinov3']
    neighbors = ['1nn']

    original_bop_path = Path('/mnt/personal/jelint19/data/bop')

    for neighbor, method, descriptor in product(neighbors, methods, descriptors):
        experiment = f'{neighbor}-{method}-{descriptor}'
        dataset_sequences = [
            ('hot3d', 'object_ref_aria_static_scenewise'),
            ('hot3d', 'object_ref_quest3_static_scenewise'),
            # ('hot3d', 'object_ref_aria_dynamic_scenewise'),
            # ('hot3d', 'object_ref_quest3_dynamic_scenewise'),
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
            SOURCE_DIRECTORY = Path(
                '/mnt/personal/jelint19/cache/detections_templates_cache/') / experiment / relative_path
            TARGET_DIRECTORY = Path('/mnt/personal/jelint19/results/condensation') / experiment / relative_path
            HISTOGRAM_FILE = TARGET_DIRECTORY.parent.parent / Path(f'histogram_{dataset}-{split}-{experiment}.png')

            print("Image Directory Flattening Script with Histogram")
            print("=" * 50)
            print(f"Source directory: {SOURCE_DIRECTORY.resolve()}")
            print(f"Target directory: {TARGET_DIRECTORY.resolve()}")
            print(f"Histogram file: {HISTOGRAM_FILE.resolve()}")
            print()

            # Copy images and get selected counts
            image_counts = copy_images_to_flat_structure(SOURCE_DIRECTORY, TARGET_DIRECTORY, dry_run=True)

            # Get total counts from original dataset
            total_counts = get_total_counts(original_bop_path, dataset, split)

            # Create histogram with percentages
            create_histogram(image_counts, dataset, split, HISTOGRAM_FILE, experiment, total_counts)


if __name__ == "__main__":
    main()