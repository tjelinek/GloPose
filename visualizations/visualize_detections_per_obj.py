"""
Script to copy images from obj_<id>/rgb/<image> structure to all_images/obj_<id>_<image>
and create a histogram of image counts per object with percentage overlay
"""

import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


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
        rgb_folder = sequence / 'rgb' if 'quest3' not in str(path_to_split) else sequence / 'gray1'
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


def create_boxplot(all_counts_by_dataset, output_file, experiment_name):
    output_file = Path(output_file)

    labels = []
    data = []

    for (dataset, split), counts in sorted(all_counts_by_dataset.items()):
        labels.append(f'{dataset}\n{split}')
        data.append(list(counts.values()))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data, labels=labels)
    ax.set_ylabel('Number of Templates')
    ax.set_yscale('log', base=2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
    ax.set_title(f'Templates distribution: {experiment_name}')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Box plot saved as: {output_file}")


def create_linechart(all_stats, output_file, descriptor):
    output_file = Path(output_file)

    dataset_splits = sorted(set((ds, sp) for ds, sp, _ in all_stats.keys()))
    methods = sorted(set(m for _, _, m in all_stats.keys()))

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    method_colors = {method: colors[i] for i, method in enumerate(methods)}

    markers = {'min': 2, 'median': '_', 'max': 3}
    linestyles = {'min': ':', 'median': '--', 'max': '-'}

    fig, ax = plt.subplots(figsize=(14, 6))

    x_positions = range(len(dataset_splits))
    x_labels = [f'{ds}\n{sp}' for ds, sp in dataset_splits]

    for method in methods:
        mins = []
        medians = []
        maxs = []

        for dataset, split in dataset_splits:
            stats = all_stats.get((dataset, split, method))
            if stats:
                mins.append(stats['min'])
                medians.append(stats['median'])
                maxs.append(stats['max'])
            else:
                mins.append(None)
                medians.append(None)
                maxs.append(None)

        color = method_colors[method]
        ax.plot(x_positions, mins, color=color, linestyle=linestyles['min'], linewidth=1.5,
                label=f'{method} (min)')
        ax.plot(x_positions, medians, color=color, linestyle=linestyles['median'], linewidth=1.5,
                label=f'{method} (median)')
        ax.plot(x_positions, maxs, color=color, linestyle=linestyles['max'], linewidth=1.5,
                label=f'{method} (max)')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel('Number of Templates')
    ax.set_yscale('log', base=2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
    ax.set_title(f'Templates statistics per dataset/split ({descriptor})')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    fig.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Line chart saved as: {output_file}")


def main():
    """
    Main function - processes multiple experiments and datasets
    """
    methods = [
        'hart',
        'hart_symmetric',
        # 'hart_imblearn',
        # 'hart_imblearn_adapted'
    ]
    descriptors = [
        # 'dinov2'
        'dinov3',
    ]
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

    results_base = Path('/mnt/personal/jelint19/results/condensation')
    original_bop_path = Path('/mnt/data/vrg/public_datasets/bop')

    for descriptor in descriptors:
        all_stats = {}

        for method in methods:
            experiment = f'1nn-{method}-{descriptor}'

            all_counts_by_dataset = {}

            for dataset, split in dataset_sequences:
                relative_path = Path(dataset) / split
                SOURCE_DIRECTORY = Path(
                    '/mnt/personal/jelint19/cache/detections_templates_cache/') / experiment / relative_path
                TARGET_DIRECTORY = results_base / experiment / relative_path
                HISTOGRAM_FILE = TARGET_DIRECTORY.parent.parent / f'histogram_{dataset}-{split}-{experiment}.png'

                print("Image Directory Flattening Script with Histogram")
                print("=" * 50)
                print(f"Source directory: {SOURCE_DIRECTORY.resolve()}")
                print(f"Target directory: {TARGET_DIRECTORY.resolve()}")
                print(f"Histogram file: {HISTOGRAM_FILE.resolve()}")
                print()

                image_counts = copy_images_to_flat_structure(SOURCE_DIRECTORY, TARGET_DIRECTORY, dry_run=False)

                total_counts = get_total_counts(original_bop_path, dataset, split)

                create_histogram(image_counts, dataset, split, HISTOGRAM_FILE, experiment, total_counts)

                all_counts_by_dataset[(dataset, split)] = image_counts

                if image_counts:
                    counts_values = list(image_counts.values())
                    all_stats[(dataset, split, method)] = {
                        'min': np.min(counts_values),
                        'median': np.median(counts_values),
                        'max': np.max(counts_values)
                    }

            BOXPLOT_FILE = TARGET_DIRECTORY.parent.parent / f'box-plot_{experiment}.png'
            create_boxplot(all_counts_by_dataset, BOXPLOT_FILE, experiment)

        LINECHART_FILE = results_base / f'linechart_{descriptor}.png'
        create_linechart(all_stats, LINECHART_FILE, descriptor)


if __name__ == "__main__":
    main()
