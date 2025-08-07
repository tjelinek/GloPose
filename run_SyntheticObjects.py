from pathlib import Path
from dataset_generators import scenarios
from utils.experiment_runners import run_on_synthetic_data
from utils.runtime_utils import parse_args
from utils.general import load_config


def main():
    dataset = 'SyntheticObjects'
    args = parse_args()

    # Load configuration first
    config = load_config(args.config)

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [
            'Textured_Sphere_5_y',
            'Textured_Cube_5_y',
            'Textured_Sphere_5_z',
            'Translating_Textured_Sphere',
            'Textured_Sphere_10_xy',
            'Rotating_Translating_Textured_Sphere_5_y',
            'Rotating_Translating_Textured_Sphere_5_xy',
            'Rotating_Contra_Translating_Textured_Sphere_5_y',
            'Rotating_Contra_Translating_Textured_Sphere_5_xy',
            '8_Colored_Sphere_5_x',
            '6_Colored_Cube_5_z'][0:1]

    for sequence in sequences:

        # Get mesh and texture paths based on sequence
        if '8_Colored_Sphere' in sequence:
            gt_mesh_path = Path('prototypes/8-colored-sphere/8-colored-sphere.obj')
            gt_texture_path = Path('prototypes/8-colored-sphere/8-colored-sphere-tex.png')
        elif 'Textured_Sphere' in sequence:
            gt_mesh_path = Path('prototypes/sphere.obj')
            gt_texture_path = Path('prototypes/tex.png')
        elif 'Textured_Cube' in sequence:
            gt_mesh_path = Path('prototypes/textured-cube/textured-cube.obj')
            gt_texture_path = Path('prototypes/textured-cube/tex.png')
        else:
            gt_texture_path = None
            gt_mesh_path = None

        # Run tracking with y-axis rotations by default
        run_on_synthetic_data(
            config=config,
            dataset=dataset,
            sequence=sequence,
            experiment=args.experiment,
            output_folder=args.output_folder,
            gt_mesh_path=gt_mesh_path,
            gt_texture_path=gt_texture_path,
            rotation_generator=scenarios.generate_rotations_y
        )


if __name__ == "__main__":
    main()