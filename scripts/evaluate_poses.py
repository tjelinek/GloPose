import csv

import numpy as np


def evaluate_csv(csv_path, max_rot_error=5.0, max_trans_error=0.05):
    """
    Evaluate the pose reconstruction from a CSV file.

    Parameters:
        csv_path: Path to the CSV file with reconstruction statistics.
        max_rot_error: Maximum allowed rotation error (degrees) for accuracy calculation.
        max_trans_error: Maximum allowed translation error (meters) for accuracy calculation.

    Returns:
        Dictionary with calculated statistics.
    """
    data = []

    # Read the CSV file
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append({
                'dataset': row['dataset'],
                'sequence': row['sequence'],
                'image_frame_id': int(row['image_frame_id']),
                'rot_error': float(row['rot_error']),
                'trans_error': float(row['trans_error'])
            })

    # Calculate statistics
    rot_errors = [row['rot_error'] for row in data]
    trans_errors = [row['trans_error'] for row in data]

    mean_rot_error = np.mean(rot_errors)
    mean_trans_error = np.mean(trans_errors)

    accuracy_rot = np.mean([err <= max_rot_error for err in rot_errors])
    accuracy_trans = np.mean([err <= max_trans_error for err in trans_errors])

    return {
        'mean_rot_error': mean_rot_error,
        'mean_trans_error': mean_trans_error,
        'accuracy_rot': accuracy_rot,
        'accuracy_trans': accuracy_trans
    }