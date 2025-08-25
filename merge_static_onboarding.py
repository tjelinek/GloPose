from pathlib import Path
from typing import List


def merge_reconstructions_for_experiment(experiment_name: str, datasets: List[str] = None):
    base_folder = Path('/mnt/personal/jelint19/cache/view_graph_cache') / experiment_name

    if datasets is None:
        datasets = ['hope', 'handal']

    for dataset in datasets:
        dataset_dir = base_folder / dataset

        all_sequences = list(dataset_dir.iterdir())

        down_sequences = sorted(sequence for sequence in all_sequences if '_down' in sequence.name)
        up_sequences = sorted(sequence for sequence in all_sequences if '_up' in sequence.name)

        down_sequences_names = [seq.name.replace('_down', '') for seq in down_sequences]
        up_sequences_names = [seq.name.replace('_up', '') for seq in up_sequences]

        pairs = []
        for i, x in enumerate(down_sequences_names):
            for j, y in enumerate(up_sequences_names):
                if x == y:
                    pairs.append((i, j))




if __name__ == '__main__':
    experiment_name = 'ufm_c095r05_viewgraph_from_matching'

    merge_reconstructions_for_experiment(experiment_name)
