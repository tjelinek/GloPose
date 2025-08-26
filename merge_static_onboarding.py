from pathlib import Path
from typing import List

from data_structures.view_graph import merge_two_view_graphs


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

        for pair_down_i, pair_up_j in pairs:

            down_sequence_folder = down_sequences[pair_down_i]
            up_sequences_folder = up_sequences[pair_up_j]

            output_path = down_sequence_folder.parent / f'{down_sequences_names[pair_down_i]}_merged'

            merge_two_view_graphs(down_sequence_folder, up_sequences_folder, output_path)


if __name__ == '__main__':
    experiment_name = 'ufm_c095r05_viewgraph_from_matching'

    merge_reconstructions_for_experiment(experiment_name)
