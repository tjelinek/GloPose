from pathlib import Path


def generate_configs():
    thresholds = [0.75, 0.95, 0.5]
    certainties = [0.95, 0.9, 0.5]
    matches = [200, 500, 1000, 2500]

    config_base_folder = Path('configs')
    config_folder = Path('glotracker/roma_thresholds')

    config_names = []
    config_paths = []

    for certainty in certainties:
        for threshold in thresholds:
            for match in matches:
                config_name = f'glotracker_roma_c{int(certainty * 100):02d}_fg_{int(threshold * 100):02d}_m{match}'
                config_path = config_folder / f'{config_name}.py'
                config_paths.append(config_folder / config_name)

                with open(config_base_folder / config_path, 'w') as f:
                    f.write(f"""from glopose_config import GloPoseConfig


def get_config() -> GloPoseConfig:
    cfg = GloPoseConfig()

    cfg.onboarding.frame_filter = 'RoMa'

    cfg.onboarding.min_certainty_threshold = {certainty}
    cfg.onboarding.flow_reliability_threshold = {threshold}
    cfg.onboarding.min_number_of_reliable_matches = {match}

    return cfg
""")

                config_names.append(f'glotracker/{config_name}')

    return config_names, config_paths


if __name__ == "__main__":
    config_names, config_paths = generate_configs()
    print("Generated config names:")
    for name in config_names:
        print(f"    '{name}',")

    for pth in config_paths:
        print(str(pth), end='\n')
