from pathlib import Path


def generate_configs():
    minmatches = [50, 100, 200, 400]
    maxmatches = [200, 400, 800]
    config_base_folder = Path('configs')
    config_folder = Path('glotracker/sift_thresholds')

    config_names = []
    config_paths = []

    for minmatch in minmatches:
        for maxmatch in maxmatches:
                config_name = f'glotracker_sift_min{minmatch}_good{maxmatch}'
                config_path = config_folder / f'{config_name}.py'
                config_paths.append(config_folder / config_name)

                with open(config_base_folder / config_path, 'w') as f:
                    f.write(f"""from glopose_config import GloPoseConfig


def get_config() -> GloPoseConfig:
    cfg = GloPoseConfig()

    cfg.onboarding.frame_filter = 'SIFT'

    cfg.onboarding.sift_filter_min_matches = 100
    cfg.onboarding.sift_filter_good_to_add_matches = 450

    return cfg
""")

                config_names.append(f'glotracker/{config_name}')

    return config_names, config_paths


if __name__ == "__main__":
    config_names, config_paths = generate_configs()
    print("Generated config names:")
    for name in config_names:
        print(f"    '{name}',")
