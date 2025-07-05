import argparse
import traceback
from contextlib import contextmanager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default="configs/base_config.py")
    parser.add_argument("--sequences", required=False, nargs='*', default=None)
    parser.add_argument("--output_folder", required=False)
    parser.add_argument("--experiment", required=False, default='')  # Experiment name
    return parser.parse_args()


@contextmanager
def exception_logger(ignore_exceptions=(Exception,)):
    try:
        yield
    except ignore_exceptions as e:
        print(f"Exception caught: {type(e).__name__}: {e}")
        traceback.print_exc()
