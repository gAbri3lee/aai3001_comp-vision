"""OpenCV view of the same detector, alert gate, depth and audio pipeline as the web UI."""
import argparse
from config import add_detector_arguments
from webapp import PipelineState
from window_view import run_window


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=0)
    add_detector_arguments(parser)
    for feature in ['depth', 'tts']:
        parser.add_argument('--no-' + feature, action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    run_window(PipelineState(args), 'VisionAid - Full Pipeline')


if __name__ == '__main__':
    main()
