"""OpenCV view of the same trained detector/depth/audio pipeline as the web UI."""
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
    for flag in ('--no-ocr', '--no-voice', '--no-product-id'):
        parser.add_argument(flag, action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--mic-index', type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    run_window(PipelineState(args), 'VisionAid - Full Pipeline')


if __name__ == '__main__':
    main()
