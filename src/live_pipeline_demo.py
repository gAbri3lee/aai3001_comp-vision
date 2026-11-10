"""OpenCV view of the same trained detector/depth/audio pipeline as the web UI."""
import argparse
from config import add_detector_arguments
from webapp import PipelineState
from window_view import run_window


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=0)
    add_detector_arguments(parser)
    for feature in ['depth', 'tts', 'ocr', 'voice']:
        parser.add_argument('--no-' + feature, action='store_true')
    parser.add_argument('--mic-index', type=int, default=None)
    parser.set_defaults(no_product_id=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run_window(PipelineState(args), 'VisionAid - Full Pipeline', ocr_enabled=not args.no_ocr)


if __name__ == '__main__':
    main()
