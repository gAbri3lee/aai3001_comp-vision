"""Detection-only demo using the trained five-class local or cloud detector."""
import argparse
from config import add_detector_arguments
from webapp import PipelineState
from window_view import run_window


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=0, help='Camera index or video path')
    add_detector_arguments(parser)
    parser.add_argument('--hazard-only', action='store_true', help='Compatibility flag: all five configured classes are hazards')
    parser.set_defaults(no_depth=True, no_tts=True, no_ocr=True, no_voice=True, no_product_id=True, mic_index=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_window(PipelineState(args), 'VisionAid - Hazard Detection')


if __name__ == '__main__':
    main()
