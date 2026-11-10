"""Detection view without depth or speech, using the same detector and alert gate as the web UI."""
import argparse
from config import add_detector_arguments
from webapp import PipelineState
from window_view import run_window


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=0, help='Camera index or video path')
    add_detector_arguments(parser)
    parser.add_argument('--hazard-only', action='store_true', help='Compatibility flag: all five configured classes are hazards')
    parser.set_defaults(no_depth=True, no_tts=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run_window(PipelineState(args), 'VisionAid - Hazard Detection')


if __name__ == '__main__':
    main()
