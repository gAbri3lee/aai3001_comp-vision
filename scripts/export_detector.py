"""Export the verified local checkpoint to FP32 ONNX; adoption requires parity/AP checks."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import load_config, resolve_path
from detection import validate_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=None)
    args = parser.parse_args()
    config = load_config()
    path = resolve_path(args.model or config['local']['file'])
    if not path.is_file():
        parser.error(f'Missing weights: {path}')
    from ultralytics import YOLO
    model = YOLO(str(path))
    validate_names(model.names, config['classes'])
    print(model.export(format='onnx', imgsz=640, batch=1, dynamic=False, half=False, simplify=False, opset=17))


if __name__ == '__main__':
    main()
