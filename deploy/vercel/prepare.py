"""Copy the ONNX model and the config subset the function needs from the repo.

Run from anywhere:  python deploy/vercel/prepare.py
Exports the ONNX model first if models/local/yolov8n_v4.onnx is missing.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
here = Path(__file__).resolve().parent
config = json.loads((root / 'visionaid_config.json').read_text())
# Vercel serves the full-precision export; local.file is now the INT8 model, so use fp32_file (the .pt).
onnx = root / config['local'].get('fp32_file', config['local']['file']).replace('.pt', '.onnx')
if not onnx.is_file():
    subprocess.run([sys.executable, str(root / 'scripts' / 'export_detector.py')], check=True)
(here / 'model').mkdir(parents=True, exist_ok=True)
shutil.copy2(onnx, here / 'model' / onnx.name)
subset = {'classes': config['classes'], 'class_thresholds': config['local']['class_thresholds'],
          'input_size': config['local']['input_size'], 'nms_iou': config['local']['nms_iou'],
          'alert_priority': config['alert_priority'], 'model_file': onnx.name}
(here / 'config.json').write_text(json.dumps(subset, indent=2) + '\n')
print(f'Ready: {onnx.name} ({onnx.stat().st_size / 1e6:.1f} MB) + config.json')
