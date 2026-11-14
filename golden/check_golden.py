"""Compare deployment outputs; returns nonzero on parity or image-coverage failure."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from golden import main

if __name__ == '__main__':
    main()
