"""
A Defect Prediction model for file level metrics
"""
import os
import sys
import pandas as pd
from pdb import set_trace

from pathlib import Path
root = Path(os.path.abspath(os.path.join(
    os.getcwd().split("se4sci")[0], 'se4sci/se4sci')))

if root not in sys.path:
    sys.path.append(str(root))

from data.data_handler import DataHandler

if __name__ == "__main__":
    set_trace()