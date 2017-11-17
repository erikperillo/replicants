#!/usr/bin/env python3

from skimage import io
import numpy as np
import pandas as pd
import glob
import os
import sys

def main():
    #command-line input
    if len(sys.argv) < 2:
        print("usage: {} <preds_dir_path>".format(sys.argv[0]))
        return
    dir_path = sys.argv[1]

    #getting filepaths
    y_trues_fps = glob.glob(os.path.join(dir_path, "*_y-true.png"))
    y_preds_fps = [fp.replace("y-true.png", "y-pred.png") for fp in y_trues_fps]

