from __future__ import division

import numpy as np
import pandas as pd
import tifffile as tf


def extract_window(image, x, y, h, w):
    hw, hh = w//2, h//2
    sample = image[max(0, y-hh):y+hh, max(0, x-hw):x+hw]
    top_pad = abs(y-hh) if y-hh < 0 else 0
    left_pad = abs(x-hw) if x-hw < 0 else 0
    sh, sw = sample.shape
    output = np.zeros((h, w), dtype=sample.dtype)
    output[top_pad:sh+top_pad, left_pad:sw+left_pad] = sample
    return output


def stack_extract_window(stack, xlist, ylist, h, w):
    n = stack.shape[0]
    out = np.zeros((n, h, w), dtype=stack.dtype)
    for i, (im, x, y) in enumerate(zip(stack, xlist, ylist)):
        out[i] = extract_window(im, x, y, h, w)
    return out
