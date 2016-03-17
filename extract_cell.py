from __future__ import division

import os

from PIL import Image
from numba import jit
import numpy as np
import tifffile as tf


@jit
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
    assert len(stack.shape) == 3
    assert len(xlist) == len(ylist)
    n = min(stack.shape[0], len(xlist))
    out = np.zeros((n, h, w), dtype=stack.dtype)
    for i, (im, x, y) in enumerate(zip(stack, xlist, ylist)):
        out[i] = extract_window(im, x, y, h, w)
    return out


def movie_of_cell(mdf_frame, filenames, h, w):
    assert len(mdf_frame.Object.unique()) == 1
    if isinstance(filenames, basestring):
        # it's a single file; open it with tifffile
        images = np.array(tf.imread(filenames), ndmin=3)
    else:
        # we have a list of filenames; open them with Pillow because they're
        # probably jpegs.
        # assert the filenames have the same extension
        assert len(set([os.path.splitext(fn)[1] for fn in filenames])) == 1
        images = np.array([np.asarray(Image.open(fn)) for fn in filenames])
    return stack_extract_window(images,
                                mdf_frame.X, mdf_frame.Y,
                                h, w)
