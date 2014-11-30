#!/usr/bin/env python
# coding=utf-8

"""Splats a xypt OME-TIFF stack into p substacks.
Reduces bit depth from 12 bits to 8 bits."""

from __future__ import print_function
import os.path
import tifffile as tf
import numpy as np
from distutils.dir_util import mkpath

def extrude(filename, outpath, n_positions, handler=print):
    tiff = tf.TiffFile(filename)
    assert len(tiff) % n_positions == 0
    # n_timepoints = len(tiff) / n_positions
    mkpath(outpath)
    for p in xrange(n_positions):
        handler(p)
        outfn = os.path.join(outpath, "{}.tif".format(p))
        with tf.TiffWriter(outfn) as tw:
            array = tf.stack_pages(tiff[p::n_positions])
            array >>= 4
            array = array.astype(np.uint8)
            tw.save(array)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_stack")
    parser.add_argument("output_path")
    parser.add_argument("n_positions", type=int)
    args = parser.parse_args()
    extrude(args.input_stack, args.output_path, args.n_positions)
