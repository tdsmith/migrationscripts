#!/usr/bin/env python
# coding=utf-8

"""Splats a xypt OME-TIFF stack into p substacks.
Reduces bit depth from 12 bits to 8 bits."""

from __future__ import print_function
import os
import tifffile as tf
import numpy as np

def extrude(filename, outpath, handler=print):
    tiff = tf.TiffFile(filename, multifile=True, multifile_close=False)
    serieses = tiff.series
    os.makedirs(outpath)
    for p, series in enumerate(serieses):
        handler(p)
        outfn = os.path.join(outpath, "{}.tif".format(p))
        with tf.TiffWriter(outfn) as tw:
            array = tf.stack_pages(series['pages'])
            array >>= 4
            array = array.astype(np.uint8)
            tw.save(array)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_stack")
    parser.add_argument("output_path")
    args = parser.parse_args()
    extrude(args.input_stack, args.output_path)
