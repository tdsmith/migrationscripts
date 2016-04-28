#!/usr/bin/env python
# coding=utf-8

"""Splats a xypt OME-TIFF stack into p substacks.
Reduces bit depth from 12 bits to 8 bits."""

from __future__ import print_function
import multiprocessing
import os
import tifffile as tf
import numpy as np


def process(argtuple):
    filename, outpath, p = argtuple
    print(p)
    tiff = tf.TiffFile(filename, multifile=True, multifile_close=False)
    serieses = tiff.series
    outfn = os.path.join(outpath, "{}.tif".format(p))
    with tf.TiffWriter(outfn) as tw:
        array = tf.tiffile.stack_pages(serieses[p]['pages'])
        array >>= 4
        array = array.astype(np.uint8)
        tw.save(array)


def extrude(filename, outpath, jobs=1):
    tiff = tf.TiffFile(filename, multifile=True, multifile_close=False)
    serieses = tiff.series
    os.makedirs(outpath)
    pool = multiprocessing.Pool(jobs)
    pool.map(process, zip([filename]*len(serieses),
                          [outpath]*len(serieses),
                          range(len(serieses))))
    pool.close()
    pool.join()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_stack")
    parser.add_argument("output_path")
    parser.add_argument("--jobs", "-j", help="Number of cores to use", default=1, type=int)
    args = parser.parse_args()
    extrude(args.input_stack, args.output_path, args.jobs)
