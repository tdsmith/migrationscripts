"Extract Micromanager metadata JSON from an image."

from __future__ import print_function

import argparse
import json

import tifffile as tf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_image")
    parser.add_argument("--rename-images", action="store_true")
    args = parser.parse_args()
    tiff = tf.TiffFile(args.input_image, multifile=True, multifile_close=False)
    metadata = tiff.micromanager_metadata['summary']
    if args.rename_images:
        for i, pos in enumerate(metadata['InitialPositionList']):
            print("mv {}.tif {}.tif".format(i, pos["Label"]))
    else:
        print(json.dumps(metadata, sort_keys=True, indent=2))
