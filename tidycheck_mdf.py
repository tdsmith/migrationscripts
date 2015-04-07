#!/usr/bin/env python
"""Given a MTrackJ .mdf file, keep tracks with the right number of points."""

import codecs
import argparse
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mdf_filename', help='File to process')
    args = parser.parse_args()

    f = codecs.open(args.mdf_filename, 'r', 'mac-roman')
    line = f.readline()
    if not line.startswith("MTrackJ"):
        raise ValueError("File {} is not in MTrackJ format.".format(args.mdf_filename))

    n_points_by_track = Counter()
    tracks_by_point = {}

    track = None
    for line in f:
        if line.startswith("Track"):
            track = int(line.split()[1])
        elif line.startswith("Point"):
            split = line.split()
            frameno = int(float(split[-2]))
            tracks_by_point.setdefault(frameno, []).append(track)
            n_points_by_track[track] += 1

    mode_n_points = Counter(n_points_by_track.values()).most_common(1)[0][0]
    # these are the right length
    goldilocks_tracks = [track for track, n_points in n_points_by_track.most_common() if n_points == mode_n_points]
    goldilocks_tracks = [track for track in goldilocks_tracks if track in tracks_by_point[1]]

    f.seek(0)
    track = None
    for line in f:
        line = line.strip()
        if line.startswith("Track"):
            track = int(line.split()[1])
            if track in goldilocks_tracks:
                print line
        elif line.startswith("Point"):
            if track in goldilocks_tracks:
                print line
        else:
            print line

if __name__ == '__main__':
    main()
