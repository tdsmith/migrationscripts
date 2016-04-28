"""Given a MTrackJ .mdf file sampled every N frames and a reduction factor R, yield the .mdf file
that would have been produced if the .mdf had originally been sampled every N/R frames."""

import codecs
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('r', type=int, help='Factor by which to reduce the number of samples')
    parser.add_argument('mdf_filename', help='File to process')
    args = parser.parse_args()

    f = codecs.open(args.mdf_filename, 'r', 'mac-roman')
    line = f.readline()
    if not line.startswith("MTrackJ"):
        raise ValueError("File {} is not in MTrackJ format.".format(args.mdf_filename))
    # learn the maximum number of sampled points
    N = 0
    for line in f:
        if line.startswith("Point"):
            split = line.split()
            this_n = int(split[1])
            if this_n > N:
                N = this_n
    keep = range(1, N, args.r) + [N]
    f.seek(0)
    for line in f:
        if line.startswith("Point"):
            split = line.split()
            if int(float(split[-2])) in keep:
                print line,
        else:
            print line,


if __name__ == '__main__':
    main()
