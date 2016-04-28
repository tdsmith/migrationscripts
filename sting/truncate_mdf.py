import codecs
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, help='Keep points from the first n frames')
    parser.add_argument('mdf_filename', help='File to process')
    args = parser.parse_args()

    f = codecs.open(args.mdf_filename, 'r', 'mac-roman')
    line = f.readline()
    if not line.startswith("MTrackJ"):
        raise ValueError("File {} is not in MTrackJ format.".format(args.mdf_filename))
    f.seek(0)
    for line in f:
        if line.startswith("Point"):
            split = line.split()
            this_frame = int(float(split[-2])) 
            if this_frame <= args.n:
                print line,
        else:
            print line,


if __name__ == '__main__':
    main()
