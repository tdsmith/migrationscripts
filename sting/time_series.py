#!/usr/bin/env python

from __future__ import division
import ggplot as gg
import pandas as pd
from numpy import sqrt, cumsum, array
from sting import read_mtrackj_mdf, center


def segment_lengths(obj):
    obj = obj.copy()
    obj['SegmentLength'] = obj['cX']
    # use array() to prevent index alignment
    obj.ix[1:, 'SegmentLength'] = sqrt((obj['X'][1:] - array(obj['X'][:-1]))**2 +
                                       (obj['Y'][1:] - array(obj['Y'][:-1]))**2)
    obj['PartialPathLength'] = cumsum(obj['SegmentLength'])
    return obj


def individual_plots(data):
    rms = lambda x: sqrt(sum(x**2))/len(x)

    data['Distance'] = sqrt(data['cX']**2 + data['cY']**2)
    data = data.groupby('Object').apply(segment_lengths)

    g = gg.ggplot(data, gg.aes('Frame', 'PartialPathLength', color='Object')) + gg.geom_line()
    print(g)

    g = gg.ggplot(data, gg.aes('Frame', 'Distance', color='Object')) + gg.geom_line()
    print(g)

    dx = data.groupby('Frame')['Distance']
    rms_dx = pd.DataFrame({'RMS_Distance': dx.aggregate(rms), 'n': dx.aggregate(len)})
    rms_dx = rms_dx.reset_index()
    g = (gg.ggplot(rms_dx, gg.aes(x='Frame', y='RMS_Distance', size='n')) +
         gg.geom_line(size=1) +
         gg.geom_point(alpha=0.5) +
         gg.theme_bw())
    print(g)


def open_mdf(mdf_file):
    data = read_mtrackj_mdf(mdf_file)
    return center(data)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Make some plots.')
    # parser.add_argument('--individual', '-i', action='store_true')
    parser.add_argument('mdf_file', nargs='1')
    args = parser.parse_args()
    if args.individual:
        for infile in parser.mdf_file:
            data = open_mdf(infile)
            individual_plots(data)
    else:
        pass
        # plot_ensemble(args.mdf_file)


if __name__ == '__main__':
    main()
