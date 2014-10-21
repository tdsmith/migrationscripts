#!/usr/bin/env python
# encoding=utf-8

# named sting because: if you're a cell,
# every breath you take, every step you make,
# we'll be watching you
# © tim smith 2014, tim.smith@uci.edu
# released under the terms of the wtfpl http://wtfpl.net

from __future__ import division
import argparse
import codecs
import ggplot as gg
import pandas as pd
import sys
import time
from numpy import sqrt, sum, array
from StringIO import StringIO

def read_mtrack2(filename):
    with open(filename) as f:
        buf = f.readlines()
    if '\n' in buf:
    	# there might be an extra table at the end that we don't care about
    	buf = buf[:buf.index('\n')]
    buf = ''.join(buf)
    iobuf = StringIO(buf)
    # row 0 is headers and data starts on row 2 so skip row 1
    df = pd.read_csv(iobuf, sep='\t', skiprows=[1])
    # df.replace(to_replace=' ', value=float('NaN'), inplace=True)
    df = df.convert_objects(convert_numeric=True)
    # throw out flag columns
    df = df[df.columns[[not i.startswith('Flag') for i in df.columns]]]
    # fill NA's backwards and then forwards
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    # get X and Y columns
    melted = pd.melt(df, id_vars='Frame')
    melted['Axis'] = melted['variable'].apply(lambda x: x[0])
    melted['Object'] = melted['variable'].apply(lambda x: int(x[1:]))
    df = melted.groupby(['Object', 'Frame', 'Axis']).sum().unstack()['value']
    df = df.reset_index()
    return df

def read_manual_track(filename):
    df = pd.read_csv(filename, sep='\t', encoding='mac-roman')
    df.rename(columns={u'Track n°': 'Object',
                       u'Slice n°': 'Frame'},
              inplace=True)
    for col in ['Object', 'Frame']:
        df[col] = df[col].astype(int)
    return df

def read_mtrackj_mdf(filename):
    f = codecs.open(filename, 'r', 'mac-roman')
    if not f.readline().startswith("MTrackJ"):
        raise ValueError("File {} is not in MTrackJ MDF format.".format(filename))
    this_track = None
    x, y, obj, frame = [], [], [], []
    for line in f:
        if line.startswith('Track'):
            this_track = int(float(line.split()[1]))
        elif line.startswith('Point'):
            tmp = line.split()
            obj.append(this_track)
            x.append(float(tmp[2]))
            y.append(float(tmp[3]))
            frame.append(int(float(tmp[5])))
    return pd.DataFrame({'Object': obj, 'Frame': frame, 'X': x, 'Y': y})

def center(df):
    def center_transform(x):
        x['cX'] = x['X'] - x['X'].iloc[0]
        x['cY'] = x['Y'] - x['Y'].iloc[0]
        return x
    centered = df.groupby('Object').apply(center_transform)
    centered['cY'] = -centered['cY']
    return centered

def displacement_plot(centered, limits = None):
    centered['Object'] = centered['Object'].map(str)
    centered = centered.sort(['Frame', 'Object'])
    g = (gg.ggplot(centered, gg.aes(x='cX', y='cY', color='Object')) +
         gg.geom_path(size=0.3) +
         gg.theme_bw())
    if limits: g = g + gg.ylim(-limits, limits) + gg.xlim(-limits, limits)
    return g

def stats(df, length_scale=1, time_scale=1):
    def segment_lengths(obj):
        obj['SegmentLength'] = obj['cX']
        # use array() to prevent index alignment
        obj.ix[1:, 'SegmentLength'] = sqrt((obj['X'][1:] - array(obj['X'][:-1]))**2 +
                                           (obj['Y'][1:] - array(obj['Y'][:-1]))**2) / length_scale
        return obj
    rms = lambda x: sqrt(sum(x**2))
    df['Distance'] = sqrt(df['cX']**2 + df['cY']**2) / length_scale
    df = df.groupby('Object').apply(segment_lengths)
    per_object = df.groupby('Object')
    rms_dx = per_object['Distance'].aggregate(rms)
    max_dx = per_object['Distance'].max()
    path_length = per_object['SegmentLength'].sum()
    n_points = per_object['SegmentLength'].aggregate(len)
    last_frame = per_object['Frame'].max()
    velocity = path_length/(last_frame * time_scale / 60.0)
    return pd.DataFrame({'rms_displacement': rms_dx,
                         'max_displacement': max_dx,
                         'path_length': path_length,
                         'n_points': n_points,
                         'velocity': velocity})

def summary(df):
    return pd.DataFrame({'filename': [df['filename'].iloc[0]],
                         'median_rms_dx': [df['rms_displacement'].median()],
                         'mean_rms_dx': [df['rms_displacement'].mean()],
                         'n': [len(df)],
                         'mean_path_length': [df['path_length'].mean()],
                         'median_path_length': [df['path_length'].median()],
                         'mean_max_dx': [df['max_displacement'].mean()],
                         'median_max_dx': [df['max_displacement'].median()],
                         'mean_velocity': [df['velocity'].mean()],
                         'sd_velocity': [df['velocity'].std()]})

def main():
    parser = argparse.ArgumentParser(description="Draws displacement plots.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--limits', type=int, help="Maximum extent of the axes")
    parser.add_argument('--no-plots', action='store_true', help="Don't save plots")
    parser.add_argument('--summary', help='Save summary stats by file')
    parser.add_argument('--imagetype', '-i', default='png', help="Extension to use for plots")
    parser.add_argument('--pixels-per-micron', '--pixels', '-p', default=1.51, type=float,
                        help="Pixels per µm (length scale of tracked images)")
    parser.add_argument('--minutes-per-frame', '--minutes', '-m', default=10, type=float,
                        help="Minutes between each frame of the tracked images")
    parser.add_argument('--plot-titles', type=argparse.FileType('r'),
                        help="CSV file with filename and title columns")
    parser.add_argument('infile', nargs='+', help="File(s) to process.")
    args = parser.parse_args()

    plot_titles = pd.read_csv(args.plot_titles, index_col="filename") if args.plot_titles else None

    all_dfs = []
    for filename in args.infile:
        # there has to be a better pattern for this
        try:
            df = read_mtrackj_mdf(filename)
        except ValueError as e:
            try:
                df = read_mtrack2(filename)
            except Exception:
                df = read_manual_track(filename)
        centered = center(df)
        centered.to_csv(filename + '.centered')
        if not args.no_plots:
            g = displacement_plot(centered, limits = args.limits) + gg.theme(axis_text=gg.element_text(size=8))
            if plot_titles is not None and filename in plot_titles.index:
                g += gg.labs(title=plot_titles.ix[filename, 'title'])
            gg.ggsave(g, '{}.{}'.format(filename, args.imagetype), width=2.5, height=1.81, units='in')
        centered['filename'] = filename
        all_dfs.append(centered)
    mega_df = pd.concat(all_dfs, ignore_index=True)
    obj_stats = (mega_df.groupby('filename', sort=False)
                        .apply(lambda x: stats(x, length_scale=args.pixels_per_micron,
                                                  time_scale=args.minutes_per_frame))
                        .reset_index())
    summary_by_file = obj_stats.groupby('filename').apply(summary)
    if args.summary:
        summary_by_file.to_csv(args.summary, index=False)
    print("# Produced by {} at {}".format(' '.join(sys.argv), time.ctime()))
    print("# {} pixels per micron, {} minutes per frame".format(args.pixels_per_micron, args.minutes_per_frame))
    print("# distance units are microns; velocity units are microns/hour")
    obj_stats.to_csv(sys.stdout, index=False)

if __name__ == '__main__':
    main()
