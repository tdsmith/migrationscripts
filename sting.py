#!/usr/bin/env python
# encoding=utf-8

# named sting because: if you're a cell,
# every breath you take, every step you make,
# we'll be watching you
# © tim smith 2014, tim.smith@uci.edu
# released under the terms of the wtfpl http://wtfpl.net

import pandas as pd
from StringIO import StringIO
import ggplot as gg
import codecs

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
    print df
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

def displacement_plot(centered):
    centered['Object'] = centered['Object'].map(str)
    centered = centered.sort(['Frame', 'Object'])
    g = gg.ggplot(centered, gg.aes(x='cX', y='cY', color='Object')) + gg.geom_path() + gg.theme_bw()
    return g

def save_plot(tsv_in, png_out):
    # there has to be a better pattern for this
    try:
        df = read_mtrackj_mdf(tsv_in)
    except ValueError as e:
        try:
            df = read_mtrack2(tsv_in)
        except Exception:
            df = read_manual_track(tsv_in)
    centered = center(df)
    centered.to_csv(tsv_in + '.centered')
    g = displacement_plot(centered)
    gg.ggsave(g, png_out)

if __name__ == '__main__':
    import sys
    save_plot(sys.argv[1], sys.argv[1] + '.png')

