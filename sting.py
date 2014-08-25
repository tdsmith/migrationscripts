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
    return df

def displacement_plot(df):
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
    # df.to_csv('intermediate.csv', index_label=False)
    def center_transform(x):
        mydf = x.copy()
        mydf['cX'] = x['X'] - x['X'].iloc[0]
        mydf['cY'] = x['Y'] - x['Y'].iloc[0]
        return mydf
    centered = df.groupby('Object').apply(center_transform)
    centered['cY'] = -centered['cY']
    centered['Object'] = centered['Object'].map(str)

    g = gg.ggplot(centered, gg.aes(x='cX', y='cY', color='Object')) + gg.geom_line()

    return g

def save_plot(tsv_in, png_out):
    df = read_mtrack2(tsv_in)
    g = displacement_plot(df)
    gg.ggsave(g, png_out)

if __name__ == '__main__':
    import sys
    save_plot(sys.argv[1], sys.argv[1] + '.png')

