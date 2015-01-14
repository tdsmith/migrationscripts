#!/usr/bin/env python

from __future__ import division
import ggplot as gg
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.figure import SubplotParams
from numpy import sqrt, cumsum, array
from sting import read_mtrackj_mdf, center
from os.path import basename, splitext

def segment_lengths(obj):
    obj['SegmentLength'] = obj['cX']
    # use array() to prevent index alignment
    obj.ix[1:, 'SegmentLength'] = sqrt((obj['X'][1:] - array(obj['X'][:-1]))**2 +
                                       (obj['Y'][1:] - array(obj['Y'][:-1]))**2)
    obj['PartialPathLength'] = cumsum(obj['SegmentLength'])
    return obj

def fit_random_walk_to_tracks(df, n_points):
    df['Distance'] = sqrt(df['cX']**2 + df['cY']**2)
    distances = df.groupby('Frame')['Distance']
    mean_square = lambda x: sum(x**2)/len(x)
    mean_square_distance = pd.DataFrame({
            'mean_square_distance': distances.aggregate(mean_square),
            'n': distances.aggregate(len)
            })
    mean_square_distance.reset_index()
    t = 1
    fitfunc = lambda x, s, p: s**2 * p**2 * (x/p - 1 + np.exp(-x/p))
    x = (np.arange(n_points) + 1) * t
    y = mean_square_distance.ix[:n_points, 'mean_square_distance']
    popt, pcov = sp.optimize.curve_fit(fitfunc, x, y, [0.6, 40], maxfev=2000, ftol=1e-6, xtol=1e-6)
    fit = pd.DataFrame({'t': x, 'y': y, 'fit': fitfunc(x, *popt)})
    return popt, fit

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

def plot_ensemble(mdfs):
    fits = {}
    mdfs.sort(key=lambda x: int(splitext(basename(x))[0][:2]))
    for mdf in mdfs:
        data = open_mdf(mdf)
        (s, p), fit = fit_random_walk_to_tracks(data, 100)
        fits[mdf] = ((s,p), fit)
    subplotpars = SubplotParams(left=0.05, bottom=0.05,
                                right=0.95, top=0.95,
                                wspace=0.4, hspace=0.7)
    fig, ax = plt.subplots(5,6, subplotpars=subplotpars)
    for i, mdf in enumerate(mdfs):
        s, p = fits[mdf][0]
        fit = fits[mdf][1]
        ax.flat[i].plot(fit['t'], fit['y'], '.',
                        fit['t'], fit['fit'], '-')
        ax.flat[i].text(0.05, 0.95, '({:.2f}, {:.2f})'.format(s, p),
            horizontalalignment='left', verticalalignment='top',
            transform=ax.flat[i].transAxes)
        ax.flat[i].set_title(basename(mdf))
    plt.show()
        
def main():
    import sys, argparse
    parser = argparse.ArgumentParser(description='Make some plots.')
    parser.add_argument('--individual', '-i', action='store_true')
    parser.add_argument('mdf_file', nargs='+')
    args = parser.parse_args()
    if args.individual:
        for infile in parser.mdf_file:
            data = open_mdf(infile)
            individual_plots(data)
    else:
        plot_ensemble(args.mdf_file)

if __name__ == '__main__':
    main()
