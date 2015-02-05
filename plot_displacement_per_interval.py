#!/usr/bin/env python

from __future__ import division, print_function
import argparse

from lmfit import Model
from matplotlib import pyplot as plt
from matplotlib.figure import SubplotParams
from numpy import sqrt, mean, exp, arange, array
import pandas as pd

from sting import read_mtrackj_mdf, center

def calculate_ms_dx(cell, pixels_per_micron=1.51):
    # verify frames are contiguous
    assert len(cell['Object'].unique()) == 1
    max_frame = cell['Frame'].max()
    assert all(cell['Frame'] == range(1, max_frame+1))

    x = cell['cX'].values
    y = cell['cY'].values
    distance = lambda i, j: sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
    intervals = []
    for interval in xrange(1, max_frame):
        distances = []
        for start in xrange(max_frame-interval):
            distances.append(distance(start, start+interval))
        intervals.append(distances)
    ms = lambda x: mean([(i / pixels_per_micron)**2 for i in x])
    ms_dx = [ms(i) for i in intervals]
    return array(ms_dx)

def fit_model(ms_dx, hours_per_frame=5/60):
    fitfunc = lambda T, s, p: (s**2 * p**2) * (T/p - 1 + exp(-T/p))
    n = len(ms_dx)//2
    model = Model(fitfunc)
    model.set_param_hint('s', value=2, min=0)
    model.set_param_hint('p', value=0.5, min=0)
    T = (arange(len(ms_dx)) + 1) * hours_per_frame
    result = model.fit(ms_dx[:n], T=T[:n])
    return result, (T, result.best_fit)

def open_mdf(mdf_file):
    data = read_mtrackj_mdf(mdf_file)
    return center(data)

def main():
    parser = argparse.ArgumentParser("Compute and display RMS displacement as a function of time interval.")
    parser.add_argument('mdf_file')
    args = parser.parse_args()

    data = open_mdf(args.mdf_file)
    g = data.groupby("Object")
    subplotpars = SubplotParams(left=0.05, bottom=0.10,
                                right=0.95, top=0.90,
                                wspace=0.4, hspace=0.5)
    fig, ax = plt.subplots(len(g)//4 + 1, 4, sharex=True, sharey=True,
                           subplotpars=subplotpars)
    for i, (name, group) in enumerate(g):
        ms_dx = calculate_ms_dx(group)
        result, (T, best_fit) = fit_model(ms_dx)
        ax.flat[i].plot(T, ms_dx, '.', T[:len(best_fit)], best_fit, linewidth=5)
        s, p = result.best_values['s'], result.best_values['p']
        ax.flat[i].set_title(name)
        ax.flat[i].text(
            0.05, 0.95, '({:.2f}, {:.2f})'.format(s, p),
            horizontalalignment='left', verticalalignment='top',
            transform=ax.flat[i].transAxes)
    plt.show()

if __name__ == '__main__':
    main()
