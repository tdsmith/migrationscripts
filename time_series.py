#!/usr/bin/env python

from __future__ import division
import sys
import ggplot as gg
import pandas as pd
import numpy as np
import scipy as sp
from numpy import sqrt, cumsum, array
from sting import read_mtrackj_mdf, center

data = read_mtrackj_mdf(sys.argv[1])
data = center(data)

def segment_lengths(obj):
    obj['SegmentLength'] = obj['cX']
    # use array() to prevent index alignment
    obj.ix[1:, 'SegmentLength'] = sqrt((obj['X'][1:] - array(obj['X'][:-1]))**2 +
                                       (obj['Y'][1:] - array(obj['Y'][:-1]))**2)
    obj['PartialPathLength'] = cumsum(obj['SegmentLength'])
    return obj

rms = lambda x: sqrt(sum(x**2))/len(x)
ms = lambda x: sum(x**2)/len(x)

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


# <D(t)^2> = S^2 P_v^2 ( T / P_v  - 1 + exp(-T/P_v) )
ms_dx = pd.DataFrame({'MS_Distance': dx.aggregate(ms), 'n': dx.aggregate(len)})
ms_dx = ms_dx.reset_index()
g = (gg.ggplot(ms_dx, gg.aes(x='Frame', y='MS_Distance', size='n')) +
     gg.geom_line(size=1) +
     gg.geom_point(alpha=0.5) +
     gg.theme_bw()) # +
     # gg.geom_line(data=fit, mapping=gg.aes(x='x', y='y'), color='red'))
print(g)

t = 1
fitfunc = lambda x, s, p: s**2 * p**2 * (x/p - 1 + np.exp(-x/p))
x = (np.arange(151) + 1) * t
y = ms_dx.ix[:150, 'MS_Distance']
popt, pcov = sp.optimize.curve_fit(fitfunc, x, y, [0.6, 40], maxfev=2000, ftol=1e-6, xtol=1e-6)
print popt
fit = pd.DataFrame({'x': x, 'y': fitfunc(x, *popt), 'n': 1})


g = (gg.ggplot(fit, gg.aes('x', 'y')) +
     gg.geom_line() +
     gg.geom_point(data=ms_dx, mapping=gg.aes('Frame', 'MS_Distance')))
print(g)
