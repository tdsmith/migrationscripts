#!/usr/bin/env python

from __future__ import division
import sys
import ggplot as gg
import pandas as pd
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

rms = lambda x: sqrt(sum(x**2))

data['Distance'] = sqrt(data['cX']**2 + data['cY']**2)
data = data.groupby('Object').apply(segment_lengths)

g = gg.ggplot(data, gg.aes('Frame', 'PartialPathLength', color='Object')) + gg.geom_line()
print(g)

g = gg.ggplot(data, gg.aes('Frame', 'Distance', color='Object')) + gg.geom_line()
print(g)

dx = data.groupby('Frame')['Distance']
rms_dx = pd.DataFrame({'RMS_Distance': dx.aggregate(rms), 'n': dx.aggregate(len)})
rms_dx = rms_dx.reset_index()
g = (gg.ggplot(rms_dx, gg.aes('Frame', 'RMS_Distance', size='n')) + gg.geom_line(size=1) +
     gg.geom_point(alpha=0.5) + gg.theme_bw())
print(g)
