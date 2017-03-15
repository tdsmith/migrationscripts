#!/usr/bin/env python
# encoding:utf8

from __future__ import division, print_function
import argparse
from textwrap import dedent
import sys

from lmfit import Model
from matplotlib import pyplot as plt
from matplotlib.figure import SubplotParams
from numpy import sqrt, mean, exp, arange, array
import pandas as pd
import scipy as sp
import warnings

from sting import read_mtrackj_mdf, center, displacement_plot
from time_series import segment_lengths


class SkippedCellWarning(UserWarning):
    pass


class NotEnoughTimepointsWarning(SkippedCellWarning):
    pass


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


def guess_speed(cell, pixels_per_micron, hours_per_frame):
    # calculate path-length
    cell = segment_lengths(cell)
    len_h = len(cell.index) * hours_per_frame
    return cell["PartialPathLength"].iloc[-1] / pixels_per_micron / len_h


def fit_model(ms_dx, guess_s, hours_per_frame=5/60):
    fitfunc = lambda T, s, p: (s**2 * p**2) * (T/p - 1 + exp(-T/p))
    n = len(ms_dx)//2
    model = Model(fitfunc)
    model.set_param_hint('s', value=guess_s, min=0, max=250)
    # Constrain persistence time to ≥ 1 minute
    model.set_param_hint('p', value=0.5, min=1/60.)
    T = (arange(len(ms_dx)) + 1) * hours_per_frame
    result = model.fit(ms_dx[:n], T=T[:n])
    return result, (T, result.best_fit)


def open_mdf(mdf_file):
    data = read_mtrackj_mdf(mdf_file)
    return center(data)


def main():
    parser = argparse.ArgumentParser('Compute and display RMS displacement '
                                     'as a function of time interval.')
    parser.add_argument('--no-plots', action='store_true',
                        help="Don't save plots")
    parser.add_argument('--imagetype', '-i', default='pdf',
                        help='Extension to use for plots')
    parser.add_argument('--minutes', '-m', default=5, type=int,
                        help='Minutes between frames')
    parser.add_argument('--pixels', '-p', default=1.51, type=float,
                        help='Pixels per micron')
    parser.add_argument('--report', '-r', action='store_true',
                        help='Write a report')
    parser.add_argument('mdf_file', nargs='+')
    args = parser.parse_args()
    fit_table = {'filename': [],
                 'Object': [],
                 's': [],
                 'p': [],
                 'r2': []}

    report_html = []

    for mdf_file in args.mdf_file:
        data = open_mdf(mdf_file)
        g = data.groupby("Object")
        subplotpars = SubplotParams(left=0.05, bottom=0.10,
                                    right=0.95, top=0.90,
                                    wspace=0.4, hspace=0.5)
        fig, ax = plt.subplots(len(g)//4 + 1, 4, sharex=True, sharey=True,
                               subplotpars=subplotpars)
        for i, (name, group) in enumerate(g):
            solo_fig, solo_ax = plt.subplots(1, 1)
            if len(group) < 5:
                warnings.warn(
                    "Skipping object {} in file {}: "
                    "not enough timepoints".format(name, mdf_file),
                    NotEnoughTimepointsWarning)
                continue
            try:
                ms_dx = calculate_ms_dx(
                    group,
                    pixels_per_micron=args.pixels)
                s_guess = guess_speed(
                    group,
                    pixels_per_micron=args.pixels,
                    hours_per_frame=args.minutes/60.)
            except Exception as e:
                print(e)
                warnings.warn(
                    "Skipping object {} in file {}: "
                    "alignment error?".format(name, mdf_file),
                    SkippedCellWarning)
                continue
            result, (T, best_fit) = fit_model(ms_dx, s_guess, args.minutes/60.)
            s, p = result.best_values['s'], result.best_values['p']
            r2, _ = sp.stats.pearsonr(ms_dx[:len(best_fit)], best_fit)

            axes = [ax.flat[i]]
            if args.report:
                axes.append(solo_ax)

            for axis in axes:
                axis.plot(T, ms_dx, '.',
                          T[:len(best_fit)], best_fit, linewidth=5)
                axis.set_title(name)
                axis.text(
                    0.05, 0.95, '({:.2f}, {:.2f})'.format(s, p),
                    horizontalalignment='left', verticalalignment='top',
                    transform=axis.transAxes)

            if args.report:
                solo_filename = "{}-{}-rw_fit.png".format(mdf_file, i)
                solo_fig.set_size_inches(3, 3)
                solo_fig.savefig(solo_filename)
                displacement_filename = "{}-{}-rw_displacement.png".format(mdf_file, i)
                g = displacement_plot(group, limits=300)
                g.save(displacement_filename, 3, 3)
                plt.close(g.fig)
                report_html.append(dedent(
                    """
                    <p>{filename}:{object}</p>
                    <p>s: <span style="{s_style}">{s:.3f}</span> (v: {velocity:.3f}),<br>
                       p<sub>min</sub>: <span style="{p_style}">{p:.1f}</span>,<br>
                       r<sup>2</sup>: <span style="{r_style}">{r2:.3f}</span></p>
                    <table><tr>
                    <td><img src="{displacement_filename}"></td>
                    <td><img src="{solo_filename}"><td>
                    </tr></table>
                    """.format(**{
                        "filename": mdf_file,
                        "object": name,
                        "s_style": "color: red" if s >= 249.9 else "",
                        "s": s,
                        "p_style": "color: red" if p < 0.018 else "",
                        "p": p * 60,
                        "r_style": "color: red" if r2 < 0.95 else "",
                        "r2": r2,
                        "velocity": s_guess,
                        "displacement_filename": displacement_filename,
                        "solo_filename": solo_filename,
                    })
                ))

            fit_table['filename'].append(mdf_file)
            fit_table['Object'].append(name)
            fit_table['s'].append(s)
            fit_table['p'].append(p)
            fit_table['r2'].append(r2)

            plt.close(solo_fig)

        fig.savefig("{}-rw_fit.{}".format(mdf_file, args.imagetype))
        plt.close(fig)

    if args.report:
        head = dedent(
            """
            <html>
            """)
        footer = dedent(
            """
            </html>
            """
        )
        body = "\n<hr>\n".join(report_html)
        with open("report.html", "w") as f:
            f.write(head)
            f.write(body)
            f.write(footer)

    df = pd.DataFrame(fit_table)
    sys.stdout.write(df.to_csv(index=False))


if __name__ == '__main__':
    main()
