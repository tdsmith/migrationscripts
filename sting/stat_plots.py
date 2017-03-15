# encoding=utf-8
from __future__ import division

import argparse
import os
from textwrap import dedent

import pandas as pd
import matplotlib.pyplot as plt

from sting import read_mtrackj_mdf, center, displacement_plot


def open_mdf(mdf_file):
    data = read_mtrackj_mdf(mdf_file)
    return center(data)


def main():
    parser = argparse.ArgumentParser(
        description="Generates report files of statistics",
    )
    parser.add_argument("sting_output")
    args = parser.parse_args()

    df = pd.read_csv(args.sting_output, comment="#")

    try:
        os.makedirs("statplots")
    except OSError:  # already exists
        pass

    header = dedent("""\
        <!DOCTYPE html>
        <html>
        <head>
        <title>sting statplots</title>
        <style type="text/css">
            td { padding: 10px; vertical-align: middle; }
        </style>
        <body>
        """)

    footer = dedent("""\
        </body>
        </html>
        """)

    plots = []

    for (filename, subset) in df.groupby("filename"):
        plots.append(dedent("""\
            <h1>{filename}</h1>
            <table>
            <tr>
            <td>
                <img src="../{filename}.png">
            </td>
            <td>
                <p>Mean velocity: {velocity:.3f} µm/h</p>
                <p>Mean max displacement: {max_displacement:.3f} µm</p>
            </td>
            </tr>
            </table>
            <hr>
            """).format(
            filename=filename,
            velocity=subset["velocity"].mean(),
            max_displacement=subset["max_displacement"].mean(),
        ))
        for row in subset.itertuples(index=False):
            drow = dict(zip(df.columns, row))
            obj_id = drow["Object"]
            track = open_mdf(filename).query("Object == %d" % obj_id)
            plot_filename = "{}-{}-displacement.png".format(filename, obj_id)
            g = displacement_plot(track, limits=300)
            g.save("statplots/" + plot_filename, 3, 3)
            plt.close(g.fig)

            plots.append(dedent("""\
                <p>{mdf_file}, {object_id}</p>
                <table>
                <tr>
                <td><img src="{plot_filename}"></td>
                <td>
                    <p>Velocity: {velocity:.3f} µm/h</p>
                    <p>Max displacement: {max_displacement:.3f} µm</p>
                </td>
                </tr>
                </table>
                """).format(
                mdf_file=filename,
                object_id=obj_id,
                plot_filename=plot_filename,
                velocity=drow["velocity"],
                max_displacement=drow["max_displacement"],
            ))

    html = header + "\n".join(plots) + footer
    with open("statplots/index.html", "w") as f:
        f.write(html)
