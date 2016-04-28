import os
import unittest

import numpy as np
import pandas as pd
import tifffile as tf

from sting.extract_cell import extract_window, stack_extract_window, movie_of_cell

WRITE_OUTPUT = False


def fixture(filename):
    return os.path.join(
        os.path.dirname(__file__),
        "test_fixtures",
        filename)


class TestExtractWindow(unittest.TestCase):
    def setUp(self):
        self.single_page = tf.imread(fixture("single_page.tif"))
        self.multi_page = tf.imread(fixture("multi_page.tif"))

    def test_extract_window(self):
        for i, (x, y) in enumerate(((0, 0), (100, 100), (250, 250))):
            window = extract_window(self.single_page, x, y, 200, 200)
            self.assertIsInstance(window, np.ndarray)
            self.assertEqual(window.shape, (200, 200))
            if WRITE_OUTPUT:
                tf.imsave("test_extract_window_%d.tif" % i, window)

    def test_stack_extract_window(self):
        xlist = np.array([164, 165, 157, 142])
        ylist = np.array([208, 208, 194, 172])
        extracted = stack_extract_window(self.multi_page, xlist, ylist, 200, 200)
        self.assertIsInstance(extracted, np.ndarray)
        self.assertEqual(extracted.shape, (4, 200, 200))
        if WRITE_OUTPUT:
            tf.imsave("test_stack_extract_window.tif",
                      extracted,
                      photometric="minisblack")

    def test_movie_of_cell(self):
        fake_mdf_1 = pd.DataFrame({
            "Object": [1],
            "X": [100],
            "Y": [100]
        })
        movie = movie_of_cell(
            fake_mdf_1,
            fixture("single_page.tif"),
            200, 200)
        self.assertIsInstance(movie, np.ndarray)
        self.assertEqual(movie.shape, (1, 200, 200))

        # test the pillow code path
        movie = movie_of_cell(
            fake_mdf_1,
            [fixture("single_page.tif")],
            200, 200)
        self.assertIsInstance(movie, np.ndarray)
        self.assertEqual(movie.shape, (1, 200, 200))

        fake_mdf_4 = pd.DataFrame({
            "Object": [1, 1, 1, 1],
            "X": [100, 120, 140, 160],
            "Y": [200, 180, 160, 140],
        })
        movie = movie_of_cell(
            fake_mdf_4,
            fixture("multi_page.tif"),
            200, 200)
        self.assertIsInstance(movie, np.ndarray)
        self.assertEqual(movie.shape, (4, 200, 200))
        if WRITE_OUTPUT:
            tf.imsave("test_movie_of_cell.tif", movie, photometric="minisblack")


if __name__ == "__main__":
    import sys
    if "--output" in sys.argv:
        WRITE_OUTPUT = True
        del sys.argv[sys.argv.index("--output")]
    unittest.main()
