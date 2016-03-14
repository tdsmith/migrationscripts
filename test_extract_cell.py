import unittest

import tifffile as tf

from extract_cell import *

WRITE_OUTPUT = False


class TestExtractWindow(unittest.TestCase):
    def setUp(self):
        self.single_page = tf.imread("test_fixtures/single_page.tif")
        self.multi_page = tf.imread("test_fixtures/multi_page.tif")

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


if __name__ == "__main__":
    import sys
    if "--output" in sys.argv:
        WRITE_OUTPUT = True
        del sys.argv[sys.argv.index("--output")]
    unittest.main()
