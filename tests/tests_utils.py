import unittest
import sys
from pathlib import Path

src_folder = str(Path(__file__).parent.parent.joinpath('src'))
if src_folder not in sys.path:
    sys.path.append(src_folder)

from fever_challenge.fever_utils import utils


class TestEuclideanEarthTransformer(unittest.TestCase):

    def setUp(self) -> None:
        self.transformer = utils.EuclideanEarthTransformer(
            lat_min=51.451,
            lat_max=51.560,
            long_min=-0.236313,
            long_max=0.005536,
            lat_scaler=12127.115295910631,
            long_scaler=16812.114800647625,
        )

    def test_euclidean_to_earth(self) -> None:

        test_cases = (
            ({'x_y': (0, 0)}, (51.451, -0.236313)),
            ({'x_y': (16812.114800647625, 0)}, (51.451, 0.005536)),
            ({'x_y': (0, 12127.115295910631)}, (51.560, -0.236313)),
            ({'x_y': (16812.114800647625, 12127.115295910631)}, (51.560, 0.005536)),
        )

        for i_input, i_expected_output in test_cases:
            with self.subTest(f'{i_input} should give {i_expected_output}'):
                actual_output = self.transformer.euclidean_to_earth(**i_input)
                self.assertAlmostEqual(i_expected_output[0], actual_output[0])
                self.assertAlmostEqual(i_expected_output[1], actual_output[1])

    def test_earth_to_euclidean(self) -> None:

        test_cases = (
            ({'lat_long': (51.451, -0.236313)}, (0, 0)),
            ({'lat_long': (51.451, 0.005536)}, (16812.114800647625, 0)),
            ({'lat_long': (51.560, -0.236313)}, (0, 12127.115295910631)),
            ({'lat_long': (51.560, 0.005536)}, (16812.114800647625, 12127.115295910631)),
        )

        for i_input, i_expected_output in test_cases:
            with self.subTest(f'{i_input} should give {i_expected_output}'):
                actual_output = self.transformer.earth_to_euclidean(**i_input)
                self.assertAlmostEqual(i_expected_output[0], actual_output[0])
                self.assertAlmostEqual(i_expected_output[1], actual_output[1])
