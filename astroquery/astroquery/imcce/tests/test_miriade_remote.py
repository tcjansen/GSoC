# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function

from astropy.tests.helper import remote_data
import numpy.testing as npt

from .. import core


@remote_data
class TestMiriadeClass:

    def test_ephemerides(self):
        # check values of Ceres for a given epoch
        # orbital uncertainty of Ceres is basically zero
        res = core.Miriade.get_ephemerides('Ceres', location='500',
                                           epoch=2451544.5)

        assert res['target'] == "Ceres"

        npt.assert_allclose(
            [2451544.5, 188.70280, 9.09829],
            [res['epoch'][0], res['RA'][0], res['DEC'][0]],
            rtol=1e-5)
