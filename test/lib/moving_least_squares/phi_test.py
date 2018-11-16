import numpy as np

from src.lib.moving_least_squares import MovingLeastSquares
import unittest
import tensorflow as tf

from test.helpers.base.polynomial_basis import PolynomialBasis
from test.helpers.domain.one_dimensional import exponential


class TestMovingLeastSquares (unittest.TestCase):
    def test_approximation(self):
        data = exponential(20)
        base = PolynomialBasis(1, 2)
        mls = MovingLeastSquares(data[:, 0], base)
        mls.set_point(np.array([2.5]))
        approx = mls.approximate(np.array(data[:, 1]))

        sess = tf.Session()
        value = sess.run(approx)


        print(value)

        self.assertTrue(np.round(value, 2)[0][0] == 0.08)
if __name__ == '__main__':
    unittest.main()