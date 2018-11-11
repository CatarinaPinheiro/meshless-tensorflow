from functools import reduce

import numpy as np
import tensorflow as tf
import itertools as it


class PolynomialBasis:
    def __init__(self, dim, order):
        self.units = [tf.constant(1,dtype=tf.float64)] + [tf.Variable(0,dtype=tf.float64) for _ in range(dim)]  # 1, x, y, z, t, ...
        self.order = order  # (order = 1) => 1,x,y  (order = 2)=> 1,x,y,x*y,x*x,y*y, ...

    @property
    def basis(self):
        b = []
        # make multiplications
        for comb in it.combinations_with_replacement(range(len(self.units)), self.order):
            b.append(reduce(lambda f1, f2: self.units[f1] * self.units[f2], comb))
        return b

    @property
    def zeros(self):
        return [np.array(0, dtype=np.float64) for _ in it.combinations_with_replacement(range(len(self.units)), self.order)]

    @property
    def shape(self):
        return 1, len(self.units)

    def evaluate(self, data):
        for i in range(len(data)):
            self.units[i + 1] = tf.assign(self.units[i + 1], data[i], name="polynomial_basis_evaluate")
        return self.basis
