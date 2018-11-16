import tensorflow as tf
import numpy as np

from src.lib.weights import gaussian_with_radius


class MovingLeastSquares():
    def __init__(self, data, base):
        self.base = base
        self.data = data
        self.point = np.zeros(np.shape(data[0]))
        self.TFs = [[self.base.evaluate(p), self.base.zeros] for p in self.data]

    @property
    def r_min(self):
        distances = [np.linalg.norm(np.subtract(d, self.point)) for d in self.data]
        return np.sort(distances)[self.base.shape[1]+1]

    def AB(self, r):
        def inside_support(p):
            return np.linalg.norm(p - self.point) < r

        P = [tf.cond(inside_support(p), lambda: self.TFs[i][0], lambda: self.TFs[i][1],name="is_inside_support") for i, p in enumerate(self.data)]
        distances = [tf.norm(tf.subtract(d, self.point)) for d in self.data]
        B = tf.linalg.matmul(tf.transpose(P, name="Pt"), tf.diag([gaussian_with_radius(dist, r) for dist in distances],name="weights"),name="B")
        A = tf.linalg.matmul(B, P,name="A")
        return A, B

    @property
    def phi(self):
        pt = self.base.basis
        condition = lambda ri: tf.linalg.det(self.AB(ri)[0], name = 'det.') < tf.constant(1e-6, dtype=tf.float64)
        def increase_r(ri):
            return ri * 1.05
        r = tf.while_loop(condition, increase_r, [self.r_min], name="phi_find_r_loop")

        A, B = self.AB(r)
        return tf.linalg.matmul([pt], tf.linalg.matmul( tf.matrix_inverse(A) , B,name="invA_B"),name="phi")

    def set_point(self, point):
        self.point = point

    def approximate(self, u):
        return self.phi@ u

