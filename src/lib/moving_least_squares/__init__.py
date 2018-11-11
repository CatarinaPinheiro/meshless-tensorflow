import tensorflow as tf

from src.lib.weights import gaussian_with_radius


class MovingLeastSquares():
    def __init__(self, data, base):
        self.base = base
        self.data = data
        self.point = tf.Variable(tf.zeros(tf.shape(data[0]),dtype=tf.float64))
        self.TFs = [[self.base.evaluate(p), self.base.zeros] for p in self.data]

    @property
    def r_min(self):
        distances = [tf.norm(tf.subtract(d, self.point)) for d in self.data]
        return tf.contrib.framework.sort(distances)[self.base.shape[1]+1]

    def AB(self, r):
        def inside_support(p):
            return tf.norm(p - self.point) < r

        P = [tf.cond(inside_support(p), lambda: self.TFs[i][0], lambda: self.TFs[i][1]) for i, p in enumerate(self.data)]
        distances = [tf.norm(tf.subtract(d, self.point)) for d in self.data]
        B = tf.transpose(P) @ tf.diag([gaussian_with_radius(dist, r) for dist in distances])
        A = B @ P
        return A, B

    @property
    def phi(self):
        pt = self.base.basis
        condition = lambda ri: tf.linalg.det(self.AB(ri)[0]) < tf.constant(1e-6,dtype=tf.float64)
        def increase_r(ri):
            return ri * 1.05
        r = tf.while_loop(condition, increase_r, [self.r_min], name="phi_find_r_loop")

        A, B = self.AB(r)
        return [pt] @ tf.matrix_inverse(A) @ B

    def set_point(self, point):
        tf.assign(self.point, point)

    def approximate(self, u):
        return self.phi@ u

