import tensorflow as tf

from stylegan2_ref.generator import Synthesis
from stylegan2_ref.utils import lerp


class SynthesisW(tf.keras.models.Model):
    def __init__(self, g_clone, truncation_psi, **kwargs):
        super(SynthesisW, self).__init__(**kwargs)
        assert isinstance(truncation_psi, float)

        # create layers
        self.truncation_psi = truncation_psi
        self.w_avg = tf.Variable(g_clone.w_avg, dtype=tf.float32)
        self.n_broadcast = g_clone.n_broadcast
        self.synthesis = Synthesis(g_clone.resolutions, g_clone.featuremaps, name='g_synthesis')

    def set_weights(self, g_clone):
        self.w_avg.assign(g_clone.w_avg)
        tf.debugging.assert_equal(self.w_avg, g_clone.w_avg)
        for w1, w2 in zip(self.synthesis.weights, g_clone.synthesis.weights):
            assert w1.name in w2.name
            assert w1.shape == w2.shape
            w1.assign(w2)
            tf.debugging.assert_equal(w1, w2)
        return

    def call(self, inputs, truncation_psi=None, training=None, mask=None):
        w = inputs

        truncation_coefs = tf.ones([1, self.n_broadcast, 1], dtype=tf.float32)
        if truncation_psi is None:
            truncation_coefs = truncation_coefs * self.truncation_psi
        else:
            truncation_coefs = truncation_coefs * truncation_psi

        w_broadcasted = tf.tile(w[:, tf.newaxis], [1, self.n_broadcast, 1])
        w_broadcasted = lerp(self.w_avg, w_broadcasted, truncation_coefs)
        fake_images = self.synthesis(w_broadcasted)
        return fake_images
