import tensorflow as tf

from stylegan2_ref.generator import Synthesis


class SynthesisWPlus(tf.keras.models.Model):
    def __init__(self, g_clone, **kwargs):
        super(SynthesisWPlus, self).__init__(**kwargs)

        # create layers
        self.synthesis = Synthesis(g_clone.resolutions, g_clone.featuremaps, name='g_synthesis')

    def set_weights(self, g_clone):
        for w1, w2 in zip(self.synthesis.weights, g_clone.synthesis.weights):
            assert w1.name in w2.name
            assert w1.shape == w2.shape
            w1.assign(w2)
            tf.debugging.assert_equal(w1, w2)
        return

    def call(self, inputs, training=None, mask=None):
        w_broadcasted = inputs
        fake_images = self.synthesis(w_broadcasted)
        return fake_images
