import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from stylegan2.generator import Synthesis
from stylegan2.utils import adjust_dynamic_range


class EncoderModel(tf.keras.Model):
    def __init__(self, resolutions, featuremaps, vgg16_layer_names, image_size, **kwargs):
        super(EncoderModel, self).__init__(**kwargs)
        self.resolutions = resolutions
        self.featuremaps = featuremaps
        self.vgg16_layer_names = vgg16_layer_names
        self.image_size = image_size

        self.synthesis = Synthesis(self.resolutions, self.featuremaps, name='g_synthesis')
        self.perceptual_model = self.load_perceptual_network()

    def load_perceptual_network(self):
        vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(self.image_size, self.image_size, 3))

        vgg16_output_layers = [l.output for l in vgg16.layers if l.name in self.vgg16_layer_names]
        perceptual_model = Model(vgg16.input, vgg16_output_layers)

        # freeze weights
        for layer in perceptual_model.layers:
            layer.trainable = False

        return perceptual_model

    def set_weights(self, src_net):
        def split_first_name(name):
            splitted = name.split('/')
            loc = splitted.index('g_synthesis') + 1
            new_name = '/'.join(splitted[loc:])
            return new_name

        n_synthesis_weights = 0
        successful_copies = 0
        for cw in self.trainable_weights:
            if 'g_synthesis' in cw.name:
                n_synthesis_weights += 1

                cw_name = split_first_name(cw.name)
                for sw in src_net.trainable_weights:
                    sw_name = split_first_name(sw.name)
                    if cw_name == sw_name:
                        assert sw.shape == cw.shape
                        cw.assign(sw)
                        successful_copies += 1
                        break

        assert successful_copies == n_synthesis_weights
        return

    @staticmethod
    def post_process_image(image, image_size):
        image = adjust_dynamic_range(image, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
        image = tf.transpose(image, [0, 2, 3, 1])
        image = tf.image.resize(image, size=(image_size, image_size))
        image = preprocess_input(image)
        return image

    def run_synthesis_model(self, embeddings):
        return self.synthesis(embeddings)

    def run_perceptual_model(self, image):
        return self.perceptual_model(image)

    def call(self, inputs, training=None, mask=None):
        w_broadcast = inputs

        fake_image = self.synthesis(w_broadcast)
        fake_image = self.post_process_image(fake_image, self.image_size)
        embeddings = self.perceptual_model(fake_image)
        return fake_image, embeddings
