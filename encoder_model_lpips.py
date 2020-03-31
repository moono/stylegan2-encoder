import os
import numpy as np
import tensorflow as tf

from lpips.lpips_tensorflow import learned_perceptual_metric_model
from stylegan2.generator import Synthesis
from utils import adjust_dynamic_range


class EncoderModelLpips(tf.keras.Model):
    def __init__(self, resolutions, featuremaps, image_size, lpips_ckpt_dir, is_on_w, **kwargs):
        super(EncoderModelLpips, self).__init__(**kwargs)
        vgg_ckpt_fn = os.path.join(lpips_ckpt_dir, 'vgg', 'exported')
        lin_ckpt_fn = os.path.join(lpips_ckpt_dir, 'lin', 'exported')

        self.resolutions = resolutions
        self.featuremaps = featuremaps
        self.image_size = image_size
        self.is_on_w = is_on_w
        self.n_broadcast = len(self.resolutions) * 2

        self.broadcast = tf.keras.layers.Lambda(lambda x: tf.tile(x[:, np.newaxis], [1, self.n_broadcast, 1]))
        self.synthesis = Synthesis(self.resolutions, self.featuremaps, name='g_synthesis')
        self.post_process = tf.keras.layers.Lambda(lambda x: self.post_process_image(x[0], x[1]))
        self.perceptual_model = learned_perceptual_metric_model(self.image_size, vgg_ckpt_fn, lin_ckpt_fn)

    def set_weights(self, src_net):
        def split_first_name(name):
            splitted = name.split('/')
            loc = splitted.index('g_synthesis') + 1
            new_name = '/'.join(splitted[loc:])
            return new_name

        n_synthesis_weights = 0
        successful_copies = 0
        for cw in self.weights:
            if 'g_synthesis' in cw.name:
                n_synthesis_weights += 1

                cw_name = split_first_name(cw.name)
                for sw in src_net.weights:
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
        return image

    def run_synthesis_model(self, x):
        w_broadcast = self.broadcast(x) if self.is_on_w else x
        return self.synthesis(w_broadcast)

    def call(self, inputs, training=None, mask=None):
        x, target_image = inputs

        w_broadcast = self.broadcast(x) if self.is_on_w else x
        fake_image = self.synthesis(w_broadcast)
        fake_image = self.post_process([fake_image, self.image_size])
        distance = self.perceptual_model([fake_image, target_image])
        return fake_image, distance


# from tensorflow.keras.models import Model
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input
#
#
# class EncoderModelVgg(tf.keras.Model):
#     def __init__(self, resolutions, featuremaps, image_size, vgg16_layer_names, **kwargs):
#         super(EncoderModelVgg, self).__init__(**kwargs)
#         self.resolutions = resolutions
#         self.featuremaps = featuremaps
#         self.image_size = image_size
#         self.vgg16_layer_names = vgg16_layer_names
#
#         self.synthesis = Synthesis(self.resolutions, self.featuremaps, name='g_synthesis')
#         self.perceptual_model = self.load_perceptual_network()
#
#     def load_perceptual_network(self):
#         vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(self.image_size, self.image_size, 3))
#
#         vgg16_output_layers = [l.output for l in vgg16.layers if l.name in self.vgg16_layer_names]
#         perceptual_model = Model(vgg16.input, vgg16_output_layers)
#
#         # freeze weights
#         for layer in perceptual_model.layers:
#             layer.trainable = False
#
#         return perceptual_model
#
#     def set_weights(self, src_net):
#         def split_first_name(name):
#             splitted = name.split('/')
#             loc = splitted.index('g_synthesis') + 1
#             new_name = '/'.join(splitted[loc:])
#             return new_name
#
#         n_synthesis_weights = 0
#         successful_copies = 0
#         for cw in self.weights:
#             if 'g_synthesis' in cw.name:
#                 n_synthesis_weights += 1
#
#                 cw_name = split_first_name(cw.name)
#                 for sw in src_net.weights:
#                     sw_name = split_first_name(sw.name)
#                     if cw_name == sw_name:
#                         assert sw.shape == cw.shape
#                         cw.assign(sw)
#                         successful_copies += 1
#                         break
#
#         assert successful_copies == n_synthesis_weights
#         return
#
#     @staticmethod
#     def post_process_image(image, image_size):
#         image = adjust_dynamic_range(image, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
#         image = tf.transpose(image, [0, 2, 3, 1])
#         image = tf.image.resize(image, size=(image_size, image_size))
#         image = preprocess_input(image)
#         return image
#
#     def run_synthesis_model(self, embeddings):
#         return self.synthesis(embeddings)
#
#     def run_perceptual_model(self, image):
#         return self.perceptual_model(image)
#
#     def call(self, inputs, training=None, mask=None):
#         w_broadcast = inputs
#
#         fake_image = self.synthesis(w_broadcast)
#         fake_image = self.post_process_image(fake_image, self.image_size)
#         embeddings = self.perceptual_model(fake_image)
#         return fake_image, embeddings
