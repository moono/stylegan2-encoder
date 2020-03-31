import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D

from utils import adjust_dynamic_range
from stylegan2.generator import Generator
from lpips.lpips_tensorflow import learned_perceptual_metric_model


class LinkedEncoder(Model):
    def __init__(self, image_size, n_output_classes, generator_ckpt_dir, lpips_ckpt_dir, **kwargs):
        super(LinkedEncoder, self).__init__(**kwargs)
        vgg_ckpt_fn = os.path.join(lpips_ckpt_dir, 'vgg', 'exported')
        lin_ckpt_fn = os.path.join(lpips_ckpt_dir, 'lin', 'exported')

        self.image_size = image_size
        self.n_output_classes = n_output_classes
        self.generator_ckpt_dir = generator_ckpt_dir

        self.resnet = self.build_encoder_model()
        self.generator, self.dummy_label = self.load_generator()
        self.post_process = tf.keras.layers.Lambda(lambda x: self.post_process_image(x[0], x[1]))
        self.perceptual_model = learned_perceptual_metric_model(self.image_size, vgg_ckpt_fn, lin_ckpt_fn)

    def build_encoder_model(self):
        inputs = Input(shape=(self.image_size, self.image_size, 3), dtype='float32', name='inputs')
        base_model = ResNet50V2(include_top=False, weights='imagenet', input_tensor=inputs)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)

        # and a logistic layers
        predictions = [Dense(n_classes, activation='softmax')(x) for n_classes in self.n_output_classes]
        model = Model(base_model.input, predictions, name='encoder_model')

        # set all layers trainable
        for layer in model.layers:
            layer.trainable = True
        return model

    def load_generator(self):
        g_params = {
            'z_dim': 512,
            'w_dim': 512,
            'labels_dim': 0,
            'n_mapping': 8,
            'resolutions': [4, 8, 16, 32, 64, 128, 256, 512, 1024],
            'featuremaps': [512, 512, 512, 512, 512, 256, 128, 64, 32],
            'w_ema_decay': 0.995,
            'style_mixing_prob': 0.9,
        }
        generator = Generator(g_params)
        test_latent = np.random.normal(loc=0.0, scale=1.0, size=(1, g_params['z_dim']))
        test_labels = np.ones((1, g_params['labels_dim']), dtype=np.float32)
        _, __ = generator([test_latent, test_labels], training=False)

        # try to restore from g_clone
        ckpt = tf.train.Checkpoint(g_clone=generator)
        manager = tf.train.CheckpointManager(ckpt, self.generator_ckpt_dir, max_to_keep=1)
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            print('Restored from {}'.format(manager.latest_checkpoint))
        else:
            raise ValueError('Wrong checkpoint dir!!')

        _, __ = generator([test_latent, test_labels], training=False)

        # set all layers not-trainable
        for layer in generator.layers:
            layer.trainable = False
        return generator, test_labels

    @staticmethod
    def post_process_image(image, image_size):
        image = adjust_dynamic_range(image, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
        image = tf.transpose(image, [0, 2, 3, 1])
        image = tf.image.resize(image, size=(image_size, image_size))
        return image

    def call(self, inputs, training=None, mask=None):
        target_image = inputs

        # images: (None, 256, 256, 3), [0.0 ~ 255.0] float32
        target_image = preprocess_input(target_image)

        feature_list = self.resnet(target_image)
        generator_inputs = tf.concat(feature_list, axis=1)
        fake_image, _ = self.generator([generator_inputs, self.dummy_label])
        fake_image = self.post_process([fake_image, self.image_size])
        distance = self.perceptual_model([fake_image, target_image])
        return distance


@tf.function()
def step(model, images):
    with tf.GradientTape() as tape:
        tape.watch(images)

        # forward pass
    return


def main():
    learning_rate = 0.01
    image_size = 256
    n_output_classes = [100, 100, 100, 100, 100, 12]
    generator_ckpt_dir = './stylegan2/official-converted'
    lpips_ckpt_dir = './lpips'
    model = LinkedEncoder(image_size, n_output_classes, generator_ckpt_dir, lpips_ckpt_dir)
    temp_image = np.random.uniform(low=0.0, high=255.0, size=(1, image_size, image_size, 3))
    distance = model(temp_image)
    print()

    # dataset = None
    # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #
    # for x in dataset:

    return


if __name__ == '__main__':
    main()
