import os
import glob
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

from utils import allow_memory_growth, adjust_dynamic_range
from stylegan2.generator import Generator
from encoder_model_lpips import EncoderModelLpips


class ImageEncoder(object):
    def __init__(self, params):
        # set variables
        self.is_on_w = params['is_on_w']
        self.image_size = params['image_size']
        self.learning_rate = params['learning_rate']
        self.n_train_step = params['n_train_step']
        self.results_on_tensorboard = params['results_on_tensorboard']
        self.generator_ckpt_dir = params['generator_ckpt_dir']
        self.lpips_ckpt_dir = params['lpips_ckpt_dir']
        self.output_dir = os.path.join(params['output_dir'], 'on_w' if params['is_on_w'] else 'on_w_plus')

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.output_name_prefix = ''
        self.output_template_npy = '{:s}_encoded.npy'
        self.output_template_png = '{:s}_encoded.png'
        self.save_every = 100
        self.n_w_samples_to_draw = 10000
        self.run_encoder = False

        # prepare result dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # set model
        self.encoder_model, self.initial_x, sample_image = self.load_encoder_model()

        # prepare variables to optimize
        self.target_image = tf.Variable(
            initial_value=tf.zeros(shape=(1, self.image_size, self.image_size, 3), dtype=np.float32),
            trainable=False)
        self.x = tf.Variable(
            initial_value=tf.zeros_like(self.initial_x, dtype=np.float32),
            trainable=True)

        # save initial state images
        self.x.assign(self.initial_x)
        initial_image = self.encoder_model.run_synthesis_model(self.x)
        self.save_image(sample_image, os.path.join(self.output_dir, 'generator_sample.png'))
        self.save_image(initial_image, out_fn=os.path.join(self.output_dir, 'initial_w.png'))

        # prepare tensorboard writer
        if self.results_on_tensorboard:
            self.train_summary_writer = tf.summary.create_file_writer(self.output_dir)
        else:
            self.train_summary_writer = None
        return

    @staticmethod
    def save_image(fake_image, out_fn):
        image = adjust_dynamic_range(fake_image, range_in=(-1.0, 1.0), range_out=(0.0, 255.0),
                                     out_dtype=tf.dtypes.float32)
        image = tf.transpose(image, [0, 2, 3, 1])
        image = tf.cast(image, dtype=tf.dtypes.uint8)
        image = tf.squeeze(image, axis=0)
        image = Image.fromarray(image.numpy())
        image.save(out_fn)
        return

    @staticmethod
    def load_image(image_fn, image_size):
        image = Image.open(image_fn)
        image = image.resize((image_size, image_size))
        image = np.asarray(image)
        image = np.expand_dims(image, axis=0)
        image = tf.constant(image, dtype=tf.dtypes.float32)
        return image

    @staticmethod
    def convert_image_to_uint8(fake_image):
        image = adjust_dynamic_range(fake_image, range_in=(-1.0, 1.0), range_out=(0.0, 255.0),
                                     out_dtype=tf.dtypes.float32)
        image = tf.transpose(image, [0, 2, 3, 1])
        image = tf.cast(image, dtype=tf.dtypes.uint8)
        return image

    def load_encoder_model(self):
        # build generator object
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

        # sample image
        sample_image, __ = generator([test_latent, test_labels], truncation_psi=0.5, training=False)

        # sample w for statistics
        n_broadcast = len(g_params['resolutions']) * 2
        initial_zs = np.random.RandomState(123).randn(self.n_w_samples_to_draw, g_params['z_dim'])
        initial_ls = np.ones((self.n_w_samples_to_draw, g_params['labels_dim']), dtype=np.float32)
        initial_ws = generator.g_mapping([initial_zs, initial_ls])
        initial_w = tf.reduce_mean(initial_ws, axis=0, keepdims=True)
        initial_w_broadcast = tf.tile(initial_w[:, np.newaxis], [1, n_broadcast, 1])
        initial_var = initial_w if self.is_on_w else initial_w_broadcast

        # build encoder model
        encoder_model = EncoderModelLpips(g_params['resolutions'], g_params['featuremaps'],
                                          self.image_size, self.lpips_ckpt_dir, self.is_on_w)
        if self.is_on_w:
            test_inputs = np.random.normal(loc=0.0, scale=1.0, size=(1, g_params['w_dim']))
        else:
            test_inputs = np.random.normal(loc=0.0, scale=1.0, size=(1, n_broadcast, g_params['w_dim']))
        test_target_image = np.ones((1, self.image_size, self.image_size, 3), dtype=np.float32)
        _, __ = encoder_model([test_inputs, test_target_image])

        # copy weights from generator
        encoder_model.set_weights(generator.synthesis)
        _, __ = encoder_model([test_inputs, test_target_image])

        # freeze weights
        for layer in encoder_model.layers:
            layer.trainable = False

        return encoder_model, initial_var, sample_image

    def set_target_image(self, image_fn):
        self.output_name_prefix = os.path.basename(image_fn)

        # check if result already exists
        full_path = os.path.join(self.output_dir, self.output_template_npy.format(self.output_name_prefix))
        if os.path.exists(full_path):
            print('Already encoded: {} !!!'.format(self.output_name_prefix))
            self.run_encoder = False
        else:
            # reset target image & output name
            self.target_image.assign(self.load_image(image_fn, self.image_size))

            # reset w too
            self.x.assign(self.initial_x)
            self.run_encoder = True
        return

    @tf.function
    def step(self):
        with tf.GradientTape() as tape:
            tape.watch([self.x, self.target_image])

            # forward pass
            fake_image, loss = self.encoder_model([self.x, self.target_image])

        t_vars = [self.x]
        gradients = tape.gradient(loss, t_vars)
        self.optimizer.apply_gradients(zip(gradients, t_vars))
        return loss

    def encode_image(self):
        if not self.run_encoder:
            print('Not encoding: {}'.format(self.output_name_prefix))
            return

        # save initial state
        if self.results_on_tensorboard:
            self.write_to_tensorboard(step=0)

        print('')
        print('Running: {}'.format(self.output_name_prefix))
        for ts in range(1, self.n_train_step + 1):
            # optimize step
            loss_val = self.step()

            # save results
            if ts % self.save_every == 0:
                print('[step {:05d}/{:05d}]: {:.3f}'.format(ts, self.n_train_step, loss_val.numpy()))
                if self.results_on_tensorboard:
                    self.write_to_tensorboard(step=ts)

        # lets restore with optimized embeddings
        final_image = self.encoder_model.run_synthesis_model(self.x)
        self.save_image(final_image,
                        out_fn=os.path.join(self.output_dir, self.output_template_png.format(self.output_name_prefix)))
        np.save(os.path.join(self.output_dir, self.output_template_npy.format(self.output_name_prefix)), self.x.numpy())
        return

    def write_to_tensorboard(self, step):
        # get current fake image
        fake_image = self.encoder_model.run_synthesis_model(self.x)
        fake_image = self.convert_image_to_uint8(fake_image)

        # save to tensorboard
        with self.train_summary_writer.as_default():
            tf.summary.histogram('x_{}'.format(self.output_name_prefix), self.x, step=step)
            tf.summary.image('encoded_{}'.format(self.output_name_prefix), fake_image, step=step)
        return


def encode_image(input_image_fn, output_dir, is_on_w, generator_ckpt_dir, lpips_ckpt_dir, results_on_tensorboard):
    encode_params = {
        'is_on_w': is_on_w,
        'image_size': 256,
        'learning_rate': 0.01,
        'n_train_step': 1000,
        'generator_ckpt_dir': generator_ckpt_dir,
        'lpips_ckpt_dir': lpips_ckpt_dir,
        'output_dir': output_dir,
        'results_on_tensorboard': results_on_tensorboard,
    }

    image_encoder = ImageEncoder(encode_params)
    image_encoder.set_target_image(input_image_fn)
    image_encoder.encode_image()

    result_fn = os.path.join(image_encoder.output_dir,
                             image_encoder.output_template_npy.format(image_encoder.output_name_prefix))
    return result_fn


def batch_encode_images(input_images_dir, output_dir, is_on_w, generator_ckpt_dir, lpips_ckpt_dir, results_on_tensorboard):
    encode_params = {
        'is_on_w': is_on_w,
        'image_size': 256,
        'learning_rate': 0.01,
        'n_train_step': 1000,
        'generator_ckpt_dir': generator_ckpt_dir,
        'lpips_ckpt_dir': lpips_ckpt_dir,
        'output_dir': output_dir,
        'results_on_tensorboard': results_on_tensorboard,
    }

    target_images = glob.glob(os.path.join(input_images_dir, '*.png'))
    target_images = sorted(target_images)
    # target_images = target_images[:3]

    image_encoder = ImageEncoder(encode_params)
    for image_fn in target_images:
        image_encoder.set_target_image(image_fn)
        image_encoder.encode_image()
    return


def main():
    # global program arguments parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--allow_memory_growth', default='TRUE', type=str)
    parser.add_argument('--input_images_dir', default='/home/mookyung/Downloads/labeledAll', type=str)
    parser.add_argument('--generator_ckpt_dir', default='./stylegan2/official-converted', type=str)
    parser.add_argument('--lpips_ckpt_dir', default='./lpips', type=str)
    parser.add_argument('--output_base_dir', default='./outputs', type=str)
    parser.add_argument('--is_on_w', default='TRUE', type=str)
    parser.add_argument('--results_on_tensorboard', default='TRUE', type=str)
    args = vars(parser.parse_args())

    if args['allow_memory_growth'] == 'TRUE':
        allow_memory_growth()

    input_images_dir = args['input_images_dir']
    generator_ckpt_dir = args['generator_ckpt_dir']
    lpips_ckpt_dir = args['lpips_ckpt_dir']
    output_base_dir = args['output_base_dir']
    is_on_w = True if args['is_on_w'] == 'TRUE' else False
    results_on_tensorboard = True if args['results_on_tensorboard'] == 'TRUE' else False
    batch_encode_images(input_images_dir, output_base_dir, is_on_w, generator_ckpt_dir, lpips_ckpt_dir, results_on_tensorboard)
    return


if __name__ == '__main__':
    main()
