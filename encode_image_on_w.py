import os
import numpy as np
import tensorflow as tf
from PIL import Image

from utils import allow_memory_growth
from stylegan2.generator import Generator
from stylegan2.utils import adjust_dynamic_range
from encoder_models.encoder_model import EncoderModel


class EncodeImage(object):
    def __init__(self, params):
        # set variables
        self.learning_rate = params['learning_rate']
        self.n_train_step = params['n_train_step']
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.image_size = params['image_size']
        self.generator_ckpt_dir = params['generator_ckpt_dir']
        self.lpips_ckpt_dir = params['lpips_ckpt_dir']
        self.output_dir = params['output_dir']
        self.results_on_tensorboard = params['results_on_tensorboard']
        self.output_name_prefix = ''
        self.w_dim = 512
        self.save_every = 100
        self.n_w_samples_to_draw = 10000
        self.run_encoder = False

        # prepare result dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # set model
        self.encoder_model, self.initial_w, sample_image = self.load_encoder_model()

        # prepare variables to optimize
        self.target_image = tf.Variable(
            tf.zeros(shape=(1, self.image_size, self.image_size, 3), dtype=np.float32),
            trainable=False)
        self.w = tf.Variable(
            tf.zeros_like(self.initial_w, dtype=np.float32),
            trainable=True)

        # save initial state images
        self.w.assign(self.initial_w)
        initial_image = self.encoder_model.run_synthesis_model(self.w)
        self.save_image(sample_image, os.path.join(self.output_dir, 'generator_sample.png'))
        self.save_image(initial_image, out_fn=os.path.join(self.output_dir, 'initial_w.png'))

        # prepare tensorboard writer
        if self.results_on_tensorboard:
            self.train_summary_writer = tf.summary.create_file_writer(self.output_dir)
        else:
            self.train_summary_writer = None
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
        initial_zs = np.random.RandomState(123).randn(self.n_w_samples_to_draw, g_params['z_dim'])
        initial_ls = np.ones((self.n_w_samples_to_draw, g_params['labels_dim']), dtype=np.float32)
        initial_ws = generator.g_mapping([initial_zs, initial_ls])
        initial_w = tf.reduce_mean(initial_ws, axis=0, keepdims=True)

        # build encoder model
        encoder_model = EncoderModel(g_params['resolutions'], g_params['featuremaps'], self.image_size,
                                     self.lpips_ckpt_dir)
        test_dlatent = np.random.normal(loc=0.0, scale=1.0, size=(1, g_params['w_dim']))
        test_target_image = np.ones((1, self.image_size, self.image_size, 3), dtype=np.float32)
        _, __ = encoder_model([test_dlatent, test_target_image])

        # copy weights from generator
        encoder_model.set_weights(generator.synthesis)
        _, __ = encoder_model([test_dlatent, test_target_image])

        # freeze weights
        for layer in encoder_model.layers:
            layer.trainable = False

        return encoder_model, initial_w, sample_image

    def set_target_image(self, image_fn):
        self.output_name_prefix = os.path.basename(image_fn)

        # check if result already exists
        full_path = os.path.join(self.output_dir, '{}_final_encoded.npy'.format(self.output_name_prefix))
        if os.path.exists(full_path):
            print('Already encoded: {} !!!'.format(self.output_name_prefix))
            self.run_encoder = False
        else:
            # reset target image & output name
            self.target_image.assign(self.load_image(image_fn, self.image_size))

            # reset w too
            self.w.assign(self.initial_w)
            self.run_encoder = True
        return

    @tf.function
    def step(self):
        with tf.GradientTape() as tape:
            tape.watch([self.w, self.target_image])

            # forward pass
            fake_image, loss = self.encoder_model([self.w, self.target_image])

        t_vars = [self.w]
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
        final_image = self.encoder_model.run_synthesis_model(self.w)
        self.save_image(final_image, out_fn=os.path.join(self.output_dir,
                                                         '{}_final_encoded.png'.format(self.output_name_prefix)))
        np.save(os.path.join(self.output_dir, '{}_final_encoded.npy'.format(self.output_name_prefix)), self.w.numpy())
        return

    def write_to_tensorboard(self, step):
        # get current fake image
        fake_image = self.encoder_model.run_synthesis_model(self.w)
        fake_image = self.convert_image_to_uint8(fake_image)

        # save to tensorboard
        with self.train_summary_writer.as_default():
            tf.summary.histogram('w_{}'.format(self.output_name_prefix), self.w, step=step)
            tf.summary.image('encoded_{}'.format(self.output_name_prefix), fake_image, step=step)
        return


def main():
    import glob

    allow_memory_growth()

    abs_path = os.path.dirname(os.path.abspath(__file__))
    encode_params = {
        'image_size': 256,
        'generator_ckpt_dir': os.path.join(abs_path, './stylegan2/official-converted'),
        'lpips_ckpt_dir': os.path.join(abs_path, 'encoder_models'),
        'output_dir': os.path.join(abs_path, './encode_results', 'on_w'),
        'results_on_tensorboard': True,
        'learning_rate': 0.01,
        'n_train_step': 1000,
    }

    target_images = glob.glob(os.path.join('/home/mookyung/Downloads/labeledAll', '*.png'))
    target_images = sorted(target_images)
    target_images = target_images[:10]
    image_encoder = EncodeImage(encode_params)

    for image_fn in target_images:
        image_encoder.set_target_image(image_fn)
        image_encoder.encode_image()
    return


if __name__ == '__main__':
    main()
