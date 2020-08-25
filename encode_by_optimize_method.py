import os
import glob
import time
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

from utils import str_to_bool, allow_memory_growth, adjust_dynamic_range
from load_models import create_synthesis_from_trained_generator, load_lpips, load_generator


def sample_initial_w(stylegan2_ckpt_dir, is_on_w, n_w_samples_to_draw=10000):
    # create generator instance
    g_clone = load_generator(is_g_clone=True, ckpt_dir=stylegan2_ckpt_dir)

    # sample w for statistics
    initial_zs = tf.random.normal(shape=[n_w_samples_to_draw, g_clone.z_dim])
    initial_ls = tf.random.normal(shape=[n_w_samples_to_draw, g_clone.labels_dim])
    initial_ws = g_clone.g_mapping([initial_zs, initial_ls])
    initial_w = tf.reduce_mean(initial_ws, axis=0, keepdims=True)
    initial_w_broadcast = g_clone.broadcast(initial_w)
    initial_var = initial_w if is_on_w else initial_w_broadcast
    return initial_var


def load_image(image_fn, image_size):
    image = Image.open(image_fn)
    image = image.resize((image_size, image_size))
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = tf.constant(image, dtype=tf.float32)
    return image


def save_image(fake_image, out_fn):
    image = adjust_dynamic_range(fake_image, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.float32)
    image = tf.transpose(image, [0, 2, 3, 1])
    image = tf.cast(image, dtype=tf.uint8)
    image = tf.squeeze(image, axis=0)
    image = Image.fromarray(image.numpy())
    image.save(out_fn)
    return


@tf.function
def step(x, target_image, image_size, synthesis, lpips, optimizer):
    with tf.GradientTape() as tape:
        tape.watch([x, target_image])

        # forward pass
        fake_image = synthesis(x)
        fake_image = adjust_dynamic_range(fake_image, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.float32)
        fake_image = tf.transpose(fake_image, [0, 2, 3, 1])
        fake_image = tf.image.resize(fake_image, size=(image_size, image_size))

        loss = lpips([fake_image, target_image])

    t_vars = [x]
    gradients = tape.gradient(loss, t_vars)
    optimizer.apply_gradients(zip(gradients, t_vars))
    return loss


def write_to_tensorboard(summary_writer, name, x, synthesis, step_count, loss):
    # get current fake image
    fake_image = synthesis(x)
    fake_image = adjust_dynamic_range(fake_image, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.float32)
    fake_image = tf.transpose(fake_image, [0, 2, 3, 1])
    fake_image = tf.cast(fake_image, dtype=tf.uint8)

    # save to tensorboard
    with summary_writer.as_default():
        tf.summary.scalar(f'loss_{name}', loss, step=step_count)
        tf.summary.image(f'encoded_{name}', fake_image, step=step_count)
    return


def encode(image_fn, synthesis, lpips, initial_x, e_params, save_every, summary_writer):
    fn_only = os.path.basename(image_fn)
    initial_image = load_image(image_fn, e_params['image_size'])

    # check if result already exists
    full_path_npy = os.path.join(e_params['output_dir'], f'{fn_only:s}_encoded.npy')
    if os.path.exists(full_path_npy):
        print(f'Already encoded: {fn_only} !!!')
        return

    # create variables to optimize
    target_image = tf.Variable(tf.zeros(shape=(1, e_params['image_size'], e_params['image_size'], 3), dtype=np.float32), trainable=False)
    x = tf.Variable(tf.zeros_like(initial_x, dtype=np.float32), trainable=True)

    # set initial values for variables
    target_image.assign(initial_image)
    x.assign(initial_x)

    # initialize optimizer
    optimizer = tf.keras.optimizers.Adam(e_params['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    # start optimizing
    print(f'Running: {fn_only}')
    for ts in range(1, e_params['n_train_step'] + 1):
        # optimize step
        loss_val = step(x, target_image, e_params['image_size'], synthesis, lpips, optimizer)

        # save results
        if ts % save_every == 0:
            # check nan
            if np.isnan(loss_val.numpy()):
                print(f'{fn_only}: Nan value during optimization!!')
                return

            # print status
            print(f'[step {ts:05d}/{e_params["n_train_step"]:05d}]: {loss_val.numpy():.3f}')
            if summary_writer is not None:
                write_to_tensorboard(summary_writer, fn_only, x, synthesis, step_count=ts, loss=loss_val)

    # check Nan before saving
    to_save = x.numpy()
    if np.isnan(to_save).all():
        print(f'{fn_only}: Nan value after optimization!!')
        return

    # lets restore with optimized embeddings
    final_image = synthesis(x)
    save_image(final_image, out_fn=os.path.join(e_params['output_dir'], f'{fn_only:s}_encoded.png'))
    np.save(os.path.join(e_params['output_dir'], f'{fn_only:s}_encoded.npy'), x.numpy())
    return


def encode_images(images_dir, e_params):
    # prepare variables
    save_every = 100
    truncation_psi = 0.5 if e_params['is_on_w'] else None

    # prepare result dir
    if not os.path.exists(e_params['output_dir']):
        os.makedirs(e_params['output_dir'])

    # prepare target images
    target_images = glob.glob(os.path.join(images_dir, '*.jpg'))
    target_images += glob.glob(os.path.join(images_dir, '*.png'))
    target_images = sorted(target_images)
    target_images = target_images[:1]

    # sample initial starting point
    initial_x = sample_initial_w(e_params['stylegan2_ckpt_dir'], e_params['is_on_w'])

    if e_params['results_on_tensorboard']:
        summary_writer = tf.summary.create_file_writer(e_params['output_dir'])
    else:
        summary_writer = None

    # prepare models
    synthesis = create_synthesis_from_trained_generator(e_params['stylegan2_ckpt_dir'], truncation_psi)
    lpips = load_lpips(e_params['lpips_ckpt_dir'], e_params['image_size'])

    # start encode images
    for image_fn in target_images:
        encode(image_fn, synthesis, lpips, initial_x, e_params, save_every, summary_writer)
    return


def main():
    # global program arguments parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--allow_memory_growth', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--images_dir', default='/home/mookyung/Downloads/labeledAll', type=str)
    parser.add_argument('--stylegan2_ckpt_dir', default='./stylegan2_ref/official-converted', type=str)
    parser.add_argument('--lpips_ckpt_dir', default='./lpips', type=str)
    parser.add_argument('--output_base_dir', default='./outputs', type=str)
    parser.add_argument('--is_on_w', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--results_on_tensorboard', type=str_to_bool, nargs='?', const=True, default=True)
    args = vars(parser.parse_args())

    if args['allow_memory_growth']:
        allow_memory_growth()

    if args['is_on_w']:
        output_dir = os.path.join(args['output_base_dir'], 'w')
    else:
        output_dir = os.path.join(args['output_base_dir'], 'w_plus')

    encode_params = {
        'is_on_w': args['is_on_w'],
        'image_size': 256,
        'learning_rate': 0.01,
        'n_train_step': 1000,
        'stylegan2_ckpt_dir': args['stylegan2_ckpt_dir'],
        'lpips_ckpt_dir': args['lpips_ckpt_dir'],
        'output_dir': output_dir,
        'results_on_tensorboard': args['results_on_tensorboard'],
    }

    encode_images(args['images_dir'], encode_params)
    return


if __name__ == '__main__':
    main()
