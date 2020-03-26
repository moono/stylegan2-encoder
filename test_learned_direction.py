import os
import glob
import numpy as np
import tensorflow as tf

from utils import allow_memory_growth
from learn_latent_space_direction import load_generator, move_n_save
# from stylegan2.generator import Generator
# from stylegan2.utils import adjust_dynamic_range
from align_images import compute_facial_landmarks, image_align
from encode_image_on_w import EncodeImage


def encode_image(image_fn):
    encode_params = {
        'image_size': 256,
        'generator_ckpt_dir': './stylegan2/official-converted',
        'lpips_ckpt_dir': './encoder_models',
        'output_dir': './test_images/encoded',
        'results_on_tensorboard': False,
        'learning_rate': 0.01,
        'n_train_step': 1000,
    }
    image_encoder = EncodeImage(encode_params)
    image_encoder.set_target_image(image_fn)
    image_encoder.encode_image()

    npy_fn = os.path.join(image_encoder.output_dir, '{}_final_encoded.npy'.format(image_encoder.output_name_prefix))
    return npy_fn


def preprocess_image(src_file):
    # set path
    fn_only = os.path.basename(src_file)
    fn_only_only = fn_only.split('.')[0]
    dst_file = os.path.join('./test_images/aligned', '{}.png'.format(fn_only_only))

    # check if exists
    print('preprocess...align face')
    if os.path.exists(dst_file):
        print('already exists!!')
    else:
        face_landmarks = compute_facial_landmarks(src_file)
        image_align(src_file, dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True)

    print('preprocess...encode aligned image')
    latent_vector_file = encode_image(dst_file)
    latent_vector = np.load(latent_vector_file)
    return latent_vector


def main():
    allow_memory_growth()

    src_file = './test_images/raw/iu-01.jpg'
    # src_file = './test_images/raw/irene-01.jpeg'
    latent_vector = preprocess_image(src_file)

    # load saved learned direction
    latent_direction = np.load(os.path.join('./learned_directions', 'attractive_direction.npy'))

    # load generator for testing
    generator = load_generator(generator_ckpt_dir='./stylegan2/official-converted')

    # try to move to attractive direction
    print('move to specified direction...')
    output_dir = './test_images/latent_space_moved'
    truncation_psi = 0.7
    start, stop, num = -5.0, 5.0, 21
    # start, stop, num = -2.0, 1.0, 21
    coefficients = np.linspace(start=start, stop=stop, num=num, dtype=np.float32)
    output_fn = os.path.join(output_dir, '{}_{}_{}_psi-{}.png'.format(os.path.basename(src_file), start, stop, truncation_psi))
    move_n_save(generator, latent_vector, latent_direction, coefficients, truncation_psi, output_fn)
    return


if __name__ == '__main__':
    main()
