import os
import argparse
import numpy as np

from utils import allow_memory_growth
from learn_latent_space_direction import load_generator, move_n_save
from align_images import compute_facial_landmarks, image_align
from image_encoder import encode_image


def preprocess_image(src_file, output_base_dir, is_on_w, generator_ckpt_dir, lpips_ckpt_dir):
    # set path
    fn_only = os.path.basename(src_file)
    fn_only_only = fn_only.split('.')[0]
    output_dir = os.path.join(output_base_dir, 'aligned')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dst_file = os.path.join(output_dir, '{}.png'.format(fn_only_only))

    # check if exists
    print('preprocess...align face')
    if os.path.exists(dst_file):
        print('already exists!!')
    else:
        face_landmarks = compute_facial_landmarks(src_file)
        image_align(src_file, dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True)

    print('preprocess...encode aligned image')
    latent_vector_file = encode_image(dst_file, {
        'is_on_w': is_on_w,
        'image_size': 256,
        'learning_rate': 0.01,
        'n_train_step': 1000,
        'generator_ckpt_dir': generator_ckpt_dir,
        'lpips_ckpt_dir': lpips_ckpt_dir,
        'output_dir': output_base_dir,
        'results_on_tensorboard': False,
    })
    latent_vector = np.load(latent_vector_file)
    latent_vector = np.squeeze(latent_vector, axis=0)
    return latent_vector


def main():
    # global program arguments parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--allow_memory_growth', default='TRUE', type=str)
    parser.add_argument('--src_file', default='./test_images/irene-01.jpg', type=str)
    parser.add_argument('--generator_ckpt_dir', default='./stylegan2/official-converted', type=str)
    parser.add_argument('--lpips_ckpt_dir', default='./lpips', type=str)
    parser.add_argument('--output_base_dir', default='./outputs', type=str)
    parser.add_argument('--direction_vector_fn', default='./outputs/attractive_direction_on_w_plus.npy', type=str)
    parser.add_argument('--is_on_w', default='FALSE', type=str)
    parser.add_argument('--coeff_start', default=-2, type=int)
    parser.add_argument('--coeff_end', default=2, type=int)
    parser.add_argument('--coeff_num', default=7, type=int)
    parser.add_argument('--truncation_psi', default=1.0, type=float)
    args = vars(parser.parse_args())

    if args['allow_memory_growth'] == 'TRUE':
        allow_memory_growth()

    src_file = args['src_file']
    output_base_dir = args['output_base_dir']
    is_on_w = True if args['is_on_w'] == 'TRUE' else False
    generator_ckpt_dir = args['generator_ckpt_dir']
    lpips_ckpt_dir = args['lpips_ckpt_dir']

    # compute latent vector from input image
    latent_vector = preprocess_image(src_file, output_base_dir, is_on_w, generator_ckpt_dir, lpips_ckpt_dir)

    # load saved learned direction
    direction_vector = np.load(args['direction_vector_fn'])

    # load generator for testing
    generator = load_generator(generator_ckpt_dir=generator_ckpt_dir)

    # try to move to direction
    print('move to specified direction...')
    truncation_psi = args['truncation_psi']
    start, stop, num = args['coeff_start'], args['coeff_end'], args['coeff_num']
    coefficients = np.linspace(start=start, stop=stop, num=num, dtype=np.float32)

    if truncation_psi is None or not is_on_w:
        output_fn = os.path.join(output_base_dir, '{}_{}_{}.png'.format(os.path.basename(src_file),
                                                                        start, stop))
    else:
        output_fn = os.path.join(output_base_dir, '{}_{}_{}_psi-{}.png'.format(os.path.basename(src_file),
                                                                               start, stop, truncation_psi))
    move_n_save(generator, latent_vector, direction_vector, is_on_w, coefficients, truncation_psi, output_fn)
    return


if __name__ == '__main__':
    main()
