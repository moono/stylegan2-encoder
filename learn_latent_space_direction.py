# ref: https://github.com/Puzer/stylegan-encoder/blob/master/Learn_direction_in_latent_space.ipynb
# ref: https://www.reddit.com/r/MachineLearning/comments/aq6jxf/p_stylegan_encoder_from_real_images_to_latent/
import os
import glob
import argparse
import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils import allow_memory_growth
from stylegan2.generator import Generator
from stylegan2.utils import adjust_dynamic_range


def load_n_merge_encoded_data(data0_dir, data1_dir):
    def check_nan(list_of_npys):
        n_nans = 0
        good_data = list()
        for fn in list_of_npys:
            # check nan
            data = np.load(fn)
            if np.isnan(data).any():
                n_nans += 1
                if np.isnan(data).all():
                    print('Nan: {} ALL'.format(os.path.basename(fn)))
                else:
                    print('Nan: {}'.format(os.path.basename(fn)))
            else:
                good_data.append(data)

        print('There are {} Nans...'.format(n_nans))
        good_data = np.concatenate(good_data, axis=0)
        return good_data

    # load all encoded data
    data0 = glob.glob(os.path.join(data0_dir, '*.npy'))
    data1 = glob.glob(os.path.join(data1_dir, '*.npy'))
    data0 = sorted(data0)
    data1 = sorted(data1)

    # remove nan value latent vectors
    data0 = check_nan(data0)
    data1 = check_nan(data1)

    # create corresponding labels
    label1 = np.ones(shape=(data1.shape[0],), dtype=np.int32)
    label0 = np.zeros(shape=(data0.shape[0],), dtype=np.int32)

    # merge data into x, y
    x_data = np.concatenate((data1, data0), axis=0)
    y_label = np.concatenate((label1, label0), axis=0)

    print('n_data0: {}'.format(data0.shape[0]))
    print('n_data1: {}'.format(data1.shape[0]))
    return x_data, y_label


def try_score_on_splitted_data(x_data, y_label):
    train_features, test_features, train_labels, test_labels = train_test_split(x_data, y_label)
    n_train, n_test = train_features.shape[0], test_features.shape[0]
    print('n_train: {}, n_test: {}'.format(n_train, n_test))

    # normalize data
    train_features = np.reshape(train_features, newshape=(n_train, -1))
    test_features = np.reshape(test_features, newshape=(n_test, -1))
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    model = LogisticRegression(class_weight='balanced').fit(train_features, train_labels)
    print('train score: {}'.format(model.score(train_features, train_labels)))
    print('test score: {}'.format(model.score(test_features, test_labels)))
    return


def learn_direction(x_data, y_label):
    n_data = x_data.shape[0]
    w_dim = x_data.shape[-1]

    # check train / test split score
    try_score_on_splitted_data(x_data, y_label)

    # run on all data
    x_features = np.reshape(x_data, newshape=(n_data, -1))
    scaler = StandardScaler()
    x_features = scaler.fit_transform(x_features)

    model = LogisticRegression(class_weight='balanced').fit(x_features, y_label)
    print('All score: {}'.format(model.score(x_features, y_label)))
    direction_vector = model.coef_.reshape((-1, w_dim))
    return direction_vector


def load_generator(generator_ckpt_dir):
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
    manager = tf.train.CheckpointManager(ckpt, generator_ckpt_dir, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        print('Restored from {}'.format(manager.latest_checkpoint))
    else:
        raise ValueError('Wrong checkpoint dir!!')

    _, __ = generator([test_latent, test_labels], training=False)
    return generator


def generate_image(generator, is_on_w, x, truncation_psi, draw_bounding_box):
    if is_on_w:
        w = np.reshape(x, newshape=(1, -1))
        w_broadcasted = generator.broadcast(w)
        w_broadcasted = generator.truncation_trick(w_broadcasted, truncation_cutoff=None,
                                                   truncation_psi=truncation_psi)  # needs check
    else:
        w_broadcasted = np.reshape(x, newshape=(1, -1, generator.w_dim))

    fake_image = generator.synthesis(w_broadcasted)
    fake_image = adjust_dynamic_range(fake_image, range_in=(-1.0, 1.0), range_out=(0.0, 255.0),
                                      out_dtype=tf.dtypes.float32)
    fake_image = tf.transpose(fake_image, [0, 2, 3, 1])
    fake_image = tf.cast(fake_image, dtype=tf.dtypes.uint8)
    fake_image = tf.squeeze(fake_image, axis=0)

    fake_image = fake_image.numpy()
    fake_image = Image.fromarray(fake_image)
    fake_image = fake_image.resize((256, 256))

    if draw_bounding_box:
        shape = [(0, 0), (255, 255)]
        image_draw = ImageDraw.Draw(fake_image)
        image_draw.rectangle(xy=shape, fill=None, outline='red', width=2)
    return fake_image


def move_n_save(g, latent_vector, direction_vector, is_on_w, coefficients, truncation_psi, output_fn):
    assert latent_vector.shape == direction_vector.shape
    if is_on_w and not isinstance(truncation_psi, float):
        raise ValueError('Need proper truncation psi value!!')

    # flatten input vectors
    latent_vector = np.reshape(latent_vector, newshape=(-1,))
    direction_vector = np.reshape(direction_vector, newshape=(-1,))

    # replace a value if needed
    if 0.0 not in coefficients.tolist():
        loc = (np.abs(coefficients - 0.0)).argmin()
        coefficients[loc] = 0.0

    print('coefficients: {}'.format(coefficients))
    n_rows, n_cols, image_size = 1, len(coefficients), 256
    canvas = np.zeros((n_rows * image_size, n_cols * image_size, 3), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            y_start, y_end = r * image_size, (r + 1) * image_size
            x_start, x_end = c * image_size, (c + 1) * image_size
            index = r + r * n_cols + c
            coeff = coefficients[index]
            draw_bounding_box = True if coeff == 0.0 else False

            new_latent_vector = latent_vector + coeff * direction_vector
            image = generate_image(g, is_on_w, new_latent_vector, truncation_psi, draw_bounding_box)
            image = np.asarray(image)

            canvas[y_start:y_end, x_start:x_end, :] = image

    canvas = Image.fromarray(canvas)
    canvas.save(output_fn)
    return


def main():
    # global program arguments parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--allow_memory_growth', default='TRUE', type=str)
    parser.add_argument('--data0_dir', default='./outputs/encoded_data0', type=str)
    parser.add_argument('--data1_dir', default='./outputs/encoded_data0', type=str)
    parser.add_argument('--output_base_dir', default='./outputs', type=str)
    parser.add_argument('--n_samples', default=5, type=int)
    parser.add_argument('--coeff_start', default=-5, type=int)
    parser.add_argument('--coeff_end', default=5, type=int)
    parser.add_argument('--coeff_num', default=7, type=int)
    parser.add_argument('--truncation_psi', default=1.0, type=float)
    parser.add_argument('--is_on_w', default='FALSE', type=str)
    args = vars(parser.parse_args())

    if args['allow_memory_growth'] == 'TRUE':
        allow_memory_growth()

    # find direction vector
    x_data, y_label = load_n_merge_encoded_data(args['data0_dir'], args['data1_dir'])
    direction_vector = learn_direction(x_data, y_label)
    np.save(os.path.join(args['output_base_dir'], 'direction_vector.npy'), direction_vector)

    # load generator for testing
    generator = load_generator(generator_ckpt_dir='./stylegan2/official-converted')

    # try to move input latent vector with direction vector
    output_dir = os.path.join(args['output_base_dir'], 'direction_walk')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    is_on_w = True if args['is_on_w'] == 'TRUE' else False
    n_samples = args['n_samples']
    truncation_psi = args['truncation_psi']
    start, stop, num = args['coeff_start'], args['coeff_end'], args['coeff_num']
    coefficients = np.linspace(start=start, stop=stop, num=num, dtype=np.float32)

    # data1
    for ii, latent_vector in enumerate(x_data[:n_samples]):
        output_fn = os.path.join(output_dir, 'data1_{}_{}_{}.png'.format(start, stop, ii))
        move_n_save(generator, latent_vector, direction_vector, is_on_w, coefficients, truncation_psi, output_fn)

    # data0
    for ii, latent_vector in enumerate(x_data[-n_samples:]):
        output_fn = os.path.join(output_dir, 'data0_{}_{}_{}.png'.format(start, stop, ii))
        move_n_save(generator, latent_vector, direction_vector, is_on_w, coefficients, truncation_psi, output_fn)
    return


if __name__ == '__main__':
    main()
