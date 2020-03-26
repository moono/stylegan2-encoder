# ref: https://github.com/Puzer/stylegan-encoder/blob/master/Learn_direction_in_latent_space.ipynb
# ref: https://www.reddit.com/r/MachineLearning/comments/aq6jxf/p_stylegan_encoder_from_real_images_to_latent/
import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils import allow_memory_growth
from stylegan2.generator import Generator
from stylegan2.utils import adjust_dynamic_range


def load_n_merge_encoded_data(encoded_result_dir):
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

    most_attractive = glob.glob(os.path.join(encoded_result_dir, 'most_attractive', '*.npy'))
    least_attractive = glob.glob(os.path.join(encoded_result_dir, 'least_attractive', '*.npy'))
    assert len(most_attractive) == len(least_attractive)

    most_attractive = sorted(most_attractive)
    least_attractive = sorted(least_attractive)
    most_attractive_data = check_nan(most_attractive)
    least_attractive_data = check_nan(least_attractive)
    x_data = np.concatenate((most_attractive_data, least_attractive_data), axis=0)

    y_label1 = np.ones(shape=(most_attractive_data.shape[0],), dtype=np.int32)
    y_label0 = np.zeros(shape=(least_attractive_data.shape[0],), dtype=np.int32)
    y_label = np.concatenate((y_label1, y_label0), axis=0)

    print('n_most_attractive_data: {}'.format(most_attractive_data.shape[0]))
    print('n_least_attractive_data: {}'.format(least_attractive_data.shape[0]))
    return x_data, y_label


def try_score_on_splitted_data(x_data, y_label):
    train_features, test_features, train_labels, test_labels = train_test_split(x_data, y_label)
    print('n_train: {}, n_test: {}'.format(train_features.shape[0], test_features.shape[0]))

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    model = LogisticRegression(class_weight='balanced').fit(train_features, train_labels)
    print('train score: {}'.format(model.score(train_features, train_labels)))
    print('test score: {}'.format(model.score(test_features, test_labels)))
    return


def learn_direction(x_data, y_label):
    w_dim = x_data.shape[-1]

    try_score_on_splitted_data(x_data, y_label)

    scaler = StandardScaler()
    x_features = scaler.fit_transform(x_data)

    model = LogisticRegression(class_weight='balanced').fit(x_features, y_label)
    print('All score: {}'.format(model.score(x_features, y_label)))
    attractive_direction = model.coef_.reshape((w_dim,))
    return attractive_direction


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


def generate_image_with_w(generator, w, truncation_psi, draw_bounding_box=False):
    w = np.reshape(w, newshape=(1, -1))
    w_broadcasted = generator.broadcast(w)
    w_broadcasted = generator.truncation_trick(w_broadcasted, truncation_cutoff=None, truncation_psi=truncation_psi)   # needs check
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


def move_n_show(g, latent_vector, direction, coefficients, truncation_psi):
    fig, ax = plt.subplots(1, len(coefficients), figsize=(15, 10), dpi=80)
    for ii, coeff in enumerate(coefficients):
        new_latent_vector = latent_vector + coeff * direction
        ax[ii].imshow(generate_image_with_w(g, new_latent_vector, truncation_psi))
        ax[ii].set_title('Coeff: {:.1f}'.format(coeff))
    [x.axis('off') for x in ax]
    plt.show()
    return


def move_n_save(g, latent_vector, direction, coefficients, truncation_psi, output_fn):
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

            new_latent_vector = latent_vector + coeff * direction
            image = generate_image_with_w(g, new_latent_vector, truncation_psi, draw_bounding_box)
            image = np.asarray(image)

            canvas[y_start:y_end, x_start:x_end, :] = image

    canvas = Image.fromarray(canvas)
    canvas.save(output_fn)
    return


def main():
    # find attractive direction vector
    encoded_result_dir = '/home/mookyung/Downloads/encoded_result'
    x_data, y_label = load_n_merge_encoded_data(encoded_result_dir)
    attractive_direction = learn_direction(x_data, y_label)
    np.save(os.path.join('./learned_directions', 'attractive_direction.npy'), attractive_direction)
    print()

    # load generator for testing
    allow_memory_growth()
    generator = load_generator(generator_ckpt_dir='./stylegan2/official-converted')

    # try to move to attractive direction
    output_dir = './latent_direction_result'
    n_samples = 5
    truncation_psi = 0.5
    start, stop, num = -5.0, 5.0, 21
    # start, stop, num = -2.0, 1.0, 21
    coefficients = np.linspace(start=start, stop=stop, num=num, dtype=np.float32)

    # most attractive data
    for ii, latent_vector in enumerate(x_data[:n_samples]):
        output_fn = os.path.join(output_dir, 'most_attractive_{}_{}_{}.png'.format(start, stop, ii))
        move_n_save(generator, latent_vector, attractive_direction, coefficients, truncation_psi, output_fn)

    # least attractive data
    for ii, latent_vector in enumerate(x_data[-n_samples:]):
        output_fn = os.path.join(output_dir, 'least_attractive_{}_{}_{}.png'.format(start, stop, ii))
        move_n_save(generator, latent_vector, attractive_direction, coefficients, truncation_psi, output_fn)
    return


if __name__ == '__main__':
    main()
