import tensorflow as tf

FFHQ_G_PARAMS = {
    'z_dim': 512,
    'w_dim': 512,
    'labels_dim': 0,
    'n_mapping': 8,
    'resolutions': [4, 8, 16, 32, 64, 128, 256, 512, 1024],
    'featuremaps': [512, 512, 512, 512, 512, 256, 128, 64, 32],
}


def load_generator(is_g_clone=False, ckpt_dir=None, trainable=False):
    from stylegan2_ref.generator import Generator

    test_latent = tf.ones((1, FFHQ_G_PARAMS['z_dim']), dtype=tf.float32)
    test_labels = tf.ones((1, FFHQ_G_PARAMS['labels_dim']), dtype=tf.float32)

    # build generator model
    generator = Generator(FFHQ_G_PARAMS)
    _ = generator([test_latent, test_labels])

    if ckpt_dir is not None:
        if is_g_clone:
            ckpt = tf.train.Checkpoint(g_clone=generator)
        else:
            ckpt = tf.train.Checkpoint(generator=generator)
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            print(f'Generator restored from {manager.latest_checkpoint}')

    # set weight status
    for layer in generator.layers:
        layer.trainable = trainable
    return generator


def create_synthesis_from_trained_generator(stylegan2_ckpt_dir, truncation_psi, trainable=False):
    from partial_models.synthesis_w import SynthesisW
    from partial_models.synthesis_w_plus import SynthesisWPlus

    is_on_w = False if truncation_psi is None else True

    # create generator instance
    g_clone = load_generator(is_g_clone=True, ckpt_dir=stylegan2_ckpt_dir)

    # create synthesis model and copy weight from pretrained generator
    if is_on_w:
        synthesis = SynthesisW(g_clone, truncation_psi)
        synthesis.build((None, g_clone.w_dim))
        synthesis.set_weights(g_clone)
    else:
        synthesis = SynthesisWPlus(g_clone)
        synthesis.build((None, g_clone.n_broadcast, g_clone.w_dim))
        synthesis.set_weights(g_clone)

    # set weight status
    for layer in synthesis.layers:
        layer.trainable = trainable
    return synthesis


def load_lpips(lpips_ckpt_dir, image_size, trainable=False):
    import os
    from lpips.lpips_tensorflow import learned_perceptual_metric_model

    # load model
    vgg_ckpt_fn = os.path.join(lpips_ckpt_dir, 'vgg', 'exported')
    lin_ckpt_fn = os.path.join(lpips_ckpt_dir, 'lin', 'exported')
    perceptual_model = learned_perceptual_metric_model(image_size, vgg_ckpt_fn, lin_ckpt_fn)

    # set weight status
    for layer in perceptual_model.layers:
        layer.trainable = trainable
    return perceptual_model


def main():
    from PIL import Image
    from utils import allow_memory_growth, adjust_dynamic_range

    def posprocess_images(images):
        image = adjust_dynamic_range(images, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
        image = tf.transpose(image, [0, 2, 3, 1])
        image = tf.cast(image, dtype=tf.dtypes.uint8)
        image = image[0]
        return image

    allow_memory_growth()

    # test
    stylegan2_ckpt_dir = './stylegan2_ref/official-converted'
    test_cases = [0.5, 1.0, None]
    for truncation_psi in test_cases:
        is_on_w = False if truncation_psi is None else True

        gen = load_generator(is_g_clone=True, ckpt_dir=stylegan2_ckpt_dir)
        syn = create_synthesis_from_trained_generator(stylegan2_ckpt_dir, truncation_psi=truncation_psi)

        batch_size = 1
        z = tf.random.normal([batch_size, 512])
        l = tf.random.normal([batch_size, 0])
        if is_on_w:
            w = gen.g_mapping([z, l])
        else:
            w = gen.g_mapping([z, l])
            w = gen.broadcast(w)
            w = gen.truncation_trick(w, truncation_psi=1.0)
        fake_images1 = gen([z, l], truncation_psi=1.0 if truncation_psi is None else truncation_psi)
        fake_images2 = syn(w)

        fake_images1 = posprocess_images(fake_images1)
        fake_images1 = Image.fromarray(fake_images1.numpy())
        fake_images1.show()

        fake_images2 = posprocess_images(fake_images2)
        fake_images2 = Image.fromarray(fake_images2.numpy())
        fake_images2.show()
    return


if __name__ == '__main__':
    main()
