import tensorflow as tf
from tensorflow.keras.layers import RandomContrast, RandomZoom, RandomFlip


DEFAULT_SIZE = 320


class DataGen(tf.data.Dataset):

    def _generator(Xtrain, ytrain):
        for x, y in zip(Xtrain, ytrain):
            yield (x, y)

    def __new__(cls, input_size=(640, 640), dataset=None):

        x_size = input_size + (3,)
        y_size = input_size + (3,)
        output_signature = (tf.TensorSpec(shape=x_size, dtype=tf.float32),
                            tf.TensorSpec(shape=y_size, dtype=tf.float32))

        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=output_signature,
            args=dataset
        )


def resize(x, size):
    return tf.image.resize(x, [size, size])


def process_augmentation(x, y, SIZE=DEFAULT_SIZE):

    random_size = int(tf.random.uniform(shape=[1], minval=0.9)*SIZE)[0]

    img = tf.stack([x, y])
    img = tf.image.random_crop(img, [2, 1, random_size, random_size, 3])
    img = tf.transpose(img, perm=[1, 0, 2, 3, 4])
    img = tf.map_fn(lambda x: resize(x, SIZE), img)
    img = tf.transpose(img, perm=[1, 0, 2, 3, 4])
    return img[0], img[1][..., 0, None]


def get_coords(y, augmented=False):
        y = tf.reduce_max(y, axis=-1)
        # HEIGTH AND WIDTH SEGMENTATIN DELIMITATIONS
        h_args = tf.where(tf.reduce_max(y, axis=1) > 0)
        w_args = tf.where(tf.reduce_max(y, axis=0) > 0)

        # TOP-LEFT BBOX AND HEIGTH AND WIDTH CALCULATION
        min_h, min_w = h_args[0], w_args[0]
        heigth, width = h_args[-1] - min_h, w_args[-1] - min_w

        heigth = tf.cast(heigth, tf.int32)
        width = tf.cast(width, tf.int32)

        # SQUARE SIZE FOR BOX CALCULATION
        target_size = tf.maximum(width[0], heigth[0])
        
        
        if augmented:
            factor = 0.4  # HYPERPARAMETRO
            target_size = tf.cast(target_size, tf.float32)
            zoom_factor = 1 + tf.multiply(tf.random.uniform([1]) - 0.5, factor)
            target_size = tf.cast(target_size*zoom_factor, tf.int32)

        min_h, min_w = tf.cast(min_h, tf.int32)[0], tf.cast(min_w, tf.int32)[0]

        # OFFSET FOR CENTER SEGMENTATION
        offset_h = tf.cast((target_size - heigth)/2, tf.int32)[0]
        offset_w = tf.cast((target_size - width)/2, tf.int32)[0]

        # NEW_CORDS CASE: NEW BOX OUT OF THE IMAGE SIZE
        cord_h = tf.squeeze( tf.maximum(0, min_h - offset_h) )
        cord_w = tf.squeeze( tf.maximum(0, min_w - offset_w) )

        # NEW SIZE CASE: NEW BOX OUT OF THE IMAGE SIZE
        h_size = tf.squeeze( tf.minimum(640 - cord_h, target_size) )
        w_size = tf.squeeze( tf.minimum(640 - cord_w, target_size) )

        return cord_h, cord_w, h_size, w_size

def prepare_data(x, y, detection=True, augmented=False, SIZE=DEFAULT_SIZE, original=False):

    # SEGMENTATION MASK TO 1 CHANNEL DIMENSION
    
    y_crop = y
    if detection:

        cord_h, cord_w, h_size, w_size = get_coords(y, augmented=augmented)
        # CROPPING IMAGE AND SEGMENTATION
        x = tf.image.crop_to_bounding_box(x, cord_h, cord_w, h_size, w_size)
        y_crop = tf.image.crop_to_bounding_box(y_crop, cord_h, cord_w, h_size, w_size)

    # RESIZE IMAGE TO DEFINED DEFAULT SIZE
    x = tf.image.resize(x, (SIZE, SIZE))
    y_crop = tf.image.resize(y_crop, (SIZE, SIZE))

    if original:
        return x, (y_crop, y)

    return x, y_crop


def data_pipeline(dataset, SIZE=DEFAULT_SIZE, batch_size=8, contrast_factor=0.25, augmented=False, detection=False, original=False):

    contrast = tf.keras.Sequential([
        RandomContrast(factor=contrast_factor),
    ])

    train_ds = tf.data.Dataset.zip( dataset )
    train_ds = train_ds.cache('')

    train_ds = train_ds.map(lambda x, y:  prepare_data(x, y, augmented=augmented, detection=detection, original=original),
                num_parallel_calls= tf.data.AUTOTUNE)

    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    
    if augmented:
        train_ds = train_ds.map(lambda x, y: (contrast(x), y ),
                    num_parallel_calls= tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(10)

    return train_ds

