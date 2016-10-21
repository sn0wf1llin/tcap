__author__ = 'MA573RWARR10R'
import numpy as np
import os
import re
import sys
import random
from settings import captcha_stored_dir, CAPTCHA_FILENAME_PATTERN
from image_preprocessor.image_preprocessor import ImagePreprocessor
import digits_voc_
from settings import *


def get_filepath_under_dir(dirpath, shuffle=True):
    filenames = (os.listdir(dirpath))
    if shuffle:
        random.shuffle(filenames)

    for filename in filenames:
        if "._" not in filename:# != '.DS_Store':
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                yield filepath


def get_images_shape_under_dir(captcha_in_dir):
    for captcha_filepath in get_filepath_under_dir(captcha_in_dir):
        image_data = ImagePreprocessor.get_image_data(captcha_filepath)
        return image_data.shape
    return None


def get_captchas_filenames(captcha_filepath):
    filename = os.path.basename(captcha_filepath)
    is_matched = CAPTCHA_FILENAME_PATTERN.match(filename)
    assert is_matched is not None, captcha_filepath
    return is_matched.group(1)


def get_captcha_ids_from_path(captcha_filepath):
    captchas_str_filenames_only = get_captchas_filenames(captcha_filepath)
    captchas_ids = np.zeros(len(captchas_str_filenames_only), dtype=np.int32)
    for i, captcha_correct_char in enumerate(captchas_str_filenames_only):
        CHAR_VOC, CHARS = digits_voc_.get_character_voc()
        captchas_ids[i] = CHAR_VOC[captcha_correct_char]

    return captchas_ids


class TrainData(object):
    @classmethod
    def save(cls, filepath, image_data, chars):
        np.savez(filepath, image_data=image_data, chars=chars)

    @classmethod
    def load(cls, filepath, rescale_in_preprocessing=False):

        print 'in traindata.load'

        _train_data = np.load(filepath)
        image_input = _train_data['image_data']

        if rescale_in_preprocessing:
            for row in range(image_input.shape[0]):
                image_input[row, 0, :, :] = ImagePreprocessor.rescale_image_input(image_input[row, 0, :, :])

        else:
            image_input = ImagePreprocessor.normalize_image_input(image_input)

        ret = (image_input, _train_data['chars'])

        del _train_data.f
        _train_data.close()

        print 'out traindata.load'

        return ret

    @classmethod
    def generate_train_data(cls, captcha_in_dir, train_data_dir, max_size, max_captcha_length, prefix):
        image_shape = get_images_shape_under_dir(captcha_in_dir)
        if image_shape is None:
            print 'Give correct path. There are no captcha images in your directory.'
            exit()

        _train_data_shape = tuple(
            [max_size] + list(ImagePreprocessor.get_processed_image_shape(image_shape))
        )
        _train_image_data = np.zeros(_train_data_shape, dtype=np.float32)
        _train_labels = np.zeros((max_size, max_captcha_length), dtype=np.int32)

        i = 0
        for captcha_filepath in get_filepath_under_dir(captcha_in_dir):
            try:
                image_data = ImagePreprocessor.get_image_data(captcha_filepath)
            except Exception as e:
                print e, captcha_filepath
                continue

            i += 1
            index = i % max_size
            _train_image_data[index] = ImagePreprocessor.process_image(image_data)
            captcha_ids = get_captcha_ids_from_path(captcha_filepath)
            _train_labels[index, :] = np.zeros(max_captcha_length, dtype=np.int32)
            _train_labels[index, :captcha_ids.shape[0]] = captcha_ids

            if i != 0 and (i % 1000 == 0):
                print 'processed {0} examples.'.format(i)

            if i != 0 and i % max_size == 0:
                file_path = os.path.join(train_data_dir, prefix + str(t_captcha_count) + '_images_{0}.npy'.format(i / max_size))
                try:
                    cls.save(file_path, _train_image_data, _train_labels)
                    print '\t\t {0} stored!'.format(prefix)
                except Exception as e:
                    print e, ' during saving'


if __name__ == "__main__":
    TrainData.generate_train_data(captcha_stored_dir, '../train_data_is_in/test_data', t_captcha_count, 6,
                                  prefix='test')

    TrainData.generate_train_data(captcha_stored_dir, '../train_data_is_in/train_data', t_captcha_count, 6,
                                  prefix='train')
