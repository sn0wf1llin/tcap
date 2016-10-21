import os

__author__ = 'MA573RWARR10R'
import numpy as np
from PIL import Image
import scipy.misc
import cv2
from settings import *


class ImagePreprocessor(object):
    @classmethod
    def convert(cls, image_file_path, out_ext):
        img = Image.open(image_file_path)
        in_ext = image_file_path[-4:]
        filename_ = image_file_path[:-4]

        try:
            is_exist = Image.open(filename_ + '.' + out_ext)

            return image_file_path

        except Exception as e:
            img.save(filename_ + '.' + out_ext)
            print e
            try:
                is_s = Image.open(filename_ + "." + out_ext)

                return filename_ + "." + out_ext
            except IOError:
                pass

    @classmethod
    def make_cannied(cls, image_file_path):
        im = Image.open(image_file_path)
        im.convert('RGB').save(image_file_path, 'JPEG')

        im = cv2.imread(image_file_path)
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
        im = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 7, 21)
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)

        edges = cv2.Canny(im, 127, 255)

        return edges

    @classmethod
    def get_processed_image_shape(cls, image_shape):
        assert len(image_shape) == 2, image_shape
        return 1, image_shape[0], image_shape[1]

    @classmethod
    def process_image(cls, numpy_arr):
        assert len(numpy_arr.shape) == 2, numpy_arr.shape
        return np.expand_dims(numpy_arr, axis=0)

    @classmethod
    def get_image_data(cls, image_file_path):

        edges = cls.make_cannied(image_file_path)

        if '\\' in image_file_path:
            a = image_file_path.split('\\')[-1]
        elif '/' in image_file_path:
            a = image_file_path.split('/')[-1]
        else:
            a = image_file_path

        edged_image_path = edged_captcha_path + a

        scipy.misc.imsave(edged_image_path, edges)
        return np.asarray(Image.open(edged_image_path).convert('L'))

    @classmethod
    def normalize_image_input(cls, image_input):
        return image_input / 100.0

    @classmethod
    def rescale_image_input(cls, image_input):
        image_input[image_input > np.percentile(image_input, 88)] = np.max(image_input)
        image_input /= np.max(image_input)
        return image_input

#
# if __name__ == "__main__":
#     i= ImagePreprocessor().get_image_data('../proc/3_274104.png')
#     skimage.io.imsave('new.png', i)
#     ii =  ImagePreprocessor().process_image(i)
#     print ii
