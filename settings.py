__author__ = 'MA573RWARR10RADMW'
import re

"""
    ~file to set all settings you need~
        PART_SIZE, TEST_PART_SIZE - size of training and testing parts of data
        captcha_stored_dir - directory where captcha for training stored in
        t_captcha_count - count of captcha you need to train on
        model_file - path to model .npz file with weights; for example if you want to teach you model in addition to
            which you already have, you can start training process with declared model_file as path to earlier created
            model
    recommend to change only these parameters: PART_SIZE, TEST_PART_SIZE, captcha_stored_dir, t_captcha_count,


"""

PART_SIZE = 500
TEST_PART_SIZE = 500

captcha_stored_dir = "../train_cap"
t_captcha_count = 30000
model_file = '../train_data_is_in/m/18_10_2016.npz'


urls = ["https://egrul.nalog.ru", "https://service.nalog.ru/inn.do"]

CAPTCHA_FILENAME_PATTERN = re.compile('^\d+_(.*)\..+$')

edged_captcha_path = "../train_data_is_in/edged_captchas/"
train_data_dir = '../train_data_is_in/train_data'
test_file = '../train_data_is_in/test_data/test' + str(t_captcha_count) + '_images_1.npy.npz'
train_file = '../train_data_is_in/train_data/train' + str(t_captcha_count) + '_images_1.npy.npz'
model_dir = '../train_data_is_in/m'
eval_matrix_files_path = '../train_data_is_in/train_data/eval/'