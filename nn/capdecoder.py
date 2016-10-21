__author__ = 'MA573RWARR10R'
import glob
import numpy as np
import os
import sys
from settings import *
import time
from nn import NNModel
from image_preprocessor.image_preprocessor import ImagePreprocessor
from generator.traindata_generator import TrainData, get_filepath_under_dir as getfilepaths
import generator.digits_voc_ as voc
import csv


def iter_parts(inputs, targets, part_size, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_id in range(0, len(inputs) - part_size + 1, part_size):
        if shuffle:
            e = indices[start_id:start_id + part_size]
        else:
            e = slice(start_id, start_id + part_size)
        yield inputs[e], targets[e]


def train(train_data, image_input, target_chars, just_trained_on, eval_matrix, part_size=PART_SIZE):
    train_err = 0
    train_parts = 0
    start_time = time.time()
    print("Training...")

    for image_input_part, target_chars_part in iter_parts(
            image_input, target_chars, part_size, shuffle=False):
        part_start_time = time.time()
        train_inputs = [image_input_part, target_chars_part]

        l_err = train_data(*train_inputs)

        train_err += l_err
        train_parts += 1

        total_images_trained = just_trained_on + train_parts * PART_SIZE
        part_training_time = time.time() - part_start_time
        part_training_loss = train_err / train_parts
        print("trained  {0}  images".format(total_images_trained))
        print("part training took {:.3f}s".format(part_training_time))
        print("training loss:\t\t{:.6f}".format(part_training_loss))

    training_time = (time.time() - start_time)
    training_loss = (train_err / train_parts)
    print (training_loss, type(training_loss))

    # print the results for this epoch:

    print("Training took {:.3f}s".format(training_time))
    eval_matrix.num_of_images_training_time.append((total_images_trained, training_time))
    print("training loss:\t\t{:.6f}".format(training_loss))
    eval_matrix.num_of_images_training_loss.append((total_images_trained, training_loss))


def test(test_data, image_input, target_chars, total_images_trained, testing_on_training, eval_matrix,
         part_size=PART_SIZE):
    print 'in test'
    test_err = 0
    test_acc = 0
    seq_test_acc = 0
    test_parts = 0
    start_time = time.time()

    for image_input_part, target_chars_part in iter_parts(
            image_input, target_chars, part_size, shuffle=False):
        test_inputs = [image_input_part, target_chars_part]

        out = test_data(*test_inputs)
        err, acc, seq_acc = tuple(out)

        test_err += err
        test_acc += acc
        seq_test_acc += seq_acc
        test_parts += 1

    char_acc = ((test_acc / test_parts) * 100)
    seq_acc = ((seq_test_acc / test_parts) * 100)

    if testing_on_training:
        print 'testing on training {:.3f}s'.format(time.time() - start_time)
        eval_matrix.number_of_images_testing_acc.append((total_images_trained, char_acc, seq_acc))
    else:
        print 'testing on testing {:.3f}s'.format(time.time() - start_time)
        eval_matrix.number_of_images_testing_acc.append((total_images_trained, char_acc, seq_acc))

    print '  loss:\t\t{:.6f}'.format(test_err / test_parts)
    print '  char accuracy:\t\t{:.2f} %'.format(char_acc)
    print '  seq accuracy:\t\t{:.2f} %'.format(seq_acc)


class EvalMatrix:
    def __init__(self, path_to_eval):
        parent_dir = os.path.dirname(path_to_eval)

        prefix = time.strftime('%Y_%m_%d')  # _%H_%M_%S')

        self.training_time = os.path.join(parent_dir, "_training_time_" + prefix + ".csv")
        self.training_loss = os.path.join(parent_dir, "_training_loss_" + prefix + ".csv")
        self.training_acc = os.path.join(parent_dir, "_training_acc_" + prefix + ".csv")
        self.testing_acc = os.path.join(parent_dir, "_testing_acc_" + prefix + ".csv")
        self.num_of_images_training_time = []
        self.num_of_images_training_loss = []
        self.number_of_images_training_acc = []
        self.number_of_images_testing_acc = []

    def update_files(self):
        with open(self.training_time, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            for elem in self.num_of_images_training_time:
                csvwriter.writerow(elem)

        with open(self.training_loss, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            for elem in self.num_of_images_training_loss:
                csvwriter.writerow(elem)

        with open(self.training_acc, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            for elem in self.number_of_images_training_acc:
                csvwriter.writerow(elem)

        with open(self.testing_acc, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            for elem in self.number_of_images_testing_acc:
                csvwriter.writerow(elem)


class CaptchaCracker(object):
    def __init__(self, model_params_file_path,
                 rescale_in_preprocessing=False, num_softmaxes=6,
                 rnn_steps_count=6, lstm_layer_units=256, cnn_dense_layer_sizes=[256]):

        # latest_model_saved_params = get_latest_saved_model_file(model_params_file_path)

        self.model = NNModel(count_of_softmaxes=6, path_to_saved_params=model_params_file_path)

        self._predicted_data = self.model._get_predicted_data()
        self._rescale_in_preprocessing = rescale_in_preprocessing
        self._rnn_steps_count = rnn_steps_count

    def predict(self, image_path):
        image_input = ImagePreprocessor.process_image(ImagePreprocessor.get_image_data(image_path))

        return self._predict_on_array(image_input.copy())

    def _predict_on_array(self, image_as_array):
        if self._rescale_in_preprocessing:
            image_as_array = ImagePreprocessor.rescale_image_input(image_as_array)

        else:
            image_as_array = ImagePreprocessor.normalize_image_input(image_as_array)

        _p_i = [np.expand_dims(image_as_array, axis=0)]
        _predicted_ch_id, _pred_probs = self._predicted_data(*_p_i)
        chars = self.model.CHARS

        if _predicted_ch_id.ndim == 1:
            predicted_chars = chars[_predicted_ch_id[0]]
            probs_by_chars = {}
            for i in range(_pred_probs.shape[1]):
                probs_by_chars[chars[i]] = _pred_probs[0, i]

        else:
            assert _predicted_ch_id.ndim == 2, _predicted_ch_id.shape
            predicted_chars = [0] * _predicted_ch_id.shape[1]
            probs_by_chars = [{} for _ in range(_predicted_ch_id.shape[1])]

            for i in range(_predicted_ch_id.shape[1]):
                predicted_chars[i] = chars[_predicted_ch_id[0, i]]
                for j in range(_pred_probs.shape[2]):
                    probs_by_chars[i][chars[j]] = _pred_probs[0, i, j]

        return predicted_chars, probs_by_chars


def predict(model, path_to_captcha):
    predicted_chars, probs = model.predict(path_to_captcha)

    return predicted_chars


def main(epochs):
    global PART_SIZE
    global TEST_PART_SIZE

    learning_rate = 0.1
    no_hidden_layers = 2
    rescale_in_preprocessing = False

    print 'compiling...'

    captcha_cracker_model = NNModel(count_of_softmaxes=6, rnn_steps_count=6, learning_rate=learning_rate,
                                    no_hidden_layers=no_hidden_layers, path_to_saved_params=model_file)

    print('loading validation data...')
    val_image_input, val_target_chars = TrainData.load(train_file,
                                                       rescale_in_preprocessing=rescale_in_preprocessing)
    val_image_input = val_image_input[:TEST_PART_SIZE]
    val_target_chars = val_target_chars[:TEST_PART_SIZE]
    eval_matrix = EvalMatrix(eval_matrix_files_path)

    print('start training...')

    total_images_trained = 0

    for epoch_num in range(epochs):
        for i, training_file in enumerate(getfilepaths(train_data_dir, shuffle=True)):
            print 'training file is {0}'.format(training_file)

            image_input, target_chars = TrainData.load(training_file, rescale_in_preprocessing=rescale_in_preprocessing)

            print '---------------------------------------------------------------------------'
            train(captcha_cracker_model._get_train_data(), image_input, target_chars,
                  total_images_trained, eval_matrix, PART_SIZE)
            print '---------------------------------------------------------------------------'

            # _SaveModelAndRemoveOldOnes(captcha_model, model_params_file_prefix)
            total_images_trained += image_input.shape[0]
            testing_on_train = True
            testing_on_test = False

            print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
            test(captcha_cracker_model._get_test_data(), image_input[:TEST_PART_SIZE],
                 target_chars[:TEST_PART_SIZE], total_images_trained, testing_on_train, eval_matrix,
                 part_size=TEST_PART_SIZE)

            print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
            test(captcha_cracker_model._get_test_data(), val_image_input,
                 val_target_chars, total_images_trained, testing_on_test, eval_matrix,
                 part_size=TEST_PART_SIZE)
            print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

            eval_matrix.update_files()
            eval_matrix = EvalMatrix(eval_matrix_files_path)

            print '____________________________________________________________________________'
            print('Processed epoch:{0} {1} training files.'.format(epoch_num + 1, i + 1))

            print 'save model parameters to file...'
            captcha_cracker_model._save_model_parameters_to_file(filepath_dir=model_dir)
            print '____________________________________________________________________________'

    test_image_input, test_target_chars = TrainData.load(
        test_file, rescale_in_preprocessing=rescale_in_preprocessing)

    print '=========================================================================='
    test(captcha_cracker_model._get_test_data(), test_image_input,
         test_target_chars, total_images_trained, False, eval_matrix, TEST_PART_SIZE)
    print '=========================================================================='

    print 'save model parameters to file...'
    captcha_cracker_model._save_model_parameters_to_file(filepath_dir=model_dir)


# if __name__ == "__main__":
#     """
#         field to start train model
#         epochs - how many times model will look at training database from begin to end for fixing weights
#         model_params_file_path - path to created model; after successful training model will be saved in ../train_data_is_in/m directory
#
#         uncomment lines below to start training
#     """
#     ccracker = CaptchaCracker(model_params_file_path='../train_data_is_in/m/18_10_2016.npz')
#     print predict(ccracker, '../cap/5.jpg')

#
#     main(epochs=10)

