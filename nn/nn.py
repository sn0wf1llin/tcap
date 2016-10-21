__author__ = 'MA573RWARR10R'
import theano
import numpy as np
import lasagne
import theano.tensor as T
import generator.digits_voc_ as digits_voc_
import os.path
import time


class NNModel(object):
    """
    class for nn model creating
    """

    def __init__(self,
                 count_of_softmaxes,
                 rnn_steps_count=6,
                 learning_rate=0.01,
                 no_hidden_layers=2,
                 path_to_saved_params=None):

        print 'in model init'

        if not learning_rate:
            self.learning_rate = 0.01
        else:
            self.learning_rate = float(learning_rate)

        if not count_of_softmaxes:
            self.count_of_softmaxes = 6
        else:
            self.count_of_softmaxes = count_of_softmaxes

        if not no_hidden_layers:
            self.no_hidden_layers = 2
        else:
            self.no_hidden_layers = int(no_hidden_layers)

        self.rnn_steps_count = rnn_steps_count

        self.CHARS_VOC, self.CHARS = digits_voc_.get_character_voc()

        self._network, self._train_data, self._test_data, self._pred_data = (
            self._initialize_model_to_predict_digits(self.learning_rate, self.count_of_softmaxes))

        self._prediction = lasagne.layers.get_output(self._network)

        if path_to_saved_params:
            """
            if params existed, then start from their values
            """
            if os.path.isfile(path_to_saved_params) and path_to_saved_params.endswith('.npz'):
                f = np.load(path_to_saved_params)
                print 'loaded from {0}'.format(path_to_saved_params)
                parameters = [f['arr_%d' % i] for i in range(len(f.files))]
                """
                f is taking a lot of memory. delete it
                """
                del f.f
                f.close()
                lasagne.layers.set_all_param_values(self._network, parameters)
            else:
                path_to_saved_params = None
                print '.npz file does not exists'

        print 'out of model init'

    def _save_model_parameters_to_file(self, filepath_dir, filename=None):
        """
        Save model parameters to file for using without recreating
        """
        if not os.path.exists(filepath_dir):
            os.makedirs(filepath_dir)

        if filename is None:
            filename = time.strftime("%d_%m_%Y") + '.npz'#-%H:%M:%S") + '.npz'

        np.savez(filepath_dir + '/' + filename, *lasagne.layers.get_all_param_values(self._network))

        print 'model parameters saved to {0} as {1}'.format(filepath_dir, filename)

    def _get_train_data(self):
        return self._train_data

    def _get_test_data(self):
        return self._test_data

    def _get_predicted_data(self):
        return self._pred_data

    def _get_pretty_print(self, filepath):
        # filepath = '../train_data_is_in/prettypr/print.md'
        theano.printing.pydotprint(self._pred_data, outfile=filepath, var_with_name_simple=True)

    class CNNMaxPoolConfig(object):
        def __init__(self, cnn_filters_count, cnn_filter_size, max_pool_size):
            self.cnn_filters_count = cnn_filters_count
            self.cnn_filter_size = cnn_filter_size
            self.max_pool_size = max_pool_size

    def _initialize_model_to_predict_digits(self, learning_rate, count_of_softmaxes):

        print '\t\t\tin: ~initialize model to predict digits~'

        image_input = T.tensor4('image_input')

        prediction_layer = self._build_model_to_predict_digits(image_input, count_of_softmaxes=count_of_softmaxes)

        target_chars_input = T.imatrix('target_chars_input')

        target_chars = target_chars_input[:, :count_of_softmaxes].reshape(shape=(-1,))

        prediction = lasagne.layers.get_output(prediction_layer)
        l_loss = lasagne.objectives.categorical_crossentropy(prediction, target_chars)
        loss = l_loss.mean()

        params = lasagne.layers.get_all_params(prediction_layer, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate, momentum=0.9)

        test_prediction = lasagne.layers.get_output(prediction_layer, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_chars)
        test_loss = test_loss.mean()

        predicted_chars = T.argmax(test_prediction, axis=1)
        correctly_predicted_chars = T.eq(predicted_chars, target_chars)

        test_acc = T.mean(correctly_predicted_chars, dtype=theano.config.floatX)

        predicted_chars = predicted_chars.reshape(shape=(-1, count_of_softmaxes))
        correctly_predicted_chars = correctly_predicted_chars.reshape(shape=(-1, count_of_softmaxes))
        count_of_matched_chars = T.sum(correctly_predicted_chars, axis=1, dtype=theano.config.floatX)
        seq_test_acc = T.mean(T.eq(count_of_matched_chars, T.fill(count_of_matched_chars, count_of_softmaxes)),
                              dtype=theano.config.floatX)
        test_prediction = test_prediction.reshape(shape=(-1, count_of_softmaxes, len(self.CHARS)))

        _train_f = theano.function([image_input, target_chars_input], loss, updates=updates,
                                   allow_input_downcast=True)

        _test_f = theano.function([image_input, target_chars_input],
                                  [test_loss, test_acc, seq_test_acc],
                                  allow_input_downcast=True)

        _pred_f = theano.function([image_input], [predicted_chars, test_prediction],
                                  allow_input_downcast=True)

        return prediction_layer, _train_f, _test_f, _pred_f

    def _build_model_to_predict_digits(self, image_input, count_of_softmaxes, cnn_max_pool_configs=None,
                                       cnn_dense_layer_sizes=[256], softmax_dense_layer_size=256):
        if cnn_max_pool_configs is None:
            cnn_max_pool_configs = self._default_cnn_max_pool_configs()

        _network = lasagne.layers.InputLayer(shape=(None, 1, 100, 200),
                                             input_var=image_input)

        cnn_dense_layer_sizes = [x * count_of_softmaxes for x in cnn_dense_layer_sizes]

        _network = self._build_cnn(_network, cnn_max_pool_configs, cnn_dense_layer_sizes)

        l_dense_layers = []

        for _ in range(count_of_softmaxes):
            l_dense_layer = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(_network, p=.5),
                num_units=softmax_dense_layer_size,
                nonlinearity=lasagne.nonlinearities.rectify
            )

            l_dense_layer = lasagne.layers.ReshapeLayer(l_dense_layer, ([0], 1, [1]))
            l_dense_layers.append(l_dense_layer)

        l_dense = lasagne.layers.ConcatLayer(l_dense_layers, axis=1)
        l_dense = lasagne.layers.ReshapeLayer(l_dense, (-1, [2]))

        l_softmax = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_dense, p=.5),
            num_units=len(self.CHARS),
            nonlinearity=lasagne.nonlinearities.softmax
        )

        return l_softmax

    @classmethod
    def _build_cnn(cls, network, cnn_max_pool_configs, cnn_dense_layer_sizes):
        for config in cnn_max_pool_configs:
            # convolutional layer
            network = lasagne.layers.Conv2DLayer(
                network, num_filters=config.cnn_filters_count, filter_size=config.cnn_filter_size,
                nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform()
            )

            # max pooling layer
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=config.max_pool_size)

        for dense_layer_size in cnn_dense_layer_sizes:
            # fully connected layer with 50% dropout on inputs to throw out
            # a half of inputs info
            network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=0.5), num_units=dense_layer_size,
                nonlinearity=lasagne.nonlinearities.rectify
            )

        return network

    @classmethod
    def _build_imagenet_cnn(cls, in_layer):
        ConvLayer = lasagne.layers.Conv2DLayer
        DenseLayer = lasagne.layers.DenseLayer
        DropoutLayer = lasagne.layers.DropoutLayer
        PoolLayer = lasagne.layers.MaxPool2DLayer
        NormLayer = lasagne.layers.LocalResponseNormalization2DLayer

        l_layer = in_layer
        l_layer = ConvLayer(l_layer, num_filters=96, filter_size=7, stride=2)
        l_layer = NormLayer(l_layer, alpha=0.0001)
        l_layer = PoolLayer(l_layer, pool_size=3, stride=3, ignore_border=False)
        l_layer = ConvLayer(l_layer, num_filters=256, filter_size=5)
        l_layer = PoolLayer(l_layer, pool_size=2, stride=2, ignore_border=False)
        l_layer = ConvLayer(l_layer, num_filters=512, filter_size=3, pad=1)
        l_layer = ConvLayer(l_layer, num_filters=512, filter_size=3, pad=1)
        l_layer = ConvLayer(l_layer, num_filters=512, filter_size=3, pad=1)
        l_layer = PoolLayer(l_layer, pool_size=3, stride=3, ignore_border=False)
        l_layer = DenseLayer(l_layer, num_units=4096)
        l_layer = DropoutLayer(l_layer, p=0.5)
        l_layer = DenseLayer(l_layer, num_units=4096)
        l_layer = DropoutLayer(l_layer, p=0.5)

        return l_layer

        # def _build_model_to_predict_all_chars(self, image_input, rnn_steps_count, cnn_max_pool_configs=None,
        #                                       cnn_dense_layer_sizes=[256], lstm_layer_units=256, lstm_precompute_input=True,
        #                                       lstm_unroll_scan=False, lstm_grad_clipping=False):
        #
        #     if cnn_max_pool_configs is None:
        #         cnn_max_pool_configs = self._default_cnn_max_pool_configs()
        #
        #     _network = lasagne.layers.InputLayer(shape=(None, 1, 100, 200),
        #                                          input_var=image_input)
        #
        #     l_cnn = self._build_cnn(_network, cnn_max_pool_configs, cnn_dense_layer_sizes=cnn_dense_layer_sizes)
        #     l_cnn = lasagne.layers.ReshapeLayer(l_cnn, ([0], 1, [1]))
        #     l_rnn_input = lasagne.layers.ConcatLayer([l_cnn for _ in range(rnn_steps_count)], axis=1)
        #
        #     l_forward_lstm = lasagne.layers.LSTMLayer(l_rnn_input, num_units=lstm_layer_units,
        #                                               precompute_input=lstm_precompute_input, unroll_scan=lstm_unroll_scan,
        #                                               grad_clipping=lstm_grad_clipping)
        #
        #     l_lstm = l_forward_lstm
        #     l_lstm = lasagne.layers.ReshapeLayer(l_lstm, (-1, lstm_layer_units))
        #
        #     # finally use softmax layer with 50% dropout of its inputs
        #
        #     l_softmax = lasagne.layers.DenseLayer(
        #         lasagne.layers.dropout(l_lstm, p=0.5), num_units=len(self.CHARS),
        #         nonlinearity=lasagne.nonlinearities.softmax)
        #
        #     return l_softmax, l_cnn, l_lstm

    @classmethod
    def _default_cnn_max_pool_configs(cls):
        return [
            cls.CNNMaxPoolConfig(32, (5, 5), (2, 2)),
            cls.CNNMaxPoolConfig(32, (5, 5), (2, 2)),
        ]
