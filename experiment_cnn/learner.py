import logging
import numpy
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, optimizers
import shutil
import sys
sys.path.append('/home/megumish/aiwolf/experiment')
from common import log_to_data
import common

class CNN_SimpleLearner(log_to_data.converter.BaseConverter):
    def __init__(self, image_size=-1, message_level=logging.WARNING, message_formatter=None):
        self.__logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        self.__logger.setLevel(message_level)
        handler.setLevel(message_level)
        if not message_formatter is None:
            handler.setFormatter(message_formatter)
        self.__logger.addHandler(handler)

    def build(self, learn_info):
        build_info = open('%s/build.txt' % (learn_info.output_model), 'w')
        model = Model()
        model.input_layer = 1
        model.input_size = learn_info.data.image_size
        model.input_pad = 2
        model.input_filter = model.input_size // 4
        model.conv0_output_layer = 64
        build_info.write('conv0=L.Convolution2D(%s, %s, %s, pad=%s)' % (model.input_layer, model.conv0_output_layer, model.input_filter, model.input_pad))
        # nothing pooling
        model.input_size += model.input_pad * 2
        model.hidden0_size = model.input_size - model.input_filter + 1
        model.hidden0_pad = 1
        model.hidden0_filter = model.hidden0_size // 2
        model.conv1_output_layer = 256
        build_info.write('conv0=L.Convolution2D(%s, %s, %s, pad=%s)' % (model.conv0_output_layer, model.conv1_output_layer, model.hidden0_filter, model.hidden0_pad))
        # nothing pooling
        model.hidden0_size += model.hidden0_pad * 2
        model.hidden1_size = model.hidden0_size - model.hidden0_filter + 1
        model.hidden1_vector_size = model.hidden1_size * model.hidden1_size * model.conv1_output_layer
        model.hidden2_vector_size = model.hidden1_vector_size // 16
        build_info.write('l0=L.Linear(%s, %s)' % (model.hidden1_vector_size, model.hidden2_vector_size))
        model.output_vector_size = len(common.role.types)
        build_info.write('l1=L.Linear(%s, %s)' % (model.hidden2_vector_size, model.output_vector_size))
        build_info.close()
        self.__model = CNN_SimpleModel(model)
        
    def train(self, learn_info):
        xp = numpy
        #if learn_info.use_gpu:
        #    xp = cuda.cupy
        X = xp.array(learn_info.data.inputs[None])
        y = xp.array(learn_info.data.answers[None])
        y = y.astype(xp.int32)
        N = y.size
        X = X.reshape((-1, 1, learn_info.data.image_size, learn_info.data.image_size))
        for epoch in range(learn_info.epoch_num):
            perm = numpy.random.permutation(N)
            sum_loss = 0.0
            sum_accuracy = 0.0
            for i in range(0, N, learn_info.batch_size):
                X_batch = xp.asarray(X[perm[i:i + learn_info.batch_size]])
                y_batch = xp.asarray(y[perm[i:i + learn_info.batch_size]])

                loss, accuracy = self.__model(X_batch, y_batch, False)
                loss.backward()
                sum_loss += float(loss.data) * len(y_batch)
                sum_accuracy += float(accuracy.data) * len(y_batch)
            print('train mean loss: %f' % (sum_loss  / N))
            print('train mean accuracy: %f' % (sum_accuracy / N))

    def test(self, learn_info):
        pass

class Model:
    pass

class CNN_SimpleModel(chainer.Chain):
    def __init__(self, model):
        super(CNN_SimpleModel, self).__init__(
            conv0 = L.Convolution2D(model.input_layer, model.conv0_output_layer, model.input_filter, pad=model.input_pad),
            conv1 = L.Convolution2D(model.conv0_output_layer, model.conv1_output_layer, model.hidden0_filter, pad=model.hidden0_pad),
            l0 = L.Linear(model.hidden1_vector_size, model.hidden2_vector_size),
            l1 = L.Linear(model.hidden2_vector_size, model.output_vector_size)
        )

    def __call__(self, x, t, is_test):
        h0 = F.relu(self.conv0(x))
        h1 = F.relu(self.conv1(h0))
        h2 = F.relu(F.dropout(self.l0(h1)))
        h3 = F.relu(self.l1(h2))
        if not is_test:
            return F.softmax_cross_entropy(h3, t), F.accuracy(h3, t)
        else:
            return F.softmax(h3)
