import logging
import os
import numpy
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, optimizers
from chainer import serializers
import shutil
import sys
sys.path.append(os.environ['KAWADA_AIWOLF_EXPERIMENT_PATH'])
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
        build_info = open(os.path.join(learn_info.output_model, 'build.txt'), 'w')
        model = Model()
        model.do_pooling = False
        model.input_layer = 1
        model.input_size = learn_info.data.image_size
        model.input_pad = 2
        model.input_size += model.input_pad * 2
        model.input_filter = model.input_size // 4
        model.conv0_output_layer = 20
        build_info.write('conv0=L.Convolution2D(%s, %s, %s, pad=%s)' % (model.input_layer, model.conv0_output_layer, model.input_filter, model.input_pad))
        model.hidden0_size = model.input_size - model.input_filter + 1
        if model.do_pooling:
            model.hidden0_size //= 2
        model.hidden0_pad = 2
        model.hidden0_size += model.hidden0_pad * 2
        model.hidden0_filter = model.hidden0_size // 4
        model.conv1_output_layer = 50
        build_info.write('conv0=L.Convolution2D(%s, %s, %s, pad=%s)' % (model.conv0_output_layer, model.conv1_output_layer, model.hidden0_filter, model.hidden0_pad))
        model.hidden1_size = model.hidden0_size - model.hidden0_filter + 1
        if model.do_pooling:
            model.hidden1_size //= 2
        model.hidden1_vector_size = model.hidden1_size * model.hidden1_size * model.conv1_output_layer
        model.hidden2_vector_size = 500 # model.hidden1_vector_size // 16
        build_info.write('l0=L.Linear(%s, %s)' % (model.hidden1_vector_size, model.hidden2_vector_size))
        model.output_vector_size = len(common.role.types)
        build_info.write('l1=L.Linear(%s, %s)' % (model.hidden2_vector_size, model.output_vector_size))
        build_info.close()
        self.__model = CNN_SimpleModel(model)
        if not learn_info.learned_model is None:
            serializers.load_npz(learn_info.learned_model, self.__model)
        self.__optimizer = optimizers.Adam()
        self.__optimizer.setup(self.__model)
        
    def train(self, learn_info):
        xp = numpy
#       if learn_info.use_gpu:
#           xp = cuda.cupy
#       gpu = 0
#       cuda.get_device(gpu).use()
#       model.to_gpu(gpu)
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

                self.__optimizer.zero_grads()
                loss, accuracy = self.__model(X_batch, y_batch, False)
                loss.backward()
                self.__optimizer.update()
                sum_loss += float(loss.data) * len(y_batch)
                sum_accuracy += float(accuracy.data) * len(y_batch)
            print('train mean loss: %f' % (sum_loss  / N))
            print('train mean accuracy: %f' % (sum_accuracy / N))
        #if learn_info.use_gpu:
        #   model.to_cpu()
        serializers.save_npz('%s.npz' % (learn_info.output_model), self.__model)

    def test(self, learn_info):
        xp = numpy
#       if learn_info.use_gpu:
#           xp = cuda.cupy
#       gpu = 0
#       cuda.get_device(gpu).use()
#       model.to_gpu(gpu)
        for role in common.role.types:
            print('%s:' % (role))
            print(learn_info.data.inputs.keys())
            X = xp.array(learn_info.data.inputs[role])
            y = xp.array(learn_info.data.answers[role])
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

                    self.__optimizer.zero_grads()
                    loss, accuracy = self.__model(X_batch, y_batch, False)
                    loss.backward()
                    self.__optimizer.update()
                    sum_loss += float(loss.data) * len(y_batch)
                    sum_accuracy += float(accuracy.data) * len(y_batch)
                print('train mean loss: %f' % (sum_loss  / N))
                print('train mean accuracy: %f' % (sum_accuracy / N))
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
        h2 = F.dropout(F.relu(self.l0(h1)))
        h3 = self.l1(h2)
        if not is_test:
            return F.softmax_cross_entropy(h3, t), F.accuracy(h3, t)
        else:
            return F.softmax(h3)
