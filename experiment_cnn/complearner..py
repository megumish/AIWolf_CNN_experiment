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

class Comp_SimpleLearner(log_to_data.converter.BaseConverter):
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
        model.input_size = learn_info.data.image_size
        model.input_vector_size = model.input_size * model.input_size
        model.hidden0_vector_size = 500 
        build_info.write('l0=L.Linear(%s, %s)' % (model.input_vector_size, model.hidden0_vector_size))
        model.hidden1_vector_size = 300 
        build_info.write('l1=L.Linear(%s, %s)' % (model.hidden0_vector_size, model.hidden1_vector_size))
        model.output_vector_size = len(common.role.types)
        build_info.write('l2=L.Linear(%s, %s)' % (model.hidden1_vector_size, model.output_vector_size))
        build_info.close()
        self.__model = CNN_SimpleModel(model)
        if not learn_info.learned_model is None:
            serializers.load_npz(learn_info.learned_model, self.__model)
        self.__optimizer = optimizers.Adam()
        self.__optimizer.setup(self.__model)
        
    def train(self, learn_info):
        xp = numpy
        if learn_info.use_gpu:
            xp = cuda.cupy
            gpu = 0
            cuda.get_device(gpu).use()
            self.__model.to_gpu(gpu)
        X = numpy.array(learn_info.data.inputs[None])
        y = numpy.array(learn_info.data.answers[None])
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
            print('epoch %d' % (epoch))
            print('train mean loss: %f' % (sum_loss  / N))
            print('train mean accuracy: %f' % (sum_accuracy / N))
        if learn_info.use_gpu:
           self.__model.to_cpu()
        serializers.save_npz('%s.npz' % (learn_info.output_model), self.__model)

    def test(self, learn_info):
        xp = numpy
        if learn_info.use_gpu:
            xp = cuda.cupy
            gpu = 0
            cuda.get_device(gpu).use()
            self.__model.to_gpu(gpu)
        count_hits = xp.zeros(6)
        count_hits_each_real = xp.zeros((6,6))
        N = 0
        total_N = 0
        total_sum_accuracy = 0.0
        for real_role in common.role.types:
            X = numpy.array(learn_info.data.inputs[real_role])
            y = numpy.array(learn_info.data.answers[real_role])
            y = y.astype(xp.int32)
            N = y.size
            total_N += N
            X = X.reshape((-1, 1, learn_info.data.image_size, learn_info.data.image_size))
            sum_data = xp.zeros(6)
            sum_accuracy = 0.0
            for i in range(0, N):
                X_batch = xp.asarray([X[i]])
                y_batch = xp.asarray([y[i]])

                accuracy, data = self.__model(X_batch, y_batch, True)
                sum_accuracy += float(accuracy.data)
                total_sum_accuracy
                max_num = xp.amax(data)
                new_data = xp.zeros(6)
                j = 0
                for a_s in data:
                    for n in a_s:
                        if n == max_num:
                            new_data[j] = 1.0
                        else:
                            new_data[j] = 0.0
                        j += 1
                sum_data += new_data
            total_sum_accuracy += sum_accuracy
            count_hits += sum_data
            count_hits_each_real[common.role.str_to_index(real_role)] = sum_data
        print('train mean accuracy: %f' % (total_sum_accuracy / total_N))
        precisions_results = {}
        recalls_results = {}
        f_measures_results = {}
        for real_role in common.role.types:
            real_role_index = common.role.str_to_index(real_role)
            precisions = xp.zeros(6)
            recalls = xp.zeros(6)
            f_measures = xp.zeros(6)
            for assume_role in common.role.types:
                assume_role_index = common.role.str_to_index(assume_role)
                true_positive = count_hits_each_real[real_role_index][assume_role_index]
                precision = true_positive / count_hits[assume_role_index]
                recall = true_positive / N
                f_measure = (2 * recall * precision) / (recall + precision)
                if recall == 0.0 or precision == 0.0:
                    f_measure = 0.0
                precisions[assume_role_index] = precision
                recalls[assume_role_index] = recall
                f_measures[assume_role_index] = f_measure
            precisions_results[real_role] = precisions
            recalls_results[real_role] = recalls
            f_measures_results[real_role] = f_measures
        print('Precisions')
        precisions_table = ''
        is_start = True
        precisions_table += '%9s ' % (' ')
        for real_role in common.role.types:
            if not is_start:
                precisions_table += ' &'
            precisions_table += '%10s' % (real_role)
            is_start = False
        precisions_table += ' \\\\ \\hline \\hline \n'
        for real_role in common.role.types:
            precisions = precisions_results[real_role]
            is_start = True
            precisions_table += '%9s ' % (real_role)
            for assume_role in common.role.types:
                assume_role_index = common.role.str_to_index(assume_role)
                if not is_start:
                    precisions_table += ' &'
                precisions_table += '%10.2f' % (precisions[assume_role_index])
                is_start = False
            precisions_table += ' \\\\ \\hline \n'
        print(precisions_table)
        print('Recalls')
        recalls_table = ''
        is_start = True
        recalls_table += '%9s ' % (' ')
        for real_role in common.role.types:
            if not is_start:
                recalls_table += ' &'
            recalls_table += '%10s' % (real_role)
            is_start = False
        recalls_table += ' \\\\ \\hline \\hline \n'
        for real_role in common.role.types:
            recalls = recalls_results[real_role]
            is_start = True
            recalls_table += '%9s ' % (real_role)
            for assume_role in common.role.types:
                assume_role_index = common.role.str_to_index(assume_role)
                if not is_start:
                    recalls_table += ' &'
                recalls_table += '%10.2f' % (recalls[assume_role_index])
                is_start = False
            recalls_table += ' \\\\ \\hline \n'
        print(recalls_table)
        print('F-Measures')
        f_measures_table = ''
        is_start = True
        f_measures_table += '%9s ' % (' ')
        for real_role in common.role.types:
            if not is_start:
                f_measures_table += ' &'
            f_measures_table += '%10s' % (real_role)
            is_start = False
        f_measures_table += ' \\\\ \\hline \\hline \n'
        for real_role in common.role.types:
            f_measures = f_measures_results[real_role]
            is_start = True
            f_measures_table += '%9s ' % (real_role)
            for assume_role in common.role.types:
                assume_role_index = common.role.str_to_index(assume_role)
                if not is_start:
                    f_measures_table += ' &'
                f_measures_table += '%10.2f' % (f_measures[assume_role_index])
                is_start = False
            f_measures_table += ' \\\\ \\hline \n'
        print(f_measures_table)

class Model:
    pass

class CNN_SimpleModel(chainer.Chain):
    def __init__(self, model):
        super(CNN_SimpleModel, self).__init__(
            l0 = L.Linear(model.input_vector_size, model.hidden0_vector_size),
            l1 = L.Linear(model.hidden0_vector_size, model.hidden1_vector_size),
            l2 = L.Linear(model.hidden1_vector_size, model.output_vector_size)
        )
        self.do_pooling = model.do_pooling

    def __call__(self, x, t, is_test):
        h0 = F.relu(self.l0(x))
        h1 = F.dropout(F.relu(self.l1(h0)), train=not is_test)
        h2 = self.l2(h1)
        if not is_test:
            return F.softmax_cross_entropy(h2, t), F.accuracy(h2, t)
        else:
            return F.accuracy(h2,t) ,F.softmax(h2).data
