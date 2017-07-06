import logging
import os
import numpy
from PIL import Image
import shutil
import sys
sys.path.append(os.environ['KAWADA_AIWOLF_EXPERIMENT_PATH'])
from common import log_to_data
import common

class CNN_converter(log_to_data.converter.BaseConverter):
    def __init__(self, image_size=-1, message_level=logging.WARNING, message_formatter=None):
        self.__logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        self.__logger.setLevel(message_level)
        handler.setLevel(message_level)
        if not message_formatter is None:
            handler.setFormatter(message_formatter)
        self.__logger.addHandler(handler)

        min_image_size = 0
        for instance in common.action.type_instance_map.values():
            min_image_size += instance.tail_num
        if image_size < 0:
            image_size = min_image_size
        if image_size < min_image_size:
            self.__logger.error("image size is too small, min image size:%s, your setting size:%s" % (min_image_size, image_size))
        self.__image_size = image_size
        self.__evaluator = common.evaluator_numeric.simple.SimpleEvaluator(message_level, message_formatter)

    def convert(self, convert_info):
        for log_index in range(len(convert_info.logs)):
            log_rows = convert_info.logs[log_index]
            game_setting = common.info.GameSetting(log_rows)
            self._init_game_info(game_setting)
            targets = convert_info.narrow_down_targets(log_rows, game_setting)
            data_set = DataSet(self.__logger, self.__image_size, convert_info, game_setting, log_index)
            for num_of_log_row in range(len(log_rows)):
                log_row = log_rows[num_of_log_row]
                self.__log_row_to_data_row(log_row, game_setting, targets, convert_info, data_set)
                convert_info.update_progress()
                is_end = True
                for filenum in convert_info.role_filenum_map.values():
                    if filenum < convert_info.output_num:
                        is_end = False
                        break
                if is_end: return
            self.__logger.debug("out %s" % (str(convert_info.role_filenum_map)))
        if convert_info.is_dry_run:
            shutil.rmtree(convert_info.output_dir)

    def __log_row_to_data_row(self, log_row, game_setting, targets, convert_info, data_set):
        log_type = log_row.split(',')[1]
        if log_type in common.action.types:
            content = common.content.Content(log_row, game_setting)
            if not game_setting.player_names[content.subject] in targets: return
            self.__generate_data(content, data_set, game_setting, convert_info)
            self._update_game_info(content)

    def __generate_data(self, content, data_set, game_setting, convert_info):
        index, values = self.__evaluator.evaluate(content, game_setting, self.game_info)
        data_set.add_and_write_data(content.subject, index, values, game_setting, convert_info)

class DataSet:
    def __init__(self, logger, image_size, convert_info, game_setting, log_index):
        self.__logger = logger
        self.__data = [numpy.zeros((image_size, image_size)) for i in range(game_setting.player_num)]
        self.__log_index = log_index
        self.__data_size = image_size
        self.__output_data_dir = convert_info.output_data_dir
        self.__mode = convert_info.mode
        self.__filenum = [0 for i in range(game_setting.player_num)]

    def add_and_write_data(self, subject, index, values, game_setting, convert_info):
        self.__logger.debug("subject:%s, index:%s, values:%s" % (str(subject), str(index), str(values)))
        role = game_setting.player_roles[subject]
        if convert_info.role_filenum_map[role] == convert_info.output_num:
            self.__logger.debug("over output num, so skip")
            return
        self.__data[subject] = numpy.delete(self.__data[subject], 0, 0)
        data_row = numpy.zeros(self.__data_size)
        for count in range(len(values)):
            value = values[count]
            data_row[index + count] = value
        self.__data[subject] = numpy.append(self.__data[subject], [data_row], axis=0)
        image = Image.fromarray(numpy.uint8(self.__data[subject]))
        #image = image.resize((self.__data_size * 100, self.__data_size * 100))
        if self.__mode == 'train':
            image.save(os.path.join(self.__output_data_dir, '%s_%s_%s.png' % (self.__log_index, subject, self.__filenum[subject])))
        if self.__mode == 'test':
            image.save(os.path.join(self.__output_data_dir, game_setting.player_roles[subject], '%s_%s_%s.png' % (self.__log_index, subject, self.__filenum[subject])))
        answer_file = open(os.path.join(convert_info.output_answer_dir, '%s_%s_%s' % (self.__log_index, subject, self.__filenum[subject])), 'w')
        answer_file.write(str(common.role.str_to_index(game_setting.player_roles[subject])))
        answer_file.close()
        convert_info.role_filenum_map[role] += 1
        self.__filenum[subject] += 1
