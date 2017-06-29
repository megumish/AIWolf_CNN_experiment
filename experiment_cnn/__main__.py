from argparse import ArgumentParser
import os, sys, shutil
import logging
sys.path.append('/home/megumish/aiwolf/experiment')
from common.learn_data import conf, gen, loader
import learner

__logger = logging.getLogger(__name__)
def parse_args(config, message_formatter):
    description = 'learn AIWolf data.'
    argparser = ArgumentParser(description=description)

    argparser.add_argument('input_dir', metavar='INPUT_DIR', type=str, help='input data directory')
    argparser.add_argument('epoch_num', metavar='EPOCH_NUM', type=int,  help='learning epoch_num(defalut epoch_num:1)')
    argparser.add_argument('batch_size', metavar='BATCH_SIZE', type=int, help='learning with batch_size(defalut batch_size:1)')

    message_mode = argparser.add_mutually_exclusive_group()
    message_mode.add_argument('-v', '--verbose', action='store_true', help='show verbose message')
    message_mode.add_argument('-q', '--quiet', action='store_true', help='quiet any message')

    argparser.add_argument('-o', '--output_model', metavar='OUTPUT_MODEL', dest='output_model', help='output to a model named <model>')
    argparser.add_argument('-g', '--use_gpu', action='store_true', help='use GPU')

    argparser.add_argument('--dry_run', action='store_true', help='remove the output model after execution')

    mode = argparser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train', action='store_true', help='train from the beginning')
    mode.add_argument('--train_using_model', metavar='MODEL_NAME', type=str, help='train based on the <model_name>')
    mode.add_argument('--test_using_model', metavar='MODEL_NAME', type=str, help='test based on the <model_name>')

    args = argparser.parse_args()

    global __logger
    handler = logging.StreamHandler()
    message_level = logging.WARNING
    if args.verbose:
        message_level = logging.DEBUG
    if args.quiet:
        message_level = logging.ERROR
    __logger.setLevel(message_level)
    handler.setLevel(message_level)
    handler.setFormatter(message_formatter)
    __logger.addHandler(handler)
    __logger.debug("Nyan")
    config.set_message_level_and_formatter(message_level=message_level, message_formatter=message_formatter)

    config.is_dry_run = args.dry_run

    config.set_input_dirs(args.input_dir)
    if not os.path.exists(config.get_input_dir()):
        __logger.error("not exist error INPUT DIR:%s" % (config.input_dir))
        sys.exit()

    config.epoch_num = args.epoch_num
    config.batch_size = args.batch_size

    if hasattr(args, 'output_model'):
        config.set_output_model(args.output_model)
    else:
        input_name = config.get_input_dir()
        if '/' in input_name:
            input_name = input_name.split('/')[-1]
        config.set_output_model(input_name + '_out')

    config.use_gpu = args.use_gpu

    config.learned_model = None
    if args.train:
        config.mode = "train"
    elif args.train_using_model:
        config.learned_model = args.train_using_model
        config.mode = "train"
    elif args.test_using_model:
        config.learned_model = args.test_using_model
        config.mode = "test"

    return message_level
        
if __name__ == "__main__":
    config = conf.Config()
    message_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    message_level = parse_args(config, message_formatter)
    gen.init(config, loader.image_loader.ImageLoader(), message_level=message_level, message_formatter=message_formatter)
    gen.run(learner.CNN_SimpleLearner(message_level=message_level, message_formatter=message_formatter))
