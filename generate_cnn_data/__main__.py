from argparse import ArgumentParser
import os, sys, shutil
import logging
sys.path.append(os.environ['KAWADA_AIWOLF_EXPERIMENT_PATH'])
from common.log_to_data import conf, gen
import converter

__logger = logging.getLogger(__name__)
def parse_args(config, message_formatter):
    description = 'AIWolf log to image.'
    argparser = ArgumentParser(description=description)

    argparser.add_argument('input_dir', metavar='INPUT_DIR', type=str, help='input log file')
    argparser.add_argument('output_num', metavar='OUTPUT_NUM', type=int, help='number of outputs of each role')

    message_mode = argparser.add_mutually_exclusive_group()
    message_mode.add_argument('-v', '--verbose', action='store_true', help='show verbose message')
    message_mode.add_argument('-q', '--quiet', action='store_true', help='quiet any message')

    argparser.add_argument('-o', '--output_dir', metavar='OUTPUT_DIR', dest='ouput_dir', help='output to a directory named <directory>')

    argparser.add_argument('--dry_run', action='store_true', help='remove the output file after execution')

    mode = argparser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train_all', action='store_true', help='train all agents(default)')
    mode.add_argument('--test_all', action='store_true', help='test all agents')
    mode.add_argument('--train', metavar='AGENT_NAME', type=str, nargs='*', help='learn <AGENT_NAME>s')
    mode.add_argument('--test', metavar='AGENT_NAME', type=str, nargs='*', help='test <AGENT_NAME>s')
    mode.add_argument('--train_except', metavar='AGENT_NAME', type=str, nargs='*', help='train all agent except <AGENT_NAME>s')
    mode.add_argument('--test_except', metavar='AGENT_NAME', type=str, nargs='*', help='test all agent except <AGENT_NAME>s')

    choice = argparser.add_mutually_exclusive_group()
    choice.add_argument('--winner', action='store_true', help='only use winners data. By default, use all data.')
    choice.add_argument('--loser', action='store_true', help='only use losers data. By default, use all data.')

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

    config.set_input_dir(args.input_dir)
    if not os.path.exists(config.get_input_dir()):
        __logger.error("not exist error INPUT DIR:%s" % (config.input_dir))
        sys.exit()

    config.set_output_num(args.output_num)

    if hasattr(args, 'output_dir'):
        config.set_output_dirs(args.output_dir)
    else:
        config.set_output_dirs(config.get_input_dir() + '_out')
    if os.path.exists(config.get_output_dir()):
        shutil.rmtree(config.get_output_dir())

    config.include_players = []
    config.except_players = []
    if args.train_all:
        config.mode = "train"
    elif args.test_all:
        config.mode = "test"
    elif args.train:
        for player in args.train:
            config.include_players.append(player)
        config.mode = "train"
    elif args.test:
        for player in args.test:
            config.include_players.append(player)
    elif args.train_except:
        for player in args.train_except:
            config.except_players.append(player)
        config.mode = "train"
    elif args.test_except:
        for player in args.test_except:
            config.except_players.append(player)
        config.mode = "test"

    config.choice = "all"
    if args.winner:
        config.choice = "winner"
    elif args.loser:
        config.choice = "loser"

    return message_level
        
if __name__ == "__main__":
    config = conf.Config()
    message_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    message_level = parse_args(config, message_formatter)
    gen.init(config, message_level=message_level, message_formatter=message_formatter)
    gen.run(converter.CNN_converter(message_level=message_level, message_formatter=message_formatter))
