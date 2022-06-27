"""
Reading runtime parameters from a standard configuration file (which is easier
to work on than JSON).
"""
import argparse
import json
import logging
import os
from collections import OrderedDict, namedtuple
from typing import Any, IO
import shutil
from attr import has

import numpy as np
import yaml
from pathlib import Path


class Loader(yaml.SafeLoader):
    """ YAML Loader with `!include` constructor. """

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self.root_path = os.path.split(stream.name)[0]
        except AttributeError:
            self.root_path = os.path.curdir

        super().__init__(stream)


class Config:
    """
    Retrieving configuration parameters by parsing a configuration file
    using the YAML configuration file parser.
    """

    _instance = None

    @staticmethod
    def construct_include(loader: Loader, node: yaml.Node) -> Any:
        """Include file referenced at node."""

        filename = os.path.abspath(
            os.path.join(loader.root_path, loader.construct_scalar(node)))
        extension = os.path.splitext(filename)[1].lstrip('.')

        with open(filename, 'r', encoding='utf-8') as config_file:
            if extension in ('yaml', 'yml'):
                return yaml.load(config_file, Loader)
            elif extension in ('json', ):
                return json.load(config_file)
            else:
                return ''.join(config_file.readlines())

    def __new__(cls):
        if cls._instance is None:
            parser = argparse.ArgumentParser()
            parser.add_argument('-i',
                                '--id',
                                type=str,
                                help='Unique client ID.')
            parser.add_argument('-p',
                                '--port',
                                type=str,
                                help='The port number for running a server.')
            parser.add_argument('-c',
                                '--config',
                                type=str,
                                default='./config.yml',
                                help='Federated learning configuration file.')
            parser.add_argument('-b',
                                '--base',
                                type=str,
                                default='./',
                                help='The base path for datasets and models.')
            parser.add_argument('-s',
                                '--server',
                                type=str,
                                default=None,
                                help='The server hostname and port number.')
            parser.add_argument(
                '-d',
                '--download',
                action='store_true',
                help='Download the dataset to prepare for a training session.')
            parser.add_argument(
                '-r',
                '--resume',
                action='store_true',
                help="Resume a previously interrupted training session.")
            parser.add_argument('-l',
                                '--log',
                                type=str,
                                default='info',
                                help='Log messages level.')

            args = parser.parse_args()
            Config.args = args

            if Config.args.id is not None:
                Config.args.id = int(args.id)
            if Config.args.port is not None:
                Config.args.port = int(args.port)

            numeric_level = getattr(logging, args.log.upper(), None)

            if not isinstance(numeric_level, int):
                raise ValueError(f'Invalid log level: {args.log}')

            logging.basicConfig(
                format='[%(levelname)s][%(asctime)s]: %(message)s',
                datefmt='%H:%M:%S')
            root_logger = logging.getLogger()
            root_logger.setLevel(numeric_level)

            cls._instance = super(Config, cls).__new__(cls)

            if 'config_file' in os.environ:
                filename = os.environ['config_file']
            else:
                filename = args.config

            yaml.add_constructor('!include', Config.construct_include, Loader)

            if os.path.isfile(filename):
                with open(filename, 'r', encoding="utf-8") as config_file:
                    config = yaml.load(config_file, Loader)
            else:
                # if the configuration file does not exist, raise an error
                raise ValueError("A configuration file must be supplied.")

            Config.clients = Config.namedtuple_from_dict(config['clients'])
            Config.server = Config.namedtuple_from_dict(config['server'])
            Config.data = Config.namedtuple_from_dict(config['data'])
            Config.trainer = Config.namedtuple_from_dict(config['trainer'])
            Config.algorithm = Config.namedtuple_from_dict(config['algorithm'])

            if Config.args.server is not None:
                Config.server = Config.server._replace(
                    address=args.server.split(':')[0])
                Config.server = Config.server._replace(
                    port=args.server.split(':')[1])

            if Config.args.download:
                Config.clients = Config.clients._replace(total_clients=1)
                Config.clients = Config.clients._replace(per_round=1)

            if hasattr(Config.clients,
                       "speed_simulation") and Config.clients.speed_simulation:
                Config.simulate_client_speed()

            # Customizable dictionary of global parameters
            Config.params: dict = {}

            # A run ID is unique to each client in an experiment
            Config.params['run_id'] = os.getpid()

            # The base path used for all datasets, models, checkpoints, and results
            Config.params['base_path'] = Config.args.base

            if 'results' in config:
                Config.results = Config.namedtuple_from_dict(config['results'])

            if 'general' in config:
                Config.general = Config.namedtuple_from_dict(config['general'])

                Config.switch_running_mode()

                if hasattr(Config.general, 'base_path'):
                    Config.params['base_path'] = Config().general.base_path

            # Directory of dataset
            if hasattr(Config().data, 'data_path'):
                Config.params['data_path'] = os.path.join(
                    Config.params['base_path'],
                    Config().data.data_path)
            else:
                Config.params['data_path'] = os.path.join(
                    Config.params['base_path'], "data")

            # Pretrained models
            if hasattr(Config().server, 'model_path'):
                Config.params['model_path'] = os.path.join(
                    Config.params['base_path'],
                    Config().server.model_path)
            else:
                Config.params['model_path'] = os.path.join(
                    Config.params['base_path'], "models/pretrained")
            os.makedirs(Config.params['model_path'], exist_ok=True)

            # Resume checkpoint
            if hasattr(Config().server, 'checkpoint_path'):
                Config.params['checkpoint_path'] = os.path.join(
                    Config.params['base_path'],
                    Config().server.checkpoint_path)
            else:
                Config.params['checkpoint_path'] = os.path.join(
                    Config.params['base_path'], "checkpoints")
            os.makedirs(Config.params['checkpoint_path'], exist_ok=True)

            # Directory of the .csv file containing results
            if hasattr(Config, 'results') and hasattr(Config.results,
                                                      'result_path'):
                Config.params['result_path'] = os.path.join(
                    Config.params['base_path'], Config.results.result_path)
            else:
                Config.params['result_path'] = os.path.join(
                    Config.params['base_path'], "results")
            os.makedirs(Config.params['result_path'], exist_ok=True)

            # The set of columns in the .csv file
            if hasattr(Config, 'results') and hasattr(Config.results, 'types'):
                Config().params['result_types'] = Config.results.types
            else:
                Config(
                ).params['result_types'] = "round, accuracy, elapsed_time"

            # The set of pairs to be plotted
            if hasattr(Config, 'results') and hasattr(Config.results, 'plot'):
                Config().params['plot_pairs'] = Config().results.plot
            else:
                Config().params[
                    'plot_pairs'] = "round-accuracy, elapsed_time-accuracy"

            if 'model' in config:
                Config.model = Config.namedtuple_from_dict(config['model'])

            # Saving the given config file to the corresponding
            # results/models/checkpoints

            config_file_name = os.path.basename(filename)
            config_result_path = os.path.join(Config.params['model_path'],
                                              config_file_name)
            config_checkpoint_path = os.path.join(
                Config.params['checkpoint_path'], config_file_name)
            config_result_path = os.path.join(Config.params['result_path'],
                                              config_file_name)
            for target_path in [
                    config_result_path, config_checkpoint_path,
                    config_result_path
            ]:
                if not os.path.exists(target_path):
                    shutil.copyfile(src=filename, dst=target_path)

            # Saving the logging information to the .log file
            if hasattr(Config().general,
                       "file_logging") and Config().general.file_logging:

                formatter = logging.Formatter(
                    fmt='[%(levelname)s][%(asctime)s]: %(message)s',
                    datefmt='%H:%M:%S')
                running_mode = Config.general.running_mode

                logging_file_name = Config.make_consistent_save_path(
                    running_mode)
                os.makedirs(os.path.join(Config.params['base_path'],
                                         "loggings"),
                            exist_ok=True)
                log_file_name = os.path.join(Config.params['base_path'],
                                             "loggings",
                                             logging_file_name + ".log")

                file_handler = logging.FileHandler(log_file_name)
                file_handler.setLevel(numeric_level)
                file_handler.setFormatter(formatter)

                root_logger.addHandler(file_handler)

        return cls._instance

    @staticmethod
    def namedtuple_from_dict(obj):
        """Creates a named tuple from a dictionary."""
        if isinstance(obj, dict):
            fields = sorted(obj.keys())
            namedtuple_type = namedtuple(typename='Config',
                                         field_names=fields,
                                         rename=True)
            field_value_pairs = OrderedDict(
                (str(field), Config.namedtuple_from_dict(obj[field]))
                for field in fields)
            try:
                return namedtuple_type(**field_value_pairs)
            except TypeError:
                # Cannot create namedtuple instance so fallback to dict (invalid attribute names)
                return dict(**field_value_pairs)
        elif isinstance(obj, (list, set, tuple, frozenset)):
            return [Config.namedtuple_from_dict(item) for item in obj]
        else:
            return obj

    @staticmethod
    def simulate_client_speed() -> float:
        """Randomly generate a sleep time (in seconds per epoch) for each of the clients."""
        # a random seed must be supplied to make sure that all the clients generate
        # the same set of sleep times per epoch across the board
        if hasattr(Config.clients, "random_seed"):
            np.random.seed(Config.clients.random_seed)
        else:
            np.random.seed(1)

        # Limit the simulated sleep time by the threshold 'max_sleep_time'
        max_sleep_time = 60
        if hasattr(Config.clients, "max_sleep_time"):
            max_sleep_time = Config.clients.max_sleep_time

        dist = Config.clients.simulation_distribution
        total_clients = Config.clients.total_clients
        sleep_times = []

        if hasattr(Config.clients, "simulation_distribution"):

            if dist.distribution.lower() == "normal":
                sleep_times = np.random.normal(dist.mean,
                                               dist.sd,
                                               size=total_clients)
            if dist.distribution.lower() == "pareto":
                sleep_times = np.random.pareto(dist.alpha, size=total_clients)
            if dist.distribution.lower() == "zipf":
                sleep_times = np.random.zipf(dist.s, size=total_clients)
            if dist.distribution.lower() == "uniform":
                sleep_times = np.random.uniform(dist.low,
                                                dist.high,
                                                size=total_clients)
        else:
            # By default, use Pareto distribution with a parameter of 1.0
            sleep_times = np.random.pareto(1.0, size=total_clients)

        Config.client_sleep_times = np.minimum(
            sleep_times, np.repeat(max_sleep_time, total_clients))

    @staticmethod
    def is_edge_server() -> bool:
        """Returns whether the current instance is an edge server in cross-silo FL."""
        return Config().args.port is not None

    @staticmethod
    def is_central_server() -> bool:
        """Returns whether the current instance is a central server in cross-silo FL."""
        return hasattr(Config().algorithm,
                       'cross_silo') and Config().args.port is None

    @staticmethod
    def gpu_count() -> int:
        """Returns the number of GPUs available for training."""
        if hasattr(Config().trainer, 'use_mindspore'):
            return 0

        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 0

    @staticmethod
    def device() -> str:
        """Returns the device to be used for training."""
        device = 'cpu'
        if hasattr(Config().trainer, 'use_mindspore'):
            pass
        elif hasattr(Config().trainer, 'use_tensorflow'):
            import tensorflow as tf

            gpus = tf.config.experimental.list_physical_devices('GPU')
            if len(gpus) > 0:
                device = 'GPU'
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        else:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                if Config.gpu_count() > 1 and isinstance(Config.args.id, int):
                    # A client will always run on the same GPU
                    gpu_id = Config.args.id % torch.cuda.device_count()
                    device = f'cuda:{gpu_id}'
                else:
                    device = 'cuda:0'

        return device

    @staticmethod
    def to_dict() -> dict:
        """ Converting the current run-time configuration to a dict. """

        def items_convert_to_dict(base_dict):
            for key, value in base_dict.items():

                if not isinstance(value, dict):
                    if hasattr(value, "_asdict"):
                        value = value._asdict()
                        value = items_convert_to_dict(value)
                        base_dict[key] = value
                else:
                    value = items_convert_to_dict(value)
                    base_dict[key] = value
            return base_dict

        config_data = dict()
        config_data['clients'] = Config.clients._asdict()
        config_data['server'] = Config.server._asdict()
        config_data['data'] = Config.data._asdict()
        config_data['trainer'] = Config.trainer._asdict()
        config_data['algorithm'] = Config.algorithm._asdict()
        config_data['params'] = Config.params
        for term in [
                "clients", "server", "data", "trainer", "algorithm", "params"
        ]:
            config_data[term] = items_convert_to_dict(config_data[term])

        return config_data

    @staticmethod
    def make_consistent_save_path(running_mode) -> None:
        """ Make the saving path of different parts
            be the same.
        """
        # the path name should be the combination of
        # ssl method name,
        ssl_method_name = "null"
        if hasattr(Config.data, "augment_transformer_name"
                   ) and Config.data.augment_transformer_name:
            ssl_method_name = Config.data.augment_transformer_name

        global_model_name = "null"
        if hasattr(Config.trainer,
                   "global_model_name") and Config.trainer.global_model_name:
            global_model_name = Config.trainer.global_model_name

        personalized_model_name = "null"
        if hasattr(Config.trainer, "personalized_model_name"
                   ) and Config.trainer.personalized_model_name:
            # personalized model name
            personalized_model_name = Config.trainer.personalized_model_name

            if personalized_model_name == "pure_one_layer_mlp":
                personalized_model_name = "pureMLP"

        # dataset name
        datasource = Config.data.datasource
        # encoder name
        model_name = Config.trainer.model_name

        target_name = "_".join([
            ssl_method_name, datasource, model_name, global_model_name,
            personalized_model_name
        ])
        if "central" in running_mode:
            target_name = target_name + "_central"

        return target_name

    @staticmethod
    def switch_running_mode() -> None:
        """ Update the hyper-parameters based on the running mode.

        We support four types of running mode:

        """

        running_mode = Config.general.running_mode

        # setting the base saving path
        Config.server = Config.server._replace(model_path=Path(
            os.path.join("models",
                         Config.make_consistent_save_path(running_mode))))
        Config.server = Config.server._replace(checkpoint_path=Path(
            os.path.join("checkpoints",
                         Config.make_consistent_save_path(running_mode))))
        Config.results = Config.results._replace(result_path=Path(
            os.path.join("results",
                         Config.make_consistent_save_path(running_mode))))

        if running_mode == "user":
            # do not make any changes if the program
            # needs to follow the user's settings.
            return None

        if "code_test" in running_mode:
            # perform the code test mode by using simple consiguration
            # these configurations are set to test the correcness of
            # the code
            logging.info(
                "Performing the code test with simple configurations.")
            Config.clients = Config.clients._replace(do_test=True)
            Config.clients = Config.clients._replace(do_final_eval_test=True)
            Config.clients = Config.clients._replace(test_interval=1)
            Config.clients = Config.clients._replace(eval_test_interval=1)
            Config.clients = Config.clients._replace(total_clients=10)
            Config.clients = Config.clients._replace(per_round=3)

            Config.data = Config.data._replace(partition_size=800)
            Config.data = Config.data._replace(test_partition_size=1000)
            Config.trainer = Config.trainer._replace(rounds=5)
            Config.trainer = Config.trainer._replace(epochs=2)
            Config.trainer = Config.trainer._replace(batch_size=30)

            if hasattr(Config.trainer, "epoch_log_interval"):
                Config.trainer = Config.trainer._replace(epoch_log_interval=1)
            if hasattr(Config.trainer, "epoch_model_log_interval"):
                Config.trainer = Config.trainer._replace(
                    epoch_model_log_interval=1)
            if hasattr(Config.trainer, "batch_log_interval"):
                Config.trainer = Config.trainer._replace(batch_log_interval=5)
            if hasattr(Config.trainer, "pers_epochs"):
                Config.trainer = Config.trainer._replace(pers_epochs=10)
            if hasattr(Config.trainer, "pers_batch_size"):
                Config.trainer = Config.trainer._replace(pers_batch_size=30)
            if hasattr(Config.trainer, "pers_epoch_log_interval"):
                Config.trainer = Config.trainer._replace(
                    pers_epoch_log_interval=1)
            if hasattr(Config.trainer, "pers_epoch_model_log_interval"):
                Config.trainer = Config.trainer._replace(
                    pers_epoch_model_log_interval=1)

        if "central" in running_mode:
            logging.info(
                "Performing the central learing with specific configurations.")
            # apply the central learning
            Config.clients = Config.clients._replace(total_clients=1)
            Config.clients = Config.clients._replace(per_round=1)
            Config.clients = Config.clients._replace(do_final_eval_test=False)
            Config.data = Config.data._replace(sampler="iid")
            Config.data = Config.data._replace(testset_sampler="iid")
            Config.trainer = Config.trainer._replace(rounds=1)

        if "script" in running_mode:
            logging.info(
                "Performing the learing from the script with specific configurations."
            )
            # apply the central learning
            Config.clients = Config.clients._replace(test_interval=1)
            if hasattr(Config.clients, "do_final_eval_test"):
                Config.clients = Config.clients._replace(
                    do_final_eval_test=True)
            if hasattr(Config.clients, "eval_test_interval"):
                Config.clients = Config.clients._replace(
                    eval_test_interval=True)

            Config.server = Config.server._replace(do_test=True)

            Config.trainer = Config.trainer._replace(rounds=1)
