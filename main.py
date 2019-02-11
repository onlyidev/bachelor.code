# -*- coding: utf-8 -*-
r"""
    src.main
    ~~~~~~~~

    Main module for testing and debugging the MalGAN implementation.

    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""

import argparse
import pickle
import sys
import logging
from typing import Union
from pathlib import Path

import numpy as np

import torch
from torch import nn

from malgan import MalGAN, MalwareDataset, BlackBoxDetector


def setup_logger(quiet_mode: bool, filename: str = "tester.log", log_level: int = logging.DEBUG):
    r"""
    Logger Configurator

    Configures the test logger.

    :param quiet_mode: True if quiet mode (i.e., disable logging to stdout) is used
    :param filename: Log file name
    :param log_level: Level to log
    """
    date_format = '%m/%d/%Y %I:%M:%S %p'  # Example Time Format - 12/12/2010 11:46:36 AM
    format_str = '%(asctime)s -- %(levelname)s -- %(message)s'
    logging.basicConfig(filename=filename, level=log_level, format=format_str, datefmt=date_format)

    # Also print to stdout
    if not quiet_mode:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(format_str)
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

    # Matplotlib clutters the logger so change its log level
    # noinspection PyProtectedMember
    # matplotlib._log.setLevel(logging.INFO)  # pylint: disable=protected-access

    logging.info("******************* New Run Beginning *****************")


def parse_args() -> argparse.Namespace:
    r"""
    Parse the command line arguments

    :return: Parsed argument structure
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("Z", help="Dimension of the latent vector", type=int, default=10)
    parser.add_argument("batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("num_epoch", help="Number of training epochs", type=int, default=100)

    msg = "Data file contacting the %s feature vectors"
    for x in ["malware", "benign"]:
        parser.add_argument(x[:3] + "_file", help=msg % x, type=Path, default="data/%s.npy" % x)

    parser.add_argument("-q", help="Quiet mode", action='store_true', default=False)

    help_msg = " ".join(["Dimension of the hidden layer(s) in the GENERATOR."
                         "Multiple layers should be space separated"])
    parser.add_argument("--gen-hidden-sizes", help=help_msg, type=int,
                        default=[256, 256], nargs="+")

    help_msg = " ".join(["Dimension of the hidden layer(s) in the DISCRIMINATOR."
                         "Multiple layers should be space separated"])
    parser.add_argument("--discrim-hidden-sizes", help=help_msg, type=int,
                        default=[256, 256], nargs="+")

    help_msg = " ".join(["Activation function for the generator and discriminatior hidden",
                         "layer(s). Valid choices (case insensitive) are: \"ReLU\", \"ELU\",",
                         "\"LeakyReLU\", \"tanh\" and \"sigmoid\"."])
    parser.add_argument("--activation", help=help_msg, type=str, default="LeakyReLU")

    help_msg = ["Learner algorithm used in the black box detector. Valid choices (case ",
                "insensitive) include:"]
    names = BlackBoxDetector.Type.names()
    for i, type_name in enumerate(names):
        if i > 0 and len(names) > 2:  # Need three options for a comma to make sense
            help_msg.append(",")
        if len(names) > 1 and i == len(names) - 1:  # And only makes sense if at least two options
            help_msg.append(" and")
        help_msg.extend([" \"", type_name, "\""])
    help_msg.append(".")
    parser.add_argument("--detector", help="".join(help_msg), type=str,
                        default=BlackBoxDetector.Type.RandomForest.name)

    help_msg = "Print the results to the console. Intended for slurm results analysis"
    parser.add_argument("--print-results", help=help_msg, action="store_true", default=False)

    args = parser.parse_args()
    # noinspection PyTypeChecker
    args.activation = _configure_activation_function(args.activation)
    args.detector = BlackBoxDetector.Type.get_from_name(args.detector)
    return args


def _configure_activation_function(act_func_name: str) -> nn.Module:
    r"""
    Parse the activation function from a string and return the corresponding activation function
    PyTorch module.  If the activation function cannot not be found, a \p ValueError is thrown.

    **Note**: Activation function check is case insensitive.

    :param act_func_name: Name of the activation function to
    :return: Activation function module associated with the passed name.
    """
    act_func_name = act_func_name.lower()  # Make case insensitive
    # Supported activation functions
    act_funcs = [("relu", nn.ReLU), ("elu", nn.ELU), ("leakyrelu", nn.LeakyReLU), ("tanh", nn.Tanh),
                 ("sigmoid", nn.Sigmoid)]
    for func_name, module in act_funcs:
        if act_func_name == func_name.lower():
            return module
    raise ValueError("Unknown activation function: \"%s\"" % act_func_name)


def load_dataset(file_path: Union[str, Path], y: int) -> MalwareDataset:
    r"""
    Extracts the input data from disk and packages them into format expected by \p MalGAN.  Supports
    loading files from numpy, torch, and pickle.  Other formats (based on the file extension) will
    result in a \p ValueError.

    :param file_path: Path to a NumPy data file containing tensors for the benign and malware
                      data.
    :param y: Y value for dataset
    :return: MalwareDataset objects for the malware and benign files respectively.
    """
    file_ext = Path(file_path).suffix
    if file_ext in {".npy", ".npz"}:
        data = np.load(file_path)
    elif file_ext in {".pt", ".pth"}:
        data = torch.load(str(file_path))
    elif file_ext == ".pk":
        with open(str(file_path), "rb") as f_in:
            data = pickle.load(f_in)
    else:
        raise ValueError("Unknown file extension.  Cannot determine how to import")
    return MalwareDataset(x=data, y=y)


def main():
    args = parse_args()
    setup_logger(args.q)

    MalGAN.MALWARE_BATCH_SIZE = args.batch_size

    if torch.cuda.is_available():
        logging.info("Torch GPU Available. Device #%d", torch.cuda.current_device())
    else:
        logging.info("No GPU detected. Running CPU only.")

    malgan = MalGAN(load_dataset(args.mal_file, MalGAN.Label.Malware.value),
                    load_dataset(args.ben_file, MalGAN.Label.Benign.value),
                    Z=args.Z,
                    h_gen=args.gen_hidden_sizes,
                    h_discrim=args.discrim_hidden_sizes,
                    g_hidden=args.activation,
                    detector_type=args.detector)
    malgan.fit_one_cycle(args.num_epoch, quiet_mode=args.q)
    results = malgan.measure_and_export_results()
    if args.print_results:
        print(results)


if __name__ == "__main__":
    main()
