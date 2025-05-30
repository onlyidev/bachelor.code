# -*- coding: utf-8 -*-
r"""
    malgan.__init__
    ~~~~~~~~~~~~~~~

    MalGAN complete architecture.

    Based on the paper: "Generating Adversarial Malware Examples for Black-Box Attacks Based on GAN"
    By Weiwei Hu and Ying Tan.

    :version: 0.1.0
    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""
import logging
import os
from enum import Enum
from typing import List, Tuple, Union
from pathlib import Path

import numpy as np

import tensorboardX
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, Subset

from ._export_results import _export_results
from ._log_tools import setup_logger, TrainingLogger
from .detector import BlackBoxDetector
from .discriminator import Discriminator
from .generator import Generator

import mlflow

ListOrInt = Union[List[int], int]
PathOrStr = Union[str, Path]
TensorTuple = Tuple[Tensor, Tensor]


IS_CUDA = torch.cuda.is_available()
if IS_CUDA:
    device = torch.device('cuda:0')
    # noinspection PyUnresolvedReferences
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class MalwareDataset(Dataset):
    r"""
    Encapsulates a malware dataset.  All elements in the dataset will be either malware or benign
    """
    def __init__(self, x: Union[np.ndarray, Tensor], y):
        super().__init__()

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y

    def __len__(self):
        return self.x.shape[0]

    @property
    def num_features(self):
        r""" Number of features in the dataset """
        return self.x.shape[1]


class _DataGroup:  # pylint: disable=too-few-public-methods
    r"""
    Encapsulates either PyTorch DataLoaders or Datasets.  This class is intended only for internal
    use by MalGAN.
    """
    def __init__(self, train: MalwareDataset, valid: MalwareDataset, test: MalwareDataset):
        self.train = train
        self.valid = valid
        self.test = test
        self.is_loaders = False

    def build_loader(self, batch_size: int = 0):
        r"""
        Constructs loaders from the datasets

        :param batch_size: Batch size for training
        """
        self.train = DataLoader(self.train, batch_size=batch_size, shuffle=False, pin_memory=False)
        if self.valid:
            self.valid = DataLoader(self.valid, batch_size=batch_size, pin_memory=False)
        self.test = DataLoader(self.test, batch_size=batch_size, pin_memory=False)
        self.is_loaders = True


# noinspection PyPep8Naming
class MalGAN(nn.Module):
    r""" Malware Generative Adversarial Network based on the work of Hu & Tan. """

    MALWARE_BATCH_SIZE = 32

    SAVED_MODEL_DIR = Path("saved_models")

    VALIDATION_SPLIT = 0.2

    tensorboard = None

    class Label(Enum):
        r""" Label value assigned to malware and benign examples """
        Malware = 1
        Benign = 0

    # noinspection PyPep8Naming
    def __init__(self, mal_data: MalwareDataset, ben_data: MalwareDataset, Z: int,
                 h_gen: ListOrInt, h_discrim: ListOrInt,
                 test_split: float = 0.2,
                 g_hidden: nn.Module = nn.LeakyReLU,
                 detector_type: BlackBoxDetector.Type = BlackBoxDetector.Type.LogisticRegression):
        r"""
        Malware Generative Adversarial Network Constructor

        :param mal_data: Malware training dataset.
        :param ben_data: Benign training dataset.
        :param Z: Dimension of the noise vector \p z
        :param test_split: Fraction of input data to be used for testing
        :param h_gen: Width of the hidden layer(s) in the GENERATOR.  If only a single hidden
                      layer is desired, then this can be only an integer.
        :param h_discrim: Width of the hidden layer(s) in the DISCRIMINATOR.  If only a single
                          hidden layer is desired, then this can be only an integer.
        :param detector_type: Learning algorithm to be used by the black-box detector
        """
        super().__init__()

        if mal_data.num_features != ben_data.num_features:
            raise ValueError("Mismatch in the number of features between malware and benign data")
        if Z <= 0:
            raise ValueError("Z must be a positive integers")
        if test_split <= 0. or test_split >= 1.:
            raise ValueError("test_split must be in the range (0,1)")
        self._M, self._Z = mal_data.num_features, Z  # pylint: disable=invalid-name

        # Format the hidden layer sizes and make sure all are valid values
        if isinstance(h_gen, int):
            h_gen = [h_gen]
        if isinstance(h_discrim, int):
            h_discrim = [h_discrim]
        self.d_discrim, self.d_gen = h_discrim, h_gen
        for h_size in [self.d_discrim, self.d_gen]:
            for w in h_size:
                if w <= 0:
                    raise ValueError("All hidden layer widths must be positive integers.")

        if not isinstance(g_hidden, nn.Module):
            g_hidden = g_hidden()
        self._g = g_hidden

        self._is_cuda = IS_CUDA

        logging.info("Constructing new MalGAN")
        logging.info("Malware Dimension (M): %d", self.M)
        logging.info("Latent Dimension (Z): %d", self.Z)
        logging.info("Test Split Ratio: %.3f", test_split)
        logging.info("Generator Hidden Layer Sizes: %s", h_gen)
        logging.info("Discriminator Hidden Layer Sizes: %s", h_discrim)
        logging.info("Blackbox Detector Type: %s", detector_type.name)
        logging.info("Activation Type: %s", self._g.__class__.__name__)

        self._bb = BlackBoxDetector(detector_type)
        self._gen = Generator(M=self.M, Z=self.Z, hidden_size=h_gen, g=self._g)
        self._discrim = Discriminator(M=self.M, hidden_size=h_discrim, g=self._g)

        def split_train_valid_test(dataset: Dataset, is_benign: bool):
            """Helper function to partition into test, train, and validation subsets"""
            valid_len = 0 if is_benign else int(MalGAN.VALIDATION_SPLIT * len(dataset))
            test_len = int(test_split * len(dataset))

            # Order must be train, validation, test
            lengths = [len(dataset) - valid_len - test_len, valid_len, test_len]
            return _DataGroup(*torch.utils.data.random_split(dataset, lengths,generator=torch.Generator(device="cuda")))

        # Split between train, test, and validation then construct the loaders
        self._mal_data = split_train_valid_test(mal_data, is_benign=False)
        self._ben_data = split_train_valid_test(ben_data, is_benign=True)
        # noinspection PyTypeChecker
        self._fit_blackbox(self._mal_data.train, self._ben_data.train)

        self._mal_data.build_loader(MalGAN.MALWARE_BATCH_SIZE)
        ben_bs_frac = len(ben_data) / len(mal_data)
        self._ben_data.build_loader(int(ben_bs_frac * MalGAN.MALWARE_BATCH_SIZE))
        # Set CUDA last to ensure all parameters defined
        if self._is_cuda: self.cuda()

    @property
    def M(self) -> int:
        r"""Width of the malware feature vector"""
        return self._M

    @property
    def Z(self) -> int:
        r"""Width of the generator latent noise vector"""
        return self._Z

    def _fit_blackbox(self, mal_train: Subset, ben_train: Subset) -> None:
        r"""
        Firsts the blackbox detector using the specified malware and benign training sets.

        :param mal_train: Malware training dataset
        :param ben_train: Benign training dataset
        """
        def extract_x(ds: Subset) -> Tensor:
            # noinspection PyUnresolvedReferences
            x = ds.dataset.x[ds.indices]
            return x.cpu() if self._is_cuda else x

        mal_x = extract_x(mal_train)
        ben_x = extract_x(ben_train)
        merged_x = torch.cat((mal_x, ben_x))

        merged_y = torch.cat((torch.full((len(mal_train),), MalGAN.Label.Malware.value),
                              torch.full((len(ben_train),), MalGAN.Label.Benign.value)))
        logging.info("Starting training of blackbox detector of type \"%s\"", self._bb.type.name)
        self._bb.fit(merged_x, merged_y)
        logging.info("COMPLETED training of blackbox detector of type \"%s\"", self._bb.type.name)

    def fit(self, cyc_len: int, quiet_mode: bool = False) -> None:
        r"""
        Trains the model for the specified number of epochs.  The epoch with the best validation
        loss is used as the final model.

        :param cyc_len: Number of cycles (epochs) to train the model.
        :param quiet_mode: True if no printing to console should occur in this function
        """
        if cyc_len <= 0:
            raise ValueError("At least a single training cycle is required.")

        MalGAN.tensorboard = tensorboardX.SummaryWriter()

        d_optimizer = optim.Adam(self._discrim.parameters(), lr=1e-4)
        g_optimizer = optim.Adam(self._gen.parameters(), lr=1e-3, betas=(0.5, 0.999))
        
        # mlflow.log_param("Discriminator_Learning_Rate", 1e-6)
        # mlflow.log_param("Generator_Learning_Rate", 2e-4)

        if not quiet_mode:
            names = ["Gen Train Loss", "Gen Valid Loss", "Discrim Train Loss", "Best?"]
            log = TrainingLogger(names, [20, 20, 20, 7])

        best_epoch, best_loss = None, np.inf
        for epoch_cnt in range(1, cyc_len + 1):
            train_l_g, train_l_d = self._fit_epoch(g_optimizer, d_optimizer)
            for block, loss in [("Generator", train_l_g), ("Discriminator", train_l_d)]:
                MalGAN.tensorboard.add_scalar('Train_%s_Loss' % block, loss, epoch_cnt)
                mlflow.log_metric('Train_%s_Loss' % block, loss, step=epoch_cnt)

            # noinspection PyTypeChecker
            valid_l_g = self._meas_loader_gen_loss(self._mal_data.valid)
            MalGAN.tensorboard.add_scalar('Validation_Generator_Loss', valid_l_g, epoch_cnt)
            mlflow.log_metric('Validation_Generator_Loss', valid_l_g, step=epoch_cnt)
            flds = [train_l_g, valid_l_g, train_l_d, valid_l_g < best_loss]
            if flds[-1]:
                self._save(self._build_export_name(is_final=False))
                best_loss = valid_l_g
            if not quiet_mode: log.log(epoch_cnt, flds)
        MalGAN.tensorboard.close()

        self.load(self._build_export_name(is_final=False))
        self._save(self._build_export_name(is_final=True))
        self._delete_old_backup(is_final=False)

    def _build_export_name(self, is_final: bool = True) -> str:
        r"""
        Builds the name that will be used when exporting the model.

        :param is_final: If \p True, then file name is for final (i.e., not training) model
        :return: Model name built from the model's parameters
        """
        name = ["malgan", "z=%d" % self.Z,
                "d-gen=%s" % str(self.d_gen).replace(" ", "_"),
                "d-disc=%s" % str(self.d_discrim).replace(" ", "_"),
                "bs=%d" % MalGAN.MALWARE_BATCH_SIZE,
                "bb=%s" % self._bb.type.name, "g=%s" % self._g.__class__.__name__,
                "final" if is_final else "tmp"]

        # Either add an epoch name or
        return MalGAN.SAVED_MODEL_DIR / "".join(["_".join(name).lower(), ".pth"])

    def _delete_old_backup(self,  is_final: bool = True) -> None:
        r"""
        Helper function to delete old backed up models

        :param is_final: If \p True, then file name is for final (i.e., not training) model
        """
        backup_name = self._build_export_name(is_final)
        try:
            os.remove(backup_name)
        except OSError:
            logging.warning("Error trying to delete model: %s", backup_name)

    def _fit_epoch(self, g_optim: Optimizer, d_optim: Optimizer) -> TensorTuple:
        r"""
        Trains a single entire epoch

        :param g_optim: Generator optimizer
        :param d_optim: Discriminator optimizer
        :return: Average training loss
        """
        tot_l_g = tot_l_d = 0
        num_batch = min(len(self._mal_data.train), len(self._ben_data.train))

        for (m, _), (b, _) in zip(self._mal_data.train, self._ben_data.train):
            if self._is_cuda: m, b = m.cuda(), b.cuda()
            m_prime, g_theta = self._gen.forward(m)
            l_g = self._calc_gen_loss(g_theta)
            g_optim.zero_grad()
            l_g.backward()
            # torch.nn.utils.clip_grad_value_(l_g, 1)
            g_optim.step()
            tot_l_g += l_g

            # Update the discriminator
            for x in [m_prime, b]:
                l_d = self._calc_discrim_loss(x)
                d_optim.zero_grad()
                l_d.backward()
                # torch.nn.utils.clip_grad_value_(l_d, 1)
                d_optim.step()
                tot_l_d += l_d
        # noinspection PyUnresolvedReferences
        return (tot_l_g / num_batch).item(), (tot_l_d / num_batch).item()

    def _meas_loader_gen_loss(self, loader: DataLoader) -> float:
        r""" Calculate the generator loss on malware dataset """
        loss = 0
        for m, _ in loader:
            if self._is_cuda: m = m.cuda()
            _, g_theta = self._gen.forward(m)
            loss += self._calc_gen_loss(g_theta)
        # noinspection PyUnresolvedReferences
        return (loss / len(loader)).item()

    def _calc_gen_loss(self, g_theta: Tensor) -> Tensor:
        r"""
        Calculates the parameter :math:`L_{G}` as defined in Eq. (3) of Hu & Tan's paper.

        :param g_theta: :math:`G(_{\theta_g}(m,z)` in Eq. (1) of Hu & Tan's paper
        :return: Loss for the generator smoothed output.
        """
        d_theta = self._discrim.forward(g_theta)
        return d_theta.log().mean()

    def _calc_discrim_loss(self, X: Tensor) -> Tensor:
        r"""
        Calculates the parameter :math:`L_{D}` as defined in Eq. (2) of Hu & Tan's paper.

        :param X: Examples to calculate the loss over.  May be a mix of benign and malware samples.
        """
        d_theta = self._discrim.forward(X)

        y_hat = self._bb.predict(X)
        d = torch.where(y_hat == MalGAN.Label.Malware.value, d_theta, 1 - d_theta)
        return -d.log().mean()

    def measure_and_export_results(self) -> str:
        r"""
        Measure the test accuracy and provide results information

        :return: Results information as a comma separated string
        """ 
        prev = self._gen.training
        self._gen.training = True
        # noinspection PyTypeChecker
        valid_loss = self._meas_loader_gen_loss(self._mal_data.valid)
        # noinspection PyTypeChecker
        test_loss = self._meas_loader_gen_loss(self._mal_data.test)
        logging.info("Final Validation Loss: %.6f", valid_loss)
        logging.info("Final Test Loss: %.6f", test_loss)

        num_mal_test = 0
        y_mal_orig, m_prime_arr, bits_changed = [], [], []
        for m, _ in self._mal_data.test:
            y_mal_orig.append(self._bb.predict(m.cpu()))
            if self._is_cuda:
                m = m.cuda()
            num_mal_test += m.shape[0]

            m_prime, _ = self._gen.forward(m)
            m_prime_arr.append(m_prime.cpu() if self._is_cuda else m_prime)

            m_diff = m_prime - m
            bits_changed.append(torch.sum(m_diff.cpu(), dim=1))

            # Sanity check no bits flipped 1 -> 0
            msg = "Malware signature changed to 0 which is not allowed"
            assert torch.sum(m_diff < -0.1) == 0, msg
        avg_changed_bits = torch.cat(bits_changed).mean()
        logging.info("Avg. Malware Bits Changed Changed: %2f", avg_changed_bits)

        # BB prediction of the malware before the generator
        y_mal_orig = torch.cat(y_mal_orig)

        # Build an X tensor for prediction using the detector
        ben_test_arr = [x.cpu() if self._is_cuda else x for x, _ in self._ben_data.test]
        x = torch.cat(m_prime_arr + ben_test_arr)
        y_actual = torch.cat((torch.full((num_mal_test,), MalGAN.Label.Malware.value),
                             torch.full((len(x) - num_mal_test,), MalGAN.Label.Benign.value)))

        y_hat_post = self._bb.predict(x)
        if self._is_cuda:
            y_mal_orig, y_hat_post, y_actual = y_mal_orig.cpu(), y_hat_post.cpu(), y_actual.cpu()
        # noinspection PyProtectedMember
        y_prob = self._bb._model.predict_proba(x)  # pylint: disable=protected-access
        y_prob = y_prob[:, MalGAN.Label.Malware.value]
        self._gen.training = prev
        return _export_results(self, valid_loss, test_loss, avg_changed_bits, y_actual,
                               y_mal_orig, y_prob, y_hat_post)

    def _save(self, file_path: PathOrStr) -> None:
        r"""
        Export the specified model to disk.  The function creates any files needed on the path.
        All exported models will be relative to \p EXPORT_DIR class object.

        :param file_path: Path to export the model.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(file_path))

    def forward(self, x: Tensor) -> TensorTuple:  # pylint: disable=arguments-differ
        r"""
        Passes a malware tensor and augments it to make it more undetectable by

        :param x: Malware binary tensor
        :return: :math:`m'` and :math:`g_{\theta}` respectively
        """
        return self._gen.forward(x)

    def load(self, filename: PathOrStr) -> None:
        r"""
        Load a MalGAN object from disk.  MalGAN's \p EXPORT_DIR is prepended to the specified
        filename.

        :param filename: Path to the exported torch file
        """
        if isinstance(filename, Path):
            filename = str(filename)
        self.load_state_dict(torch.load(filename))
        self.eval()
        # Based on the recommendation of Soumith Chantala et al. in GAN Hacks that enabling dropout
        # in evaluation improves performance. Source code based on:
        # https://discuss.pytorch.org/t/using-dropout-in-evaluation-mode/27721
        for m in self._gen.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    @staticmethod
    def _print_memory_usage() -> None:
        """
        Helper function to print the allocated tensor memory.  This is used to debug out of memory
        GPU errors.
        """
        import gc
        import operator as op
        from functools import reduce
        for obj in gc.get_objects():
            # noinspection PyBroadException
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    if len(obj.size()) > 0:  # pylint: disable=len-as-condition
                        obj_tot_size = reduce(op.mul, obj.size())
                    else:
                        obj_tot_size = "NA"
                    print(obj_tot_size, type(obj), obj.size())
            except:  # pylint: disable=bare-except  # NOQA E722
                pass
