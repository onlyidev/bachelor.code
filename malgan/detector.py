# -*- coding: utf-8 -*-
r"""
    malgan.detector
    ~~~~~~~~~~~~

    Black box malware detector.

    Based on the paper: "Generating Adversarial Malware Examples for Black-Box Attacks Based on GAN"
    By Weiwei Hu and Ying Tan.

    :version: 0.1.0
    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""
from enum import Enum
from typing import Union

import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import torch
from torch import Tensor
from helpers.params import load_params

TorchOrNumpy = Union[np.ndarray, Tensor]
t_params, d_params = load_params("train", "detector")

# noinspection PyPep8Naming
class BlackBoxDetector:
    r"""
    Black box detector that intends to mimic an antivirus/anti-Malware program that detects whether
    a specific program is either malware or benign.
    """
    class Type(Enum):
        r""" Learner algorithm to be used by the black-box detector """
        DecisionTree = DecisionTreeClassifier()
        LogisticRegression = LogisticRegression(solver='lbfgs', max_iter=int(1e6))
        MultiLayerPerceptron = MLPClassifier()
        RandomForest = RandomForestClassifier(n_estimators=t_params["estimators"])
        SVM = SVC(gamma="auto")

        @classmethod
        def names(cls):
            r""" Builds the list of all enum names """
            return [c.name for c in cls]

        @classmethod
        def get_from_name(cls, name):
            r"""
            Gets the enum item from the specified name

            :param name: Name of the enum object
            :return: Enum item associated with the specified name
            """
            for c in BlackBoxDetector.Type:
                if c.name == name:
                    return c
            raise ValueError("Unknown enum \"%s\" for class \"%s\"", name, cls.name)

    def __init__(self, learner_type: 'BlackBoxDetector.Type'):
        self.type = learner_type
        # # noinspection PyCallingNonCallable
        self._model = sklearn.clone(self.type.value)
        self.training = True

    def fit(self, X: TorchOrNumpy, y: TorchOrNumpy):
        r"""
        Fits the learner.  Supports NumPy and PyTorch arrays as input.  Returns a torch tensor
        as output.

        :param X: Examples upon which to train
        :param y: Labels for the examples
        """
        if isinstance(X, Tensor):
            X = X.cpu().numpy()
        if isinstance(y, Tensor):
            y = y.cpu().numpy()
        self._model.fit(X, y)
        self.training = False

    def predict(self, X: TorchOrNumpy) -> Tensor:
        r"""
        Predict the labels for \p X

        :param X: Set of examples for which label probabilities should be predicted
        :return: Predicted value for \p X
        """
        if self.training:
            raise ValueError("Detector does not appear to be trained but trying to predict")
        if torch.cuda.is_available():
            X = X.cpu()
        if isinstance(X, Tensor):
            X = X.numpy()
        y = torch.from_numpy(self._model.predict(X)).float()
        return y.cuda() if torch.cuda.is_available() else y
