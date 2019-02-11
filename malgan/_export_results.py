import datetime
from pathlib import Path
from typing import Union

import numpy as np

import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

TensorOrFloat = Union[torch.Tensor, float]


# noinspection PyProtectedMember,PyUnresolvedReferences
def _export_results(model: 'MalGAN', valid_loss: TensorOrFloat, test_loss: TensorOrFloat,
                    avg_num_bits_changed: TensorOrFloat, y: np.ndarray,
                    y_prob: Union[np.ndarray, torch.Tensor], y_hat: np.ndarray) -> str:
    r"""
    Exports MalGAN results.

    :param model: MalGAN model
    :param valid_loss: Average loss on the malware validation set
    :param test_loss: Average loss on the malware test set
    :param avg_num_bits_changed:
    :param y: Actual labels
    :param y_prob: Probability of malware
    :param y_hat: Predict labels
    :return: Results string
    """
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.numpy()

    results_file = Path("results.csv")
    exists = results_file.exists()
    with open(results_file, "a+") as f_out:
        header = ",".join(["time_completed,M,Z,batch_size,test_set_size,detector_type,activation",
                           "gen_hidden_dim,discim_hidden_dim",
                           "avg_validation_loss,avg_test_loss,avg_num_bits_changed",
                           "auc,tpr,fpr,fnr,tnr"])
        if not exists:
            f_out.write(header)

        results = ["\n%s" % datetime.datetime.now(),
                   "%d,%d,%d" % (model.M, model.Z, model.__class__.MALWARE_BATCH_SIZE),
                   "%d,%s,%s" % (len(y), model._bb.type.name, model._g.__class__.__name__),
                   "%s,%s" % (str(model.d_gen), str(model.d_discrim)),
                   "%.15f,%.15f,%.3f" % (valid_loss, test_loss, avg_num_bits_changed)]

        auc = roc_auc_score(y, y_prob)
        results.append("%.6f" % auc)

        # Write the TxR and NxR information
        tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
        tnr, tpr = tn / (tn + fp), tp / (tp + fn)
        for rate in [tpr, 1 - tnr, 1 - tpr, tnr]:
            results.append("%.6f" % rate)
        results = ",".join(results)
        f_out.write(results)

        return "".join([header, results])
