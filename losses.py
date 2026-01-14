import numpy as np

def cross_loss(y_pred, y_true):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)

    batch_size = y_pred.shape[0]
    correct_probs = y_pred[np.arange(batch_size), y_true]
    loss = -np.mean(np.log(correct_probs))

    return loss
