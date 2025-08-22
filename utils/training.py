import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from config import BATCH_SIZE, EPOCHS, N_WARMUP_EPOCHS, KL_WEIGHT, KL_MAX, OPTIMIZER
from .bayesian_model import nll_loss


class KLWarmUp(tf.keras.callbacks.Callback):
    """
    Gradually increases the KL divergence weight during training.
    This prevents the KL term from dominating too early, leading to more stable training.
    """

    def __init__(self, kl_var, kl_max, n_warmup_epochs=N_WARMUP_EPOCHS):
        super().__init__()
        self.kl_var = kl_var
        self.kl_max = kl_max
        self.n_warmup_epochs = n_warmup_epochs
        self.history = []

    def on_epoch_begin(self, epoch, logs=None):
        if self.kl_var is None:
            raise ValueError("kl_var must be provided to KLWarmUp callback.")

        ramp = np.log1p(epoch) / np.log1p(self.n_warmup_epochs)
        ramp = min(1.0, ramp)

        new_kl = self.kl_max * ramp

        # print(f"[Epoch {epoch}] KL weight: {new_kl:.6f}")
        self.kl_var.assign(new_kl)
        self.history.append(new_kl)


class EarlyStoppingAfter(tf.keras.callbacks.Callback):
    """
    EarlyStopping callback that activates only after the warm-up phase.
    Monitors validation loss and stops training if no improvement is seen for 80 consecutive epochs.
    """

    def __init__(self, start_epoch=N_WARMUP_EPOCHS, patience=80, restore_best_weights=True):
        super().__init__()
        self.start_epoch = start_epoch
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.wait = 0
        self.best = np.Inf
        self.stopped_epoch = 0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        if epoch < self.start_epoch:
            return

        if current < self.best:
            self.best = current
            self.wait = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    print(
                        f"Restoring best weights from epoch {epoch - self.patience}")
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"\nEarly stopping at epoch {self.stopped_epoch}")
            print(f"Best model was at epoch {self.best_epoch}")


def train_bayesian_model(model, cov_train, phi_train, y_train, cov_val, phi_val, y_val, plot_kl_warmup=False):
    """
    Trains a Bayesian neural network using training and validation data.
    Applies KL warm-up and early stopping to improve training stability.

    Returns the training history, which can be used to identify the best epoch.
    """

    warmup_callback = KLWarmUp(
        kl_var=KL_WEIGHT, kl_max=KL_MAX, n_warmup_epochs=N_WARMUP_EPOCHS)
    early_stopping = EarlyStoppingAfter(
        start_epoch=N_WARMUP_EPOCHS, patience=100, restore_best_weights=True)

    model.compile(optimizer=OPTIMIZER, loss=nll_loss())
    history = model.fit([cov_train, phi_train], y_train,
                        validation_data=([cov_val, phi_val], y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[warmup_callback, early_stopping], verbose=0)

    if plot_kl_warmup:
        plt.figure(figsize=(6, 4))
        plt.plot(warmup_callback.history, label="KL Weight")
        plt.title("KL Warm-up Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("KL Weight")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return history


def get_optimal_epoch(history, skip_first_n=N_WARMUP_EPOCHS):
    """
    Returns the epoch with the lowest validation loss after the KL warm-up period.
    """
    val_loss = history.history['val_loss']
    min_epoch = np.argmin(val_loss[skip_first_n:]) + skip_first_n
    return min_epoch
