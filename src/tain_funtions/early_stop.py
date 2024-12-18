import numpy as np


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.test_loss_min = np.inf
        self.delta = delta

    def __call__(self, test_loss):
        score = -test_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(test_loss)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(test_loss)
            self.counter = 0

    def save_checkpoint(self, test_loss):
        if self.verbose:
            print(f'Test loss decreased ({self.test_loss_min:.6f} --> {test_loss:.6f}).  Saving model ...')

        self.test_loss_min = test_loss