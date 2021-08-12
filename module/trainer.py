import time
from itertools import combinations

import cupy as np
from functions import clip_grads, get_norm

import matplotlib.pyplot as plt
class SingleTrainer :
    def __init__(self,
                 config,
                 model,
                 optimizer) :
        self.config = config
        self.model = model
        self.optimizer = optimizer

        self.best_loss = np.inf
        self.best_epoch = None
        self.train_acc_list = []
        self.valid_acc_list = []

    def weight_tying(self, params, grads) :
        params, grads = params[:], grads[:]

        while True :
            length = len(params)
            for a, b in combinations(np.arange(length - 1), 2) :
                a, b = int(a), int(b)
                if params[a].shape == params[b].shape :
                    if params[a] is params[b] :
                        grads[a] += grads[b]
                        params.pop(b)
                        grads.pop(b)
                        break
                elif params[a].shape == params[b].T.shape :
                    if np.all(params[a] == params[b].T) :
                        grads[a] += grads[b].T
                        params.pop(b)
                        grads.pop(b)
                        break
            else :
                break

        return params, grads

    def reset_state(self) :
        self.model.reset_state()

    def train(self, x, y, valid_data=None, shuffle=True) :
        data_size = len(x)
        max_iters = data_size // self.config.batch_size
        model, optimizer = self.model, self.optimizer
        total_train_loss = 0
        loss_count = 0

        if valid_data is not None :
            valid_X, valid_y = valid_data
            total_valid_loss = 0

        start = time.time()
        stop_cnt = 0
        if self.config.early_stopping > 0 :
            print(
                '(Message) The training will be automatically stopped when best score not updated during {} epochs'.format(
                    self.config.early_stopping))
        for epoch in range(self.config.epochs) :
            stop_cnt += 1
            if stop_cnt > self.config.early_stopping > 0 :
                print('(Message) No improvement during {} epochs, training stopped'.format(self.config.early_stopping))
                return

            if shuffle :
                idx = np.random.permutation(data_size)
                x = x[idx, :]
                y = y[idx, :]

            for iters in range(max_iters) :
                batch_x = x[iters * self.config.batch_size :(iters + 1) * self.config.batch_size]
                batch_y = y[iters * self.config.batch_size :(iters + 1) * self.config.batch_size]
                train_loss = model.forward(batch_x, batch_y)
                model.backward()
                params, grads = self.weight_tying(model.params, model.grads)
                if self.config.max_grad_norm > 0 :
                    clip_grads(grads, self.config.max_grad_norm)
                optimizer.update(params, grads)

                p_norm, g_norm = get_norm(params, grads)

                total_train_loss += train_loss
                loss_count += 1
                end = time.time()

                self.reset_state()
                valid_loss = self.model.forward(valid_X, valid_y, is_train=True)
                total_valid_loss += valid_loss
                self.reset_state()

            if self.config.verbose > 0 :
                if (epoch + 1) % self.config.verbose == 0 :
                    if valid_data is None :
                        avg_train_loss = total_train_loss / loss_count

                        if avg_train_loss < self.best_loss :
                            stop_cnt = 0
                            self.best_loss = avg_train_loss
                            self.best_epoch = epoch + 1
                        else :
                            if self.config.lr_decay > 0 :
                                self.optimizer.lr = self.optimizer.lr * self.config.lr_decay

                        print(
                            '| EPOCH ({} / {}) |  train_loss={:.4f}  best_loss={:.4f}  |param|={:.2e}  |grad|={:.2e} ({:.2f}sec)'.format(
                                epoch + 1, self.config.epochs, avg_train_loss, self.best_loss, p_norm, g_norm,
                                end - start
                            ))
                        total_train_loss, loss_count = 0, 0
                    else :
                        self.reset_state()
                        batch_y_hat = self.model.forward(batch_x, batch_y, is_train=False)
                        self.reset_state()
                        valid_y_hat = self.model.forward(valid_X, valid_y, is_train=False)
                        self.reset_state()
                        score = batch_y_hat.shape[1]

                        train_accuracy = ((batch_y_hat == batch_y[:, 1 :]).sum(axis=1) == score).sum() / len(batch_y)
                        valid_accuracy = ((valid_y_hat == valid_y[:, 1 :]).sum(axis=1) == score).sum() / len(valid_y)
                        self.train_acc_list.append(train_accuracy)
                        self.valid_acc_list.append(valid_accuracy)
                        print('TRAIN ACCURACY={:.4f}% VALID ACCURACY={:.4f}%'.format(train_accuracy * 100,
                                                                                     valid_accuracy * 100))

                        avg_train_loss = total_train_loss / loss_count
                        avg_valid_loss = total_valid_loss / loss_count

                        if avg_valid_loss < self.best_loss :
                            stop_cnt = 0
                            self.best_loss = avg_valid_loss
                            self.best_epoch = epoch + 1
                        else :
                            if self.config.lr_decay > 0 :
                                self.optimizer.lr = self.optimizer.lr * self.config.lr_decay

                        print(
                            '| EPOCH ({} / {}) |  train_loss={:.4f}  valid_loss={:.4f}  best_loss={:.4f}  |param|={:.2e}  |grad|={:.2e} ({:.2f}sec)'.format(
                                epoch + 1, self.config.epochs, avg_train_loss, avg_valid_loss, self.best_loss, p_norm,
                                g_norm, end - start
                            ))
                        total_train_loss, total_valid_loss, loss_count = 0, 0, 0

    def print_result(self) :
        print()
        print('=' * 10, 'Result', '=' * 10)
        print('Best loss', self.best_loss)
        print('Best epoch', self.best_epoch)

    def plot_result(self, validation) :
        import numpy
        x = numpy.arange(len(self.train_acc_list))
        plt.plot(x, self.train_acc_list, label='Train loss')
        if validation :
            plt.plot(x, self.valid_acc_list, label='Valid loss')
        plt.title('Training result')
        plt.xlabel('Epochs'.format(self.config.verbose))
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()