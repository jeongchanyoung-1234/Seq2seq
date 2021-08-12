import argparse

import cupy as np

from data import sequence
from seq2seq import Seq2seq
from module.optimizer import *
from module.trainer import SingleTrainer


def define_argparser():
  p = argparse.ArgumentParser()

  p.add_argument('--hidden_size', type=int, default=256)
  p.add_argument('--word_vec_size', type=int, default=16)
  p.add_argument('--batch_size', type=int, default=128)
  p.add_argument('--valid_batch_size', type=int, default=64)
  p.add_argument('--optimizer', type=str, default='adam')
  p.add_argument('--lr', type=float, default=20)
  p.add_argument('--epochs', type=int, default=10)
  p.add_argument('--max_grad_norm', type=float, default=5)
  p.add_argument('--var_dropout', type=float, default=0)
  p.add_argument('--lr_decay', type=float, default=0.25)
  p.add_argument('--warmup_epoch', type=int, default=0)
  p.add_argument('--verbose', type=int, default=1)
  p.add_argument('--early_stopping', type=int, default=0)
  p.add_argument('--train_size', type=int, default=0)

  p.add_argument('--validation', action='store_true')
  p.add_argument('--bidirectional', action='store_true')
  p.add_argument('--peeky', action='store_true')
  p.add_argument('--attention', action='store_true')

  p.add_argument('--data', type=str, default='date.txt')

  config = p.parse_args()
  return config


def main(config):
    valid_data = None

    optimizers = {'sgd': SGD, 'adam': Adam}
    (train_X, train_y), (valid_X, valid_y) = sequence.load_data(config.data)
    train_X, train_y = np.array(train_X[:, ::-1] if config.reverse_input else train_X), np.array(train_y)
    valid_X, valid_y = np.array(valid_X[:, ::-1] if config.reverse_input else valid_X), np.array(valid_y)
    char_to_id, id_to_char = sequence.get_vocab()
    vocab_size = len(char_to_id)
    print('CORPUS SIZE', len(train_y))
    print('VOCAB SIZE', vocab_size)

    if config.train_size > 0:
        train_X, train_y = train_X[:config.train_size], train_y[:config.train_size]

    if config.validation: valid_data = valid_X, valid_y
    model = Seq2seq(vocab_size, 
                   config.word_vec_size, 
                   config.hidden_size,
                   config.var_dropout,
                   config.bidirectional,
                   config.peeky,
                   config.attention)

    optimizer = optimizers[config.optimizer.lower()](lr=config.lr)

    trainer = SingleTrainer(config, model, optimizer)
    trainer.train(train_X, train_y, valid_data=valid_data)

    trainer.plot_result(config.validation)

if __name__ == '__main__':
    config = define_argparser()
    main(config)