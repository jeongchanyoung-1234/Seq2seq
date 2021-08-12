import cupy as np
import cupyx

from functions import softmax, sigmoid


class Embedding :
    def __init__(self, W) :
        self.params, self.grads = [W], [np.zeros_like(W)]
        self.shape = None
        self.dim = None
        self.cache = None

    def forward(self, xs, is_train=True) :
        self.shape = xs.shape
        self.dim = xs.ndim
        W, = self.params
        if self.dim == 3 :
            # |xs| = (bs * t)
            xs = xs.flatten()
        self.cache = xs

        # |out| = (bs * t, word_vec_size)
        out = W[xs]
        if self.dim == 3 :
            out = out.reshape(self.shape[0], self.shape[1], W.shape[1])
        return out

    def backward(self, dout) :
        W, = self.params
        dW, = self.grads
        xs = self.cache

        dW[...] = 0.
        if self.dim == 3 :
            dout = dout.reshape(-1, W.shape[1])
        cupyx.scatter_add(dW, xs, dout)
        return None


class Affine :
    def __init__(self, W, b) :
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.shape = None
        self.cache = None

    def forward(self, xs, is_train=True) :
        self.shape = xs.shape
        W, b = self.params
        self.cache = xs
        xs_dim = xs.ndim

        if xs_dim == 3 :
            xs = xs.reshape(-1, xs.shape[2])
        # |x| = (bs * l, emb)
        out = np.dot(xs, W) + b
        if xs_dim == 3 :
            out = out.reshape(self.shape[0], self.shape[1], -1)
        # |out| = (bs, l, vs)
        return out

    def backward(self, dout) :
        W, b = self.params
        xs = self.cache
        xs_dim = xs.ndim
        if xs_dim == 3 :
            xs = xs.reshape(-1, self.shape[2])
            dout = dout.reshape(-1, dout.shape[2])
        db = dout.sum(axis=0)

        dxs = np.dot(dout, W.T)
        dW = np.dot(xs.T, dout)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        if xs_dim == 3 :
            dxs = dxs.reshape(*self.shape)
        return dxs

    def backward(self, dout) :
        W, b = self.params
        xs = self.cache
        xs = xs.reshape(-1, self.shape[2])

        dout = dout.reshape(-1, dout.shape[2])
        db = dout.sum(axis=0)

        dxs = np.dot(dout, W.T)
        dW = np.dot(xs.T, dout)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        dxs = dxs.reshape(*self.shape)
        return dxs


class Softmax :
    def __init__(self, vocab_size) :
        self.params, self.grads = [], []

        self.vocab_size = vocab_size
        self.shape = None
        self.cache = None

    def forward(self, scores, ys) :
        # |scores| = (bs, t, vs)
        # |ys| = (bs, t)
        self.shape = scores.shape
        scores = scores.reshape(-1, self.shape[2])
        zs = softmax(scores)
        # |zs| = (bs * t, vs)
        ys = ys.flatten()
        # |ys| = (bs * y)
        loss = -np.sum(np.log(zs[np.arange(len(ys)), ys])) / (len(ys))

        self.cache = zs, ys
        return loss

    def backward(self, dout=1.) :
        zs, ys = self.cache
        zs[np.arange(len(zs)), ys] -= 1
        zs = (zs * dout) / (self.shape[0] * self.shape[1])
        dscores = zs.reshape(self.shape)
        return dscores


class _LSTM :
    def __init__(self, W_h, W_x, b) :
        # |W_h| = (hs, hs * 4)
        # |W_x| = (ws, hs * 4)
        # |b| = (hs * 4)

        self.params = [W_h, W_x, b]
        self.grads = [np.zeros_like(p) for p in self.params]
        self.cache = None

    def forward(self, x, h_t_1, c_t_1) :
        W_h, W_x, b = self.params
        z = np.dot(h_t_1, W_h) + np.dot(x, W_x) + b  # repeat
        # |z| = (bs, hs * 4)
        hidden_size = h_t_1.shape[1]

        f = sigmoid(z[:, :hidden_size])
        g = np.tanh(z[:, hidden_size :2 * hidden_size])
        i = sigmoid(z[:, 2 * hidden_size :3 * hidden_size])
        o = sigmoid(z[:, 3 * hidden_size :])

        c_t = c_t_1 * f + g * i
        c_t_tanh = np.tanh(c_t)
        h_t = o * c_t_tanh

        self.cache = f, g, i, o, x, h_t_1, c_t_1, c_t_tanh
        return h_t, c_t

    def backward(self, dh_t, dc_t) :
        W_h, W_x, b = self.params
        f, g, i, o, x, h_t_1, c_t_1, c_t_tanh = self.cache

        do = c_t_tanh * dh_t
        do = do * o * (1. - o)

        dc_t = dc_t + (dh_t * o) * (1. - c_t_tanh ** 2)
        dc_t_1 = f * dc_t

        df = c_t_1 * dc_t
        df = df * f * (1. - f)

        dg = i * dc_t
        dg = dg * (1. - g ** 2)

        di = g * dc_t
        di = di * i * (1. - i)

        dz = np.hstack([df, dg, di, do])
        # |dz| = (bs, 4 * hs)
        db = dz.sum(axis=0)

        dh_t_1 = np.matmul(dz, W_h.T)
        dW_h = np.matmul(h_t_1.T, dz)
        dx_t = np.matmul(dz, W_x.T)
        dW_x = np.matmul(x.T, dz)

        self.grads[0][...] = dW_h
        self.grads[1][...] = dW_x
        self.grads[2][...] = db

        return dx_t, dh_t_1, dc_t_1


class LSTM :
    def __init__(self, Wh, Wx, b, var_dropout_p=0., stateful=False) :
        self.params = [Wh, Wx, b]
        self.grads = [np.zeros_like(Wh), np.zeros_like(Wx), np.zeros_like(b)]

        self.stateful = stateful
        self.hidden_size = Wh.shape[0]
        self.length = None
        self.input_size = Wx.shape[0]

        self.h_t, self.c_t = None, None
        self.dh_t_1 = None
        self.dh_t = None
        self.layers = None

        self.var_dropout_p = var_dropout_p
        self.mask = None

    def forward(self, xs, is_train=True) :
        # |xs| = (bs, t, is)
        Wh, Wx, b = self.params
        batch_size = xs.shape[0]
        self.length = xs.shape[1]

        self.layers = []
        hs = np.empty((batch_size, self.length, self.hidden_size), dtype='f')
        if not self.stateful or self.h_t is None :
            self.h_t = np.zeros((batch_size, self.hidden_size), dtype='f')
        if not self.stateful or self.c_t is None :
            self.c_t = np.zeros((batch_size, self.hidden_size), dtype='f')

        if self.var_dropout_p > 0 and is_train :
            mask = np.random.uniform(size=self.h_t.shape) > self.var_dropout_p
            self.mask = mask.astype('f')

        for t in range(self.length) :
            layer = _LSTM(*self.params)
            if self.mask is not None and is_train :
                self.h_t *= self.mask
                self.c_t *= self.mask
            self.h_t, self.c_t = layer.forward(xs[:, t, :], self.h_t, self.c_t)
            hs[:, t, :] = self.h_t
            self.layers.append(layer)
        return hs

    def backward(self, dhs) :
        Wh, Wx, b = self.params
        batch_size = dhs.shape[0]
        dxs = np.empty((batch_size, self.length, self.input_size), dtype='f')

        if self.dh_t is None : self.dh_t = 0
        dc_t = 0

        grads = [0, 0, 0]
        for t in reversed(range(self.length)) :
            layer = self.layers[t]
            if self.mask is not None :
                self.dh_t *= self.mask
                dc_t *= self.mask
            # |dhs| = (bs, t, hs)
            # |dh_t| = (bs, 1, hs)
            dx_t, self.dh_t, dc_t = layer.backward(dhs[:, t, :] + self.dh_t, dc_t)
            dxs[:, t, :] = dx_t
            for i, grad in enumerate(layer.grads) :
                grads[i] += grad

        for i, grad in enumerate(grads) :
            self.grads[i][...] = grad
        self.mask = None
        return dxs

    def reset_state(self) :
        self.h_t, self.c_t = None, None


class BiLSTM :
    def __init__(self, W_h, W_x, b, var_dropout_p=0., stateful=False) :
        self.fore = LSTM(W_h, W_x, b, var_dropout_p, stateful)
        self.back = LSTM(W_h, W_x, b, var_dropout_p, stateful)
        self.params = self.fore.params + self.back.params
        self.grads = self.fore.grads + self.back.grads

    def forward(self, x, is_train=True) :
        return (self.fore.forward(x) + self.back.forward(x[:, : :-1, :])[:, : :-1, :]) / 2.

    def backward(self, dout) :
        dout /= 2.
        return self.fore.backward(dout) + self.back.backward(dout[:, : :-1, :])[:, : :-1, :]

    def reset_state(self) :
        self.fore.reset_state()
        self.back.reset_state()