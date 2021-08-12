import cupy as np

from module.layers import *


class Encoder :
    def __init__(self, W_emb, W_h, W_x, b_lstm, var_dropout_p=0., stateful=False, bidirectional=False) :
        self.length = None
        self.hidden_size = W_h.shape[0]
        self.layers = [Embedding(W_emb),
                       BiLSTM(W_h, W_x, b_lstm, var_dropout_p, stateful) if bidirectional \
                           else LSTM(W_h, W_x, b_lstm, var_dropout_p, stateful)]

        self.params, self.grads = [], []
        for layer in self.layers :
            self.params += layer.params
            self.grads += layer.grads
        self.rnn_layer = self.layers[1]

    def reset_state(self) :
        self.rnn_layer.reset_state()

    def forward(self, xs, is_train=True) :
        self.length = xs.shape[1]
        for layer in self.layers :
            xs = layer.forward(xs, is_train=is_train)
        return xs

    def backward(self, dh_t, dhs_enc=None) :
        if dhs_enc is None :
            dhs_enc = np.zeros((dh_t.shape[0], self.length, self.hidden_size)).astype('f')
        dhs_enc[:, -1, :] += dh_t
        for layer in reversed(self.layers) :
            dhs_enc = layer.backward(dhs_enc)
        return None


class SoftmaxNode :
    def __init__(self) :
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x) :
        self.out = softmax(x)
        return self.out

    def backward(self, dout) :
        dx = self.out * dout
        dx -= self.out * dx.sum(axis=1, keepdims=True)
        return dx


class Attention :
    def __init__(self, W=None) :
        self.params, self.grads = [], []
        if W is not None :
            self.params.append(W)
            self.grads.append(np.zeros_like(W))
        self.layers = None
        self.cache = None
        self.shape = None

    def forward(self, hs_dec, hs_enc, is_train=True) :
        batch_size = hs_dec.shape[0]
        self.shape = hs_enc.shape
        if self.params :
            W, = self.params
            hs_dec = np.matmul(hs_dec, W)

        self.layers = []
        # |hs_dec| = (bs, t, hs)
        cs = np.zeros((batch_size, hs_dec.shape[1], hs_enc.shape[-1]))
        ws = np.zeros((batch_size, hs_dec.shape[1], hs_enc.shape[1]))
        for t in range(hs_dec.shape[1]) :
            h_dec = hs_dec[:, t, :]
            w = hs_enc * np.expand_dims(h_dec, axis=1)
            # |w| = (bs, t, hs)
            softmax = SoftmaxNode()
            w = softmax.forward(w.sum(axis=2))
            self.layers.append(softmax)
            # |w| = (bs, t)
            ws[:, t, :] = w
            c = hs_enc * np.expand_dims(w, axis=-1)
            cs[:, t, :] = c.sum(axis=1)
            # |c| = (bs, hs)
        self.cache = hs_enc, hs_dec, ws, cs
        return cs

    def backward(self, dcs) :
        # |dcs| = (bs, t, hs)
        hs_enc, hs_dec, ws, cs = self.cache
        dhs_enc = 0
        dhs_dec = np.empty(dcs.shape).astype('f')

        for t in range(hs_dec.shape[1]) :
            dc = dcs[:, t, :]
            c = cs[:, t, :]
            w = ws[:, t, :]
            # |dc| = (bs, hs)
            # |dw| = (bs, t)

            dc_expanded = np.expand_dims(dc, axis=1).repeat(hs_enc.shape[1], axis=1)
            dw = (dc_expanded * hs_enc).sum(axis=2)
            # dhs_enc0 = (bs, t, hs)
            dhs_enc0 = dc_expanded * np.expand_dims(w, axis=2)

            dw = self.layers[t].backward(dw)
            # |dw| = (bs, t)
            dw_expanded = np.expand_dims(dw, axis=-1).repeat(hs_enc.shape[-1], axis=-1)
            dhs_enc1 = dw_expanded * np.expand_dims(hs_dec[:, t, :], axis=1)
            dh_dec = (hs_enc * dw_expanded).sum(axis=1)

            dhs = dhs_enc0 + dhs_enc1
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh_dec

        return dhs_dec, dhs_enc


class Decoder :
    def __init__(self, W_emb, W_h, W_x, b_lstm, W_att=None, var_dropout_p=0., stateful=False, peeky=False,
                 attention=False) :
        self.layers = [Embedding(W_emb),
                       LSTM(W_h, W_x, b_lstm, var_dropout_p, stateful)]
        if attention :
            self.layers.append(Attention(W_att))

        self.word_vec_size = W_emb.shape[1]
        self.hidden_size = W_h.shape[0]

        self.params, self.grads = [], []
        for layer in self.layers :
            self.params += layer.params
            self.grads += layer.grads
        self.rnn_layer = self.layers[1]
        self.peeky = peeky
        self.attention = attention

    def reset_state(self) :
        self.rnn_layer.reset_state()

    def forward(self, xs, hs_enc, is_train=True) :  # 임베딩벡터와 인코더 마지막 타임스탭의 은닉상태
        h_enc = hs_enc[:, -1, :]
        self.layers[1].h_t = h_enc
        embs = self.layers[0].forward(xs, is_train=is_train)

        if self.peeky :
            # embs = np.concatenate([np.expand_dims(h_enc, axis=1).repeat(embs.shape[1], axis=1), embs], axis=-1)
            embs = np.concatenate([embs, np.expand_dims(h_enc, axis=1).repeat(embs.shape[1], axis=1)], axis=-1)

        hs_dec = self.layers[1].forward(embs, is_train=is_train)

        if self.attention :
            cs = self.layers[2].forward(hs_dec, hs_enc)
            return np.concatenate([hs_dec, cs], axis=-1)

        return hs_dec

    def backward(self, dout) :
        dhs_enc = None
        if self.attention :
            dhs, dcs = dout[:, :, :(dout.shape[2] // 2)], dout[:, :, (dout.shape[2] // 2) :]
            dhs_dec, dhs_enc = self.layers[2].backward(dcs)
            dembs = self.layers[1].backward(dhs + dhs_dec)
        else :
            dembs = self.layers[1].backward(dout)

        if self.peeky :
            # dhs, dembs = dembs[:, :, :self.hidden_size], dembs[:, :, self.hidden_size:]
            dembs, dhs = dembs[:, :, :self.word_vec_size], dembs[:, :, self.word_vec_size :]

        self.layers[0].backward(dembs)
        return self.layers[1].dh_t + dhs.sum(axis=1) if self.peeky else self.layers[1].dh_t, dhs_enc


class Generator :
    def __init__(self, vocab_size, W_lin, b_lin) :
        self.vocab_size = vocab_size
        self.layers = [Affine(W_lin, b_lin),
                       Softmax(vocab_size)]

        self.params, self.grads = [], []
        for layer in self.layers :
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, hs, ys, is_train=True) :
        zs = self.layers[0].forward(hs, ys)
        if is_train :
            out = self.layers[1].forward(zs, ys)
        else :
            out = np.argmax(zs, axis=2)
        return out

    def backward(self, dout=1.) :
        for layer in reversed(self.layers) :
            dout = layer.backward(dout)
        return dout


class Seq2seq :
    def __init__(self,
                 vocab_size,
                 word_vec_size,
                 hidden_size,
                 var_dropout_p,
                 bidirectional,
                 peeky,
                 attention) :
        dec_input_shape = word_vec_size + hidden_size if peeky else word_vec_size
        lin_input_shape = hidden_size * 2 if attention else hidden_size
        # Encoder Weight
        W_emb_enc = (np.random.randn(vocab_size, word_vec_size) * (2. / np.sqrt(vocab_size + word_vec_size))).astype(
            'f')
        W_x_enc = (np.random.randn(word_vec_size, hidden_size * 4) * (
                    2. / np.sqrt(word_vec_size + hidden_size))).astype('f')
        W_h_enc = (np.random.randn(hidden_size, hidden_size * 4) * (2. / np.sqrt(hidden_size + hidden_size))).astype(
            'f')
        b_lstm_enc = np.zeros(hidden_size * 4).astype('f')

        # Decoder Weight
        W_emb_dec = (np.random.randn(vocab_size, word_vec_size) * (2. / np.sqrt(vocab_size + word_vec_size))).astype(
            'f')
        W_x_dec = (np.random.randn(dec_input_shape, hidden_size * 4) * (
                    2. / np.sqrt(dec_input_shape + hidden_size))).astype('f')
        W_h_dec = (np.random.randn(hidden_size, hidden_size * 4) * (2. / np.sqrt(hidden_size + hidden_size))).astype(
            'f')
        b_lstm_dec = np.zeros(hidden_size * 4).astype('f')
        W_att = (np.random.randn(hidden_size, hidden_size) * (2. / np.sqrt(hidden_size * 2))).astype('f')

        # Generator Weight
        W_lin = (np.random.randn(lin_input_shape, vocab_size) * (2. / np.sqrt(lin_input_shape + vocab_size))).astype(
            'f')
        b_lin = np.zeros(vocab_size).astype('f')

        self.layers = [Encoder(W_emb_enc, W_h_enc, W_x_enc, b_lstm_enc,
                               var_dropout_p, stateful=False, bidirectional=bidirectional),
                       Decoder(W_emb_dec, W_h_dec, W_x_dec, b_lstm_dec, W_att,
                               var_dropout_p, stateful=True, peeky=peeky, attention=attention),
                       Generator(vocab_size, W_lin, b_lin)]

        self.params, self.grads = [], []
        for layer in self.layers :
            self.params += layer.params
            self.grads += layer.grads

        self.hidden_size = hidden_size
        self.peeky = peeky

    def reset_state(self) :
        self.layers[0].reset_state()
        self.layers[1].reset_state()

    def forward(self, xs, ys, is_train=True) :
        xs_dec, ys_dec = ys[:, :-1], ys[:, 1 :]
        hs_enc = self.layers[0].forward(xs, is_train=is_train)
        hs_dec = self.layers[1].forward(xs_dec, hs_enc, is_train=is_train)
        out = self.layers[2].forward(hs_dec, ys_dec, is_train=is_train)
        return out

    def backward(self, dout=1.) :
        dhs = self.layers[2].backward(dout)
        dh_dec, dhs_enc = self.layers[1].backward(dhs)
        self.layers[0].backward(dh_dec, dhs_enc)
        return None