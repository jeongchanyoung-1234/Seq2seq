import cupy as np


def sigmoid(x) :
    pos_indice = x >= 0
    neg_indice = x < 0

    new_x = np.zeros_like(x, float)
    new_x[pos_indice] = 1 / (1 + np.exp(-x[pos_indice]))
    new_x[neg_indice] = np.exp(x[neg_indice]) / (1 + np.exp(x[neg_indice]))

    return new_x


def softmax(x) :
    x = x - x.max(axis=1, keepdims=True)
    x = np.exp(x)
    x /= x.sum(axis=1, keepdims=True)

    return x


def clip_grads(grads, max_norm) :
    total_norm = 0
    for grad in grads :
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm)
    if rate < 1 :
        for grad in grads :
            grad *= rate


def get_norm(params, grads, norm_type=2.) :
    p_norm = 0
    for param in params :
        p_norm += (param ** norm_type).sum()
    p_norm **= (1. / norm_type)

    g_norm = 0
    for grad in grads :
        g_norm += (grad ** norm_type).sum()
    g_norm **= (1. / norm_type)

    if np.isnan(p_norm) or np.isinf(p_norm) :
        p_norm = 0.

    if np.isnan(g_norm) or np.isinf(g_norm) :
        g_norm = 0.

    return p_norm, g_norm
