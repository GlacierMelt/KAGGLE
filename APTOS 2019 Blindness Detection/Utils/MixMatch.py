import torch
import numpy as np
import imgaug.augmenters as iaa

def get_augmenter():
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 3.0))
    ])
    def augment(images):
        return seq.augment(images.transpose(0, 2, 3, 1)).transpose(0, 2, 3, 1)
    return augment

def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(axis=1, keepdims=True)

def mixup(x1, x2, y1, y2, alpha):
    beta = np.random.beta(alpha, -alpha)
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    return x, y

def mixmatch(x, y, u, model, augment_fn, T=0.5, K=2, alpha=0.75):
    xb = augment_fn(x)
    ub = [augment_fn(u) for _ in range(K)]
    qb = sharpen(sum(map(lambda i: model(i), ub)) / K, T)
    Ux = np.concatenate(ub, axis=0)
    Uy = np.concatenate([qb for _ in range(K)], axis=0)
    indices = np.random.shuffle(np.arange(len(xb) + len(Ux)))
    Wx = np.concatenate([Ux, xb], axis=0)[indices]
    Wy = np.concatenate([qb, y], axis=0)[indices]
    X, p = mixup(xb, Wx[:len(xb)], y, Wy[:len(xb)], alpha)
    U, q = mixup(Ux, Wx[len(xb):], Uy, Wy[len(xb):], alpha)
    return X, U, p, q

class MixMatchLoss(torch.nn.Module):
    def __init__(self, lambda_u=100):
        super(MixMatchLoss, self).__init__()
        self.lambda_u = lambda_u
        self.xent = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()

    def forward(self, X, U, p, q, model):
        X_ = np.concatenate([X, U], axis=1)
        preds = model(X_)
        return self.xent(preds[:len(p)], p) + \
                        self.lambda_u * self.mse(preds[len(p):], q)
