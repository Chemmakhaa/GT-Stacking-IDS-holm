"""Conditional Wasserstein GAN with gradient penalty (adversarial minimax game).

The generator and the critic play a zero-sum game. The GAN is trained on the
training data only and generates synthetic samples for the minority classes.
Synthetic samples are never added to validation or test data.
Hyperparameters follow Table 1 of the paper.
"""
import numpy as np
import torch
import torch.nn as nn


def _mlp(sizes, out_dim, batchnorm, out_act=None):
    layers, d = [], sizes[0]
    for h in sizes[1:]:
        layers.append(nn.Linear(d, h))
        if batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.LeakyReLU(0.2))
        d = h
    layers.append(nn.Linear(d, out_dim))
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class CWGANGP:
    def __init__(self, latent_dim=100, learning_rate=1e-4, generator_hidden=(128, 256),
                 critic_hidden=(512, 256, 128), batch_size=256, epochs=300, n_critic=5,
                 gp_lambda=10, balance_ratio=1.0, seed=46, device=None):
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.g_hidden = list(generator_hidden)
        self.c_hidden = list(critic_hidden)
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_critic = n_critic
        self.gp_lambda = gp_lambda
        self.balance_ratio = balance_ratio
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # data are mapped to [-1, 1] because the generator ends with Tanh
    def _to_unit(self, X):
        return 2 * (X - self.min_) / self.range_ - 1

    def _from_unit(self, Z):
        return (Z + 1) / 2 * self.range_ + self.min_

    def _gradient_penalty(self, real, fake, cond):
        eps = torch.rand(real.size(0), 1, device=self.device)
        mixed = (eps * real + (1 - eps) * fake).requires_grad_(True)
        score = self.C(torch.cat([mixed, cond], 1))
        grad = torch.autograd.grad(score.sum(), mixed, create_graph=True)[0]
        return ((grad.norm(2, dim=1) - 1) ** 2).mean()

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.min_ = X.min(axis=0)
        self.range_ = np.where(X.max(axis=0) - self.min_ == 0, 1.0, X.max(axis=0) - self.min_)

        cond_dim = self.n_classes_
        self.G = _mlp([self.latent_dim + cond_dim] + self.g_hidden, self.n_features_, True, nn.Tanh()).to(self.device)
        self.C = _mlp([self.n_features_ + cond_dim] + self.c_hidden, 1, False).to(self.device)
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        opt_c = torch.optim.Adam(self.C.parameters(), lr=self.lr, betas=(0.5, 0.9))

        Xt = torch.tensor(self._to_unit(X), device=self.device)
        yi = torch.tensor(np.searchsorted(self.classes_, y), device=self.device)
        onehot = torch.nn.functional.one_hot(yi, cond_dim).float()
        n = len(Xt)
        bs = min(self.batch_size, n)
        self.history_ = []
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for step, start in enumerate(range(0, n - bs + 1, bs)):
                idx = perm[start:start + bs]
                real, cond = Xt[idx], onehot[idx]
                z = torch.randn(len(idx), self.latent_dim, device=self.device)
                fake = self.G(torch.cat([z, cond], 1)).detach()
                loss_c = (self.C(torch.cat([fake, cond], 1)).mean()
                          - self.C(torch.cat([real, cond], 1)).mean()
                          + self.gp_lambda * self._gradient_penalty(real, fake, cond))
                opt_c.zero_grad()
                loss_c.backward()
                opt_c.step()
                if step % self.n_critic == 0:
                    z = torch.randn(len(idx), self.latent_dim, device=self.device)
                    loss_g = -self.C(torch.cat([self.G(torch.cat([z, cond], 1)), cond], 1)).mean()
                    opt_g.zero_grad()
                    loss_g.backward()
                    opt_g.step()
            self.history_.append(float(loss_c.detach().cpu()))
        return self

    @torch.no_grad()
    def sample(self, label, n):
        self.G.eval()
        k = int(np.searchsorted(self.classes_, label))
        cond = torch.zeros(n, self.n_classes_, device=self.device)
        cond[:, k] = 1
        z = torch.randn(n, self.latent_dim, device=self.device)
        out = self.G(torch.cat([z, cond], 1)).cpu().numpy()
        self.G.train()
        return self._from_unit(out).astype(np.float32)

    def balance(self, X, y):
        """Add synthetic samples to each minority class. Returns X_aug, y_aug and a count table."""
        labels, counts = np.unique(y, return_counts=True)
        target = int(self.balance_ratio * counts.max())
        X_parts, y_parts, report = [X], [y], []
        for lab, cnt in zip(labels, counts):
            n_new = max(0, target - cnt)
            if n_new > 0:
                X_parts.append(self.sample(lab, n_new))
                y_parts.append(np.full(n_new, lab, dtype=y.dtype))
            report.append({"class": lab, "real": int(cnt), "synthetic": int(n_new), "total": int(cnt + n_new)})
        return np.vstack(X_parts), np.concatenate(y_parts), report
