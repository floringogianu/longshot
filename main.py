""" Entry point.

    Inspired by https://attentionagent.github.io/
"""
import os
from datetime import datetime
from functools import partial

import cma
import numpy as np
import rlog
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
from pytorch_model_summary import summary
from torch.utils.data import DataLoader

from src.datasets import SyncedMNIST


class SparseAttention(nn.Module):
    """ The attention module computing (XWk)(XWq)^t."""

    def __init__(self, d_in, d=4, topk=12, device=None):
        """
        Args:
            nn (nn.Module): torch.nn base class.
            d_in (int): Dimension of the inputs.
            d (int, optional): Dimension of the projection. Defaults to 4.
        """
        super().__init__()
        self.wk = torch.randn(d_in, d).to(device)
        self.wq = torch.randn(d_in, d).to(device)
        self.topk = topk
        self.scale = 1 / torch.sqrt(torch.tensor(d_in).float()).to(device)

        # move the parameters in a flat tensor
        self._flat_params = torch.tensor([], device=device)
        for w in [self.wk, self.wq]:
            self._flat_params = torch.cat([self._flat_params, w.flatten()])

        # and now we change the reference
        self.wk = self._flat_params[: self.wk.numel()].view(self.wk.shape)
        self.wq = self._flat_params[self.wk.numel() :].view(self.wq.shape)

    @torch.no_grad()
    def forward(self, x):
        """        
        Args:
            x (torch.tensor): Batch of videos of size N, T, n, d_in.
        
        Returns:
            torch.tensor: Batch of local features of size N, T, topk, d_in.
        """
        N, T, n, d_in = x.shape
        x = x.view(-1, n, d_in)  # stack the time and batch together
        wk, wq = self.wk, self.wq

        h = torch.bmm(torch.matmul(x, wk), torch.matmul(x, wq).transpose(1, 2))
        A = torch.softmax(self.scale * h, dim=-1).sum(-2)
        idxs = torch.argsort(A, descending=True)[:, : self.topk]
        return x.gather(
            1, idxs.view(N * T, self.topk, 1).expand(N * T, self.topk, d_in)
        ).view(N, T, self.topk, d_in)

    def get_genotype(self):
        """ Return parameters to be optimized by ES. """
        return self._flat_params.cpu()

    def set_genotype(self, params):
        """ Set parameters that are optimized by ES. """
        self._flat_params.copy_(torch.from_numpy(params))


def unfold(batch, kernel=7, stride=4):
    """[summary]Extract patches (sliding window) from a batch of videos.
   
    Args:
        batch (torch.tensor): Batch of videos of size N, T, C, W, H
    """
    N, T, C, W, H = batch.shape
    batch = F.unfold(batch.view(-1, C, W, H), kernel_size=kernel, stride=stride)
    return batch.view(N, T, kernel ** 2 * 1, -1).transpose(2, 3)


class Glimpsy(nn.Module):
    def __init__(self, unfold, attention, rnn, linear):
        super().__init__()
        self.unfold = unfold
        self.attention = attention
        self.rnn = rnn
        self.linear = linear
        self._flat_params = None

    def forward(self, input):
        x = self.unfold(input)
        x = self.attention(x)

        # ready for rnn
        N, T, _, _ = x.shape
        x = x.permute(1, 0, 2, 3)  # time first
        x = x.view(T, N, -1)  # collapse the local features
        _, (hn, _) = self.rnn(x)
        return F.log_softmax(self.linear(hn.squeeze()), dim=1)

    def get_genotype(self):
        return self.attention.get_genotype()

    def set_genotype(self, parameters):
        self.attention.set_genotype(parameters)


def main():
    """ Entry point."""
    epochs = 100
    kernel_size = 7
    topk = 10
    hidden_size = 64
    num_labels = 46
    batch_size = 128
    popsize = 64
    device = torch.device("cuda")
    experiment = f"{datetime.now().strftime('%b%d_%H%M%S')}_dev"
    path = f"./results/{experiment}"

    os.makedirs(path)
    rlog.init("pff", path=path, tensorboard=True)
    train_log = rlog.getLogger("pff.train")
    fmt = (
        "[{gen:03d}/{batch:04d}] acc={acc:2.2f}% | bestFit={bestFit:2.3f}"
        + ", unFit={unFit:2.3f} [μ={attnMean:2.3f}/σ={attnVar:2.3f}]"
    )

    # enough with the mess
    dset = SyncedMNIST(root="./data/SyncedMNIST")
    loader = DataLoader(dset, batch_size=batch_size)

    model = Glimpsy(
        partial(unfold, kernel=kernel_size, stride=4),
        SparseAttention(kernel_size ** 2, topk=topk, device=device),
        nn.LSTM(kernel_size ** 2 * topk, hidden_size=hidden_size, bias=False),
        nn.Linear(hidden_size, num_labels),
    )
    model.to(device)

    rlog.info(
        summary(
            model,
            torch.zeros((batch_size, 10, 1, 64, 64)).to(device),
            show_input=True,
            show_hierarchical=True,
        )
    )

    # not your grandma's optimizer
    es_optim = cma.CMAEvolutionStrategy(
        model.get_genotype().numpy(), 0.1, {"popsize": popsize}
    )
    optim = O.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        train_log.info(f"Epoch {epoch:03d} started ----------------------")

        candidates, losses = es_optim.ask(), []
        correct = 0
        for idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            if (idx % popsize == 0) and (idx != 0):
                es_optim.tell(candidates, losses)

                stats = {
                    "acc": 100.0 * correct / (idx * batch_size),
                    "bestFit": np.min(losses),
                    "unFit": np.max(losses),
                    "attnMean": model.get_genotype().mean().item(),
                    "attnVar": model.get_genotype().var().item(),
                    "gen": (idx // popsize),
                }

                train_log.info(fmt.format(batch=idx, **stats))
                train_log.trace(step=idx, **stats)

                candidates, losses = es_optim.ask(), []

            model.set_genotype(candidates[idx % popsize])
            ys = model(data)
            loss = F.nll_loss(ys, target.squeeze())

            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.cpu().item())  # append for the CMA-ES eval

            # stats
            pred = ys.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()


if __name__ == "__main__":
    main()
