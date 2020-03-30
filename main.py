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
        return idxs.view(N, T, self.topk, 1)

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
    return batch.view(N, T, kernel ** 2 * C, -1).transpose(2, 3)


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 5, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
        nn.Conv2d(out_ch, out_ch, 3, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
        nn.Conv2d(out_ch, out_ch, 3, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
    )


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.block1 = conv_block(1, 32)  # 3 conv layers
        self.block2 = conv_block(32, 32)  # double it for greater good
        self.conv1x1 = nn.Conv2d(
            32, 1, 1, bias=False
        )  # too many channels will kill you
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        N, T, C, W, H = x.shape
        x = x.view(-1, C, W, H)
        x = self.conv1x1(self.block2(self.block1(x)))
        x = self.bn(F.relu(x, inplace=True))
        _, C, W, H = x.shape
        return x.view(N, T, C, W, H)


class Glimpsy(nn.Module):
    def __init__(self, unfold, attention, rnn, linear, head=None):
        super(Glimpsy, self).__init__()
        self.head = head
        self.unfold = unfold
        self.attention = attention
        self.rnn = rnn
        self.bn = nn.BatchNorm1d(128)
        self.linear = linear

    def forward(self, x):
        if self.head is not None:
            # feature extractor
            x = self.head(x)

        # get patches
        x = self.unfold(x)

        # attention, no gradient required here
        idxs = self.attention(x.data)

        # use the idxs to reduce from n to k
        N, T, n, d_in = x.shape
        _, _, k, _ = idxs.shape
        x = x.view(N * T, n, d_in)  # collapse batch and time
        xg = x.gather(1, idxs.view(N * T, k, 1).expand(N * T, k, d_in))

        # ready for rnn
        x = xg.view(N, T, k, d_in)  # unfold batch and time
        x = x.permute(1, 0, 2, 3)  # time first
        x = x.view(T, N, -1)  # collapse the local features
        _, (hn, _) = self.rnn(x)
        hn = self.bn(hn.squeeze())
        return F.log_softmax(self.linear(hn), dim=1), xg, hn

    def get_genotype(self):
        return self.attention.get_genotype()

    def set_genotype(self, parameters):
        self.attention.set_genotype(parameters)


def model_stats(model):
    for mod_name, mod in zip(
        ["CNN", "RNN", "BN", "Classifier"],
        [model.head, model.rnn, model.bn, model.linear],
    ):
        print(f"{mod_name}: ---------")
        for name, param in mod.named_parameters():
            grad_mean = 42.424242
            if param.grad is not None:
                grad_mean = param.grad.data.mean()

            print(
                "{:16} mean={:6.5f}, var={:6.5f}. Grad: mean={:4.5f}".format(
                    name, param.data.mean(), param.data.var(), grad_mean,
                )
            )


def main():
    """ Entry point."""
    epochs = 200
    kernel_size = 7
    topk = 10
    hidden_size = 128
    num_labels = 46
    batch_size = 128
    popsize = 64
    device = torch.device("cuda")
    experiment = f"{datetime.now().strftime('%b%d_%H%M%S')}_rmsprop"
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
        head=Head(),
    )
    model.to(device)

    rlog.info(
        summary(
            model,
            torch.zeros((batch_size, 10, 1, 64, 64)).to(device),
            show_input=True,
            show_hierarchical=True,
        ),
    )

    # not your grandma's optimizer
    es_optim = cma.CMAEvolutionStrategy(
        model.get_genotype().numpy(), 0.1, {"popsize": popsize}
    )
    optim = O.Adam(model.parameters(), lr=0.001)
    # optim = O.RMSprop(
    #     model.parameters(),
    #     lr=0.00025,
    #     momentum=0.0,
    #     alpha=0.95,
    #     eps=0.01,
    #     centered=True,
    # )

    gen_cnt, step_cnt = 0, 0
    for epoch in range(epochs):
        train_log.info(f"Epoch {epoch:03d} started ----------------------")

        candidates, losses = es_optim.ask(), []
        correct = 0
        for idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            data.requires_grad_(True)

            if (idx % popsize == 0) and (idx != 0):
                es_optim.tell(candidates, losses)

                stats = {
                    "acc": 100.0 * correct / (idx * batch_size),
                    "bestFit": np.min(losses),
                    "unFit": np.max(losses),
                    "attnMean": model.get_genotype().mean().item(),
                    "attnVar": model.get_genotype().var().item(),
                    "gen": gen_cnt,
                }

                train_log.info(fmt.format(batch=idx, **stats))
                train_log.trace(step=step_cnt, **stats)

                candidates, losses = es_optim.ask(), []
                gen_cnt += 1

            model.set_genotype(candidates[idx % popsize])
            ys, xg, hn = model(data)
            loss = F.nll_loss(ys, target.squeeze())

            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.cpu().item())  # append for the CMA-ES eval

            # stats
            pred = ys.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            step_cnt += 1


if __name__ == "__main__":
    main()
