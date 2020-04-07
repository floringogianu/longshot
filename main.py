""" Entry point.

    Inspired by https://attentionagent.github.io/
"""
import os
import argparse
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

    def __init__(self, d_in, d=4, topk=12):
        """
        Args:
            nn (nn.Module): torch.nn base class.
            d_in (int): Dimension of the inputs.
            d (int, optional): Dimension of the projection. Defaults to 4.
        """
        super().__init__()
        # the attention params are optimized without gradient
        self.flat = nn.Parameter(torch.randn(2 * d_in * d), requires_grad=False)
        self.dims = (d_in, d)
        self.topk = topk
        self.scale = 1 / torch.sqrt(torch.tensor(d_in).float()).item()

    @torch.no_grad()
    def forward(self, x):  # pylint: disable=arguments-differ
        """
        Args:
            x (torch.tensor): Batch of videos of size N, T, n, d_in.

        Returns:
            torch.tensor: Batch of local features of size N, T, topk, d_in.
        """
        d_in, d = self.dims
        wk = self.flat[: d_in * d].view(d_in, d)
        wq = self.flat[d_in * d :].view(d_in, d)

        N, T, n, d_in = x.shape
        x = x.view(-1, n, d_in)  # stack the time and batch together

        h = torch.bmm(torch.matmul(x, wk), torch.matmul(x, wq).transpose(1, 2))
        A = torch.softmax(self.scale * h, dim=-1).sum(-2)
        idxs = torch.argsort(A, descending=True)[:, : self.topk]
        return idxs.view(N, T, self.topk, 1)

    def get_genotype(self):
        """ Return parameters to be optimized by ES. """
        return self.flat.cpu()

    def set_genotype(self, params):
        """ Set parameters that are optimized by ES. """
        self.flat.data.copy_(torch.from_numpy(params))


def unfold(batch, window=7, stride=4):
    """ Extract patches (sliding window) from a batch of videos.

    Args:
        batch (torch.tensor): Batch of videos of size N, T, C, W, H
    """
    N, T, C, W, H = batch.shape
    batch = F.unfold(batch.view(-1, C, W, H), kernel_size=window, stride=stride)
    return batch.view(N, T, window ** 2 * C, -1).transpose(2, 3)


def conv_block(in_ch, out_ch):
    """ A block of several convolutions. """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 5, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
        nn.Conv2d(out_ch, out_ch, 5, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
        nn.Conv2d(out_ch, out_ch, 3, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
    )


class Head(nn.Module):
    """ A feature extractor. """

    def __init__(self):
        super(Head, self).__init__()
        self.block1 = conv_block(1, 32)  # 3 conv layers
        self.block2 = conv_block(32, 32)  # double it for greater good
        self.conv1x1 = nn.Conv2d(32, 1, 1)  # too many channels will kill you
        self.bn0 = nn.BatchNorm2d(1)

    def forward(self, x):  # pylint: disable=arguments-differ
        N, T, C, W, H = x.shape
        x = x.view(-1, C, W, H)
        x = self.conv1x1(self.block2(self.block1(x)))
        x = self.bn0(F.relu(x, inplace=True))
        _, C, W, H = x.shape
        return x.view(N, T, C, W, H)


class Glimpsy(nn.Module):
    """ Super duper model. """

    def __init__(self, unfold, attention, rnn, linear, head=None):
        super(Glimpsy, self).__init__()
        self.head = head
        self.unfold = unfold
        self.attention = attention
        self.rnn = rnn
        self.bn0 = nn.BatchNorm1d(self.rnn.hidden_size)
        self.linear = linear

    def forward(self, x):  # pylint: disable=arguments-differ
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
        x = x.gather(1, idxs.view(N * T, k, 1).expand(N * T, k, d_in))

        # ready for rnn
        x = x.view(N, T, k, d_in)  # unfold batch and time
        x = x.permute(1, 0, 2, 3)  # time first
        x = x.view(T, N, -1)  # collapse the local features
        out, (_, _) = self.rnn(x)
        out = self.bn0(out.mean(0).squeeze())  # mean over time
        return F.log_softmax(self.linear(out), dim=1)

    def get_genotype(self):
        """ Returns the params to be optimized by ES """
        return self.attention.get_genotype()

    def set_genotype(self, parameters):
        """ Sets the params optimized by ES """
        self.attention.set_genotype(parameters)


class Baseline(nn.Module):
    """ Super duper model baseline. """

    def __init__(self, head, rnn, linear):
        super(Baseline, self).__init__()
        self.head = head
        self.rnn = rnn
        self.bn0 = nn.BatchNorm1d(self.rnn.hidden_size)
        self.linear = linear

    def forward(self, x):  # pylint: disable=arguments-differ
        N, T, _, _, _ = x.shape

        x = self.head(x)

        # ready for rnn
        x = x.permute(1, 0, 2, 3, 4)  # time first
        x = x.view(T, N, -1)  # collapse the features
        out, (_, _) = self.rnn(x)
        out = self.bn0(out.mean(0).squeeze())  # mean over time
        return F.log_softmax(self.linear(out), dim=1)


def train_with_cma(opt, loaders, model, optim, es_optim, log):
    """ CMA-ES training routine. """
    loader, es_loader = loaders
    es_loader_itt = iter(es_loader)

    step = 0
    for epoch in range(opt.epochs):
        log.info(f"Epoch {epoch:03d} started ----------------------")

        correct = 0
        for idx, (data, target) in enumerate(loader):
            data, target = data.to(opt.device), target.to(opt.device)

            # Gradient optimization
            ys = model(data)
            loss = F.nll_loss(ys, target.squeeze())

            optim.zero_grad()
            loss.backward()
            optim.step()

            # compute accuracy
            pred = ys.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # CMA-ES optimization
            candidates, losses = es_optim.ask(), []
            # get a single batch
            try:
                data, target_ = next(es_loader_itt)
            except StopIteration:
                es_loader_itt = iter(es_loader)
                data, target_ = next(es_loader_itt)
            data, target_ = data.to(opt.device), target_.to(opt.device)

            for candidate in candidates:
                model.set_genotype(candidate)
                with torch.no_grad():
                    loss = F.nll_loss(model(data), target_.squeeze())
                    losses.append(loss.cpu().item())
            es_optim.tell(candidates, losses)
            # set the best attention
            model.set_genotype(candidates[np.argmin(losses)])

            # report stats
            if (idx % 64 == 0) and (idx != 0):
                stats = {
                    "acc": 100.0 * correct / (idx * opt.batch_size),
                    "bestFit": np.min(losses),
                    "unFit": np.max(losses),
                    "attnMean": model.get_genotype().mean().item(),
                    "attnVar": model.get_genotype().var().item(),
                    "gen": idx,
                }

                log.info(log.fmt.format(batch=idx, **stats))
                if idx % 512 == 0:
                    log.trace(step=step, **stats)

            step += 1

        # save some models
        torch.save(
            {
                "epoch": epoch,
                "acc": 100.0 * correct / len(loader.dataset),
                "model": model.state_dict(),
            },
            f"{opt.path}/chkpt_{epoch:02d}.pth",
        )


def train(opt, loader, model, optim, log):
    """ Usual training routine. """
    step_cnt = 0
    for epoch in range(opt.epochs):
        log.info(f"Epoch {epoch:03d} started ----------------------")

        correct, losses = 0, []
        for idx, (data, target) in enumerate(loader):
            data, target = data.to(opt.device), target.to(opt.device)

            ys = model(data)
            loss = F.nll_loss(ys, target.squeeze())

            optim.zero_grad()
            loss.backward()
            optim.step()

            # stats
            with torch.no_grad():
                pred = ys.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                losses.append(loss.cpu().item())

                if (idx % 512 == 0) and (idx != 0):
                    stats = {
                        "acc": 100.0 * correct / (idx * opt.batch_size),
                        "loss": torch.tensor(losses).mean().item(),
                    }

                    log.info(log.fmt.format(batch=idx, **stats))
                    log.trace(step=step_cnt, **stats)

            step_cnt += 1

        # save some models
        torch.save(
            {
                "epoch": epoch,
                "acc": 100.0 * correct / len(loader.dataset),
                "model": model.state_dict(),
            },
            f"{opt.path}/chkpt_{epoch:02d}.pth",
        )


def model_stats(model):
    """ Prints model stats. """
    for name, param in model.named_parameters():
        print(
            "{:16} mean={:6.5f}, var={:6.5f}. Grad: mean={:4.5f}".format(
                name,
                param.data.mean(),
                param.data.var(),
                param.grad.data.mean() if param.grad is not None else 42.42424,
            )
        )


def get_model(opt, num_labels):
    """ Configure and return a model. """
    if opt.model == "baseline":
        model = Baseline(
            Head(),
            nn.LSTM(44 ** 2, hidden_size=opt.hidden_size),
            nn.Linear(opt.hidden_size, num_labels),
        )
    else:
        model = Glimpsy(
            partial(unfold, window=opt.window, stride=4),
            SparseAttention(opt.window ** 2, topk=opt.topk),
            nn.LSTM(opt.window ** 2 * opt.topk, hidden_size=opt.hidden_size),
            nn.Linear(opt.hidden_size, SyncedMNIST.num_labels),
            head=Head(),
        )

    rlog.info(
        summary(
            model,
            torch.zeros((opt.batch_size, 10, 1, 64, 64)),
            show_input=True,
            show_hierarchical=True,
        ),
    )
    return model


def make_paths_(opt):
    """ Create experiment path. """
    experiment = f"{datetime.now().strftime('%b%d_%H%M%S')}_{opt.model}"
    opt.path = f"./results/{experiment}"
    os.makedirs(opt.path)
    torch.save(vars(opt), f"{opt.path}/cfg.pkl")
    return opt


def make_rlog(opt):
    """ Configure logger. """
    rlog.init("pff", path=opt.path, tensorboard=True)
    train_log = rlog.getLogger("pff.train")
    train_log.fmt = (
        "[{gen:03d}/{batch:04d}] acc={acc:2.2f}% | bestFit={bestFit:2.3f}"
        + ", unFit={unFit:2.3f} [μ={attnMean:2.3f}/σ={attnVar:2.3f}]"
    )
    if opt.model == "baseline":
        train_log.fmt = "[{batch:04d}] acc={acc:2.2f}%, loss={loss:2.3f}"
    msg = "Configuration:\n"
    for k, v in vars(opt).items():
        msg += f"   {k:16}:  {v}\n"
    rlog.info(msg)
    return train_log


def main(opt):
    """ Entry point."""
    torch.backends.cudnn.benchmark = True

    opt.device = torch.device("cuda")
    opt = make_paths_(opt)
    train_log = make_rlog(opt)

    # enough with the mess
    dset = SyncedMNIST(root="./data/SyncedMNIST")
    loader = DataLoader(dset, batch_size=opt.batch_size)
    model = get_model(opt, SyncedMNIST.num_labels).to(opt.device)

    # not your grandma's optimizer
    es_optim = None
    if opt.model != "baseline":
        es_optim = cma.CMAEvolutionStrategy(
            model.get_genotype().numpy(), 0.1, {"popsize": opt.popsize}
        )
    optim = O.Adam(model.parameters(), lr=0.001)

    if opt.model != "baseline":
        train_with_cma(
            opt,
            (loader, DataLoader(dset, batch_size=256)),
            model,
            optim,
            es_optim,
            train_log,
        )
    else:
        train(opt, loader, model, optim, train_log)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Glimpsy options.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument("model", type=str, help="model to run")
    PARSER.add_argument("-e", "--epochs", type=int, default=200, help=":")
    PARSER.add_argument("-w", "--window", type=int, default=7, help=":")
    PARSER.add_argument("-k", "--topk", type=int, default=12, help=":")
    PARSER.add_argument("--hidden-size", type=int, default=64, help=":")
    PARSER.add_argument("-b", "--batch-size", type=int, default=128, help=":")
    PARSER.add_argument("-p", "--popsize", type=int, default=32, help=":")
    main(PARSER.parse_args())
