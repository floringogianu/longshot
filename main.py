""" Entry point.

    Inspired by https://attentionagent.github.io/
"""
from functools import partial

import cma
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.wk = torch.nn.Parameter(torch.randn(d_in, d))
        self.wq = torch.nn.Parameter(torch.randn(d_in, d))
        self.topk = topk
        self.scale = 1 / torch.sqrt(torch.tensor(d_in).float())

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

    def to(self, device):
        super().to(device)
        self.scale.to(device)


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
        N, T, topk, d_in = x.shape
        x = x.permute(1, 0, 2, 3)  # time first
        x = x.view(T, N, -1)  # collapse the local features
        _, (hn, _) = self.rnn(x)
        return self.linear(hn.squeeze())

    def flat_parameters(self):
        """ Warning, this changes the ref to params.data. """
        self._flat_params = torch.tensor([])
        for p in self.parameters():
            if p.requires_grad:
                self._flat_params = torch.cat(
                    [self._flat_params, p.data.flatten()]
                )
        # and now we change the reference
        last_idx = 0
        for p in self.parameters():
            if p.requires_grad:
                p.data = self._flat_params[
                    last_idx : last_idx + p.data.numel()
                ].view(p.shape)
                last_idx += p.data.numel()
        return self._flat_params

    def set_parameters(self, params):
        """ Warning, this also changes the ref to params.data. """
        last_idx = 0
        q = torch.from_numpy(params)
        for p in self.parameters():
            if p.requires_grad:
                p.data = self._flat_params[
                    last_idx : last_idx + p.data.numel()
                ].view(p.shape)
                last_idx += p.data.numel()


def evaluate_model(model, data, candidates):
    videos, targets = data
    losses = []
    for candidate in candidates:
        model.set_parameters(candidate)
        ys = model(videos)
        losses.append(F.cross_entropy(ys, targets.squeeze()).item())
    return losses


def main():
    """ Entry point."""
    kernel_size = 7
    topk = 7
    hidden_size = 16
    num_labels = 46
    batch_size = 32

    dset = SyncedMNIST(root="./data/SyncedMNIST")
    loader = DataLoader(dset, batch_size=batch_size)

    model = Glimpsy(
        partial(unfold, kernel=kernel_size, stride=4),
        SparseAttention(kernel_size ** 2, topk=topk),
        nn.LSTM(kernel_size ** 2 * topk, hidden_size=hidden_size, bias=False),
        nn.Linear(hidden_size, num_labels),
    )

    print(
        summary(
            model,
            torch.zeros((batch_size, 10, 1, 64, 64)),
            show_input=True,
            show_hierarchical=True,
        )
    )

    # not your grandma's optimizer
    optim = cma.CMAEvolutionStrategy(model.flat_parameters().numpy(), 0.1)

    for idx, (data, target) in enumerate(loader):
        with torch.no_grad():
            candidates = optim.ask()
            losses = evaluate_model(model, (data, target), candidates)
            optim.tell(candidates, losses)
        print(idx, losses)


if __name__ == "__main__":
    main()
