import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class SyncedMNIST(Dataset):
    def __init__(self, root, split="train"):
        self.__root = root = Path(root)
        self.__split = split
        self.__data_dir = root / f"{split}_5_digits"
        self.__parts = [x for x in self.__data_dir.iterdir() if x.is_file()]
        self.__part_idxs = pidxs = torch.randperm(len(self.__parts)).tolist()
        self.__data = pickle.load(open(self.__parts[pidxs.pop()], "rb"))
        self.__sample_cnt = 0  # up to 1000

    def __getitem__(self, idx):
        idx %= 1000

        if not self.__part_idxs:
            print("bang: ", self.__sample_cnt == 1000, idx)
            self.__parts_idxs = torch.randperm(len(self.__parts)).tolist()

        if self.__sample_cnt == 1000:
            fp = open(self.__parts[self.__part_idxs.pop()], "rb")
            self.__data = pickle.load(fp)
            self.__sample_cnt = 0
            print(len(self.__part_idxs), idx)

        self.__sample_cnt += 1

        video = torch.from_numpy(self.__data["videos"][idx]).float() / 255
        label = torch.from_numpy(self.__data["labels"][idx]).long()

        video.unsqueeze_(1)

        return video, label

    def __len__(self):
        return len(self.__parts) * 1000


if __name__ == "__main__":
    dset = SyncedMNIST(root="./datasets/SyncedMNIST")
    # print("Len: ", len(dset))
    # for i in range(600_001):
    #     data, label = dset[i]
    #     print(data.shape, label)

    loader = DataLoader(dset, batch_size=64)

    for idx, (data, target) in enumerate(loader):
        print(idx, data.shape, target.shape)
