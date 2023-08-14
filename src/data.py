# Importing the relevant libraries
import os
import torch
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from utils import iou_width_height as iou

#
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]
IMAGE_SIZE = 416
C = 11
S = [13, 26, 52]


class YOLOv3Dataset(Dataset):
    def __init__(
        self,
        metadata_path,
        data_path,
        S=S,
        image_size=IMAGE_SIZE,
        anchors=ANCHORS,
        num_classes=C,
        transform=None,
    ):
        self.annotations = pd.read_csv(metadata_path)
        self.data_path = data_path
        self.S = S
        self.image_size = image_size
        self.anchors = anchors
        self.transform = transform

        #
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.anchors_per_scale = self.anchors.shape[0] // 3
        self.total_anchors = self.anchors.shape[0]
        self.num_classes = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image = np.array(
            Image.open(os.path.join(self.data_path, row["images"]))
            .convert("RGB")
            .resize((self.image_size, self.image_size))
        )
        bboxes = np.roll(
            np.loadtxt(
                os.path.join(self.data_path, row["metadata"]), delimiter=" ", ndmin=2
            ),
            4,
            axis=1,
        ).tolist()

        # if self.transform:
        #     augmented = self.transform(**{"image": image, "bboxes": bboxes})
        #     image = augmented["image"]
        #     bboxes = augmented["bboxes"]

        bbox_tensor = torch.tensor(bboxes)

        out = [
            torch.zeros((self.anchors_per_scale, scale, scale, 5 + 1))
            for scale in self.S
        ]

        for bbx in bbox_tensor:
            class_, xmid, ymid, width, height = bbx
            # print(bbx)
            iou_anchors = iou(bbx[2:4], self.anchors)
            sorted_anchors = iou_anchors.argsort(descending=True)

            has_anchor = [False] * 3

            for anch in sorted_anchors:
                scale_idx = anch // 3
                anch_idx = anch % 3

                scale = self.S[scale_idx]

                i, j = int(scale * ymid), int(scale * xmid)
                # print(i,j)
                cell_x, cell_y = (scale * xmid) - j, (scale * ymid) - i

                status = out[scale_idx][anch_idx, i, j, 0]
                if not status and not has_anchor[scale_idx]:
                    out[scale_idx][anch_idx, i, j, 0] = 1

                    newW, newH = (scale * width), (scale * height)
                    newbbox = torch.tensor([cell_x, cell_y, newW, newH])

                    out[scale_idx][anch_idx, i, j, 1:5] = newbbox
                    out[scale_idx][anch_idx, i, j, 5] = class_

                    has_anchor[scale_idx] = True

                elif not status and iou_anchors[anch] > 0.5:
                    out[scale_idx][anch_idx, i, j, 0] = -1

        return torch.tensor(image), out


def main():
    data_path = "../data/export"
    metadata_path = os.path.join(data_path, "data.csv")

    data_class = YOLOv3Dataset(metadata_path=metadata_path, data_path=data_path)
    print(len(data_class))
    image, bboxes = data_class[0]
    print(f"Image: {image}   Bboxes: {bboxes}")


if __name__ == "__main__":
    main()
