# Importing the relevant torch libraries
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim
from config import config


# Initializing the model
# CNN block
class CNNBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_batch_norm: bool, **kwargs
    ) -> None:
        super().__init__()
        ## Initializing the Convolution Layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=not is_batch_norm,
            **kwargs
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.is_batch = is_batch_norm

    def forward(self, x):
        ## Recieves x  as [Batch, channels, X, Y]
        x = (
            self.conv(x)
            if not self.is_batch
            else self.leaky_relu(self.batch_norm(self.conv(x)))
        )
        return x


### Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_repeats, use_residual) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_repeats = num_repeats
        self.use_residual = use_residual

        ## We just shrink the channels (Feature Size Remain the same)
        for i in range(self.num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=in_channels // 2,
                        is_batch_norm=True,
                        kernel_size=1,
                    ),
                    CNNBlock(
                        in_channels=in_channels // 2,
                        out_channels=in_channels,
                        is_batch_norm=True,
                        kernel_size=3,
                        padding=1,
                    ),
                ),
            ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)

        return x


### Scale Prediction
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pred = nn.Sequential(
            CNNBlock(
                in_channels=in_channels,
                out_channels=in_channels * 2,
                is_batch_norm=True,
                kernel_size=3,
                padding=1,
            ),
            CNNBlock(
                in_channels=in_channels * 2,
                out_channels=(self.num_classes + 5) * 3,
                is_batch_norm=False,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        x = self.pred(x)
        x = x.view(x.shape[0], 3, (self.num_classes + 5), x.shape[2], x.shape[3])
        ## Output Should be (Batch, 3 (for anchors), 13, 13, Num Classes + 5)
        return x.permute(0, 1, 3, 4, 2)


### YOLOv3 Class
class YOLOv3(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers_model = self._create_conv()

    def forward(self, x):
        outputs_scale = []
        routing = []

        for layer in self.layers_model:
            if isinstance(layer, ScalePrediction):
                outputs_scale.append(layer(x))
                continue
            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                routing.append(x)
            if isinstance(layer, nn.Upsample):
                x = torch.cat([x, routing[-1]], dim=1)
                routing.pop()

        return outputs_scale

    def _create_conv(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for category in config:
            if isinstance(category, tuple):
                out_channels, kernel_size, stride = category
                layers += [
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        is_batch_norm=True,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                ]
                in_channels = out_channels

            if isinstance(category, list):
                _, repeats = category
                layers.append(
                    ResidualBlock(
                        in_channels=in_channels, num_repeats=repeats, use_residual=True
                    )
                )

            if isinstance(category, str):
                if category == "S":
                    layers += [
                        ResidualBlock(
                            in_channels=in_channels, num_repeats=1, use_residual=False
                        ),
                        CNNBlock(
                            in_channels=in_channels,
                            out_channels=in_channels // 2,
                            kernel_size=1,
                            is_batch_norm=True,
                        ),
                        ScalePrediction(
                            in_channels=in_channels // 2, num_classes=self.num_classes
                        ),
                    ]
                    in_channels = in_channels // 2
                elif category == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3

        return layers


if __name__ == "__main__":
    # Testing Code
    num_classes = 20
    image_size = 416
    model = YOLOv3(in_channels=3, num_classes=num_classes)

    rand_sample = torch.rand((2, 3, image_size, image_size))
    out = model(rand_sample)

    assert out[0].shape == (2, 3, image_size // 32, image_size // 32, num_classes + 5)
    assert out[1].shape == (2, 3, image_size // 16, image_size // 16, num_classes + 5)
    assert out[2].shape == (2, 3, image_size // 8, image_size // 8, num_classes + 5)

    print("success")
