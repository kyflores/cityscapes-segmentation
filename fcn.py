import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tm


class Fcn(nn.Module):
    def __init__(self, cin, cout, encoder="resnet50") -> None:
        super().__init__()
        assert cin == 3

        # Leverage pretrained resnet.
        # 18 - min channel dimension is 64
        # 50 - min channel dimension is 256
        if encoder == "resnet50":
            resnet = tm.resnet50(weights=tm.ResNet50_Weights.DEFAULT)
            filters = 256
        elif encoder == "resnet18":
            resnet = tm.resnet18(weights=tm.ResNet18_Weights.DEFAULT)
            filters = 64
        self.input = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.l1 = resnet.layer1
        self.cm1 = nn.Conv2d(filters, cout, kernel_size=1)

        self.l2 = resnet.layer2
        self.cm2 = nn.Conv2d(filters * 2, cout, kernel_size=1)

        self.l3 = resnet.layer3
        self.cm3 = nn.Conv2d(filters * 4, cout, kernel_size=1)

        self.l4 = resnet.layer4
        self.cm4 = nn.Conv2d(filters * 8, cout, kernel_size=1)

        self.up2x_1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                cout, cout, kernel_size=3, stride=1, padding=1, padding_mode="replicate"
            ),
            nn.ReLU(),
            nn.BatchNorm2d(cout),
        )
        self.up2x_2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                cout, cout, kernel_size=3, stride=1, padding=1, padding_mode="replicate"
            ),
            nn.ReLU(),
            nn.BatchNorm2d(cout),
        )
        self.up2x_3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                cout, cout, kernel_size=3, stride=1, padding=1, padding_mode="replicate"
            ),
            nn.ReLU(),
            nn.BatchNorm2d(cout),
        )

        self.up4x = nn.Sequential(
            nn.ConvTranspose2d(cout, cout, kernel_size=2, stride=2),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.ConvTranspose2d(cout, cout, kernel_size=2, stride=2),
            nn.BatchNorm2d(cout),
        )

    def forward(self, x):
        x = self.input(x)
        x1 = self.l1(x)
        p1 = self.cm1(x1)

        x2 = self.l2(x1)
        p2 = self.cm2(x2)

        x3 = self.l3(x2)
        p3 = self.cm3(x3)

        x4 = self.l4(x3)
        p4 = self.cm4(x4)

        u1 = self.up2x_1(p4)
        o1 = u1 + p3
        u2 = self.up2x_2(o1)
        o2 = u2 + p2
        u3 = self.up2x_3(o2)
        o3 = u3 + p1

        out = self.up4x(o3)
        return out


class FcnL(nn.Module):
    def __init__(self, cin, cout, encoder="resnet50") -> None:
        super().__init__()
        assert cin == 3

        # Leverage pretrained resnet.
        # 18 - min channel dimension is 64
        # 50 - min channel dimension is 256
        if encoder == "resnet50":
            resnet = tm.resnet50(weights=tm.ResNet50_Weights.DEFAULT)
            filters = 256
        else:
            raise ValueError("Unsupported encoder")

        self.input = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        # out = filters
        self.l1 = resnet.layer1

        # out = filters * 2
        self.l2 = resnet.layer2

        # out = filters * 4
        self.l3 = resnet.layer3

        # out = filters * 8
        self.l4 = resnet.layer4

        def get_upscaler(cin, cout):
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(cin, cin // 4, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(cin // 4),
                nn.Conv2d(
                    cin // 4,
                    cin // 4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.ReLU(),
                nn.BatchNorm2d(cin // 4),
                nn.Conv2d(
                    cin // 4,
                    cout,
                    kernel_size=1,
                    stride=1,
                ),
                nn.ReLU(),
                nn.BatchNorm2d(cout),
            )

        self.up2x_1 = get_upscaler(filters * 8, filters * 4)
        self.up2x_2 = get_upscaler(filters * 4, filters * 2)
        self.up2x_3 = get_upscaler(filters * 2, filters * 1)
        self.up2x_4 = get_upscaler(filters * 1, cout)

        self.up_out = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(cout, cout, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(cout),
            nn.Conv2d(cout, cout, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = self.input(x)
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)

        u1 = self.up2x_1(x4)
        u2 = self.up2x_2(x3 + u1)
        u3 = self.up2x_3(x2 + u2)
        u4 = self.up2x_4(x1 + u3)

        out = self.up_out(u4)
        return out


if __name__ == "__main__":
    fcn = FcnL(3, 20)
    out = fcn(torch.randn(1, 3, 512, 512))
    from torchinfo import summary

    summary(fcn, (1, 3, 512, 512))
