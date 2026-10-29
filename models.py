"""Models: a small VGG-style CNN (the main model) and a linear baseline."""
import torch.nn as nn


def conv_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SmallCNN(nn.Module):
    """[conv-bn-relu x2, pool] -> [x2, pool] -> [x1] -> global-avg-pool -> dropout -> linear.
    ~140k parameters with width=32."""

    def __init__(self, width=32, dropout=0.3, n_classes=10):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            conv_block(1, w), conv_block(w, w), nn.MaxPool2d(2),            # 28 -> 14
            conv_block(w, 2 * w), conv_block(2 * w, 2 * w), nn.MaxPool2d(2),  # 14 -> 7
            conv_block(2 * w, 4 * w), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(4 * w, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class LinearBaseline(nn.Module):
    """Multinomial logistic regression = one linear layer + softmax-CE."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, n_classes))

    def forward(self, x):
        return self.net(x)
