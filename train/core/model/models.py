import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingResNet(nn.Module):
    def __init__(self, model_name, num_channel, embedding_size, pretrained):
        super().__init__()
        self.model = eval("models." + model_name)(pretrained=pretrained)
        if num_channel == 1:
            # replace first conv layer for training grayscale images.
            self.model.conv1 = nn.Conv2d(
                in_channels=num_channel,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        # Replace fully connected layers with an embedding layer
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=embedding_size)
        )

    def forward(self, image):
        embedding = self.model(image)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class EmbeddingNet(nn.Module):
    """
    Embedding extraction
    Parameters
    ----------
    embedding_size:
        Size of embedding vector.

    pretrained:
        Whether to use pretrained weight on ImageNet.
    """

    def __init__(self, embedding_size: int, channel: int = 1, pretrained=False):
        super().__init__()
        # resnet
        self.model = models.resnet18(pretrained=pretrained)
        # Replace first conv layer for training grayscale images
        self.model.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        # Replace fully connected layers with an embedding layer
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=embedding_size)
        )

        # mobile_net
        # self.model = models.mobilenet_v2(pretrained=True)
        # # replace the first module for training grayscale images
        # self.model.features._modules['0'] = models.mobilenet.ConvBNReLU(1, 32, stride=2)
        # self.model.classifier._modules['1'] = nn.Linear(in_features=1280, out_features=embedding_size)

        # vgg19
        # self.model = models.vgg19(pretrained=pretrained)
        # # Replace fully connected layers with an embedding layer
        # self.model.classifier[6] = nn.Linear(in_features=4096, out_features=embedding_size, bias=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding features from image.

        Parameters
        ----------
        image:
            grayscale image [1 x H x W].

        Returns
        -------
        torch.Tensor
            Embedding vector
        """
        embedding: torch.Tensor = self.model(image)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
