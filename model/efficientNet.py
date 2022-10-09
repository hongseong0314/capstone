import torch
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

class DisEfficient(torch.nn.Module):
    def __init__(self) -> None:
        super(DisEfficient, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.dis_out = torch.nn.Linear(1000, 21)
        self.crop_out = torch.nn.Linear(1000, 11)
        torch.nn.init.xavier_normal_(self.dis_out.weight)
        torch.nn.init.xavier_normal_(self.crop_out.weight)
        

    def forward(self, x):
        x = self.backbone(x)
        dis_out = self.dis_out(x)
        crop_out = self.crop_out(x)
        return dis_out, crop_out

class DisEfficient2(torch.nn.Module):
    def __init__(self, args):
        super(DisEfficient2, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.crop_out = torch.nn.Linear(1280, 31)
        torch.nn.init.xavier_normal_(self.crop_out.weight)
        

    def forward(self, x):
        x = self.backbone(x)
        crop_out = self.crop_out(x)
        return crop_out

class DisEfficient3(torch.nn.Module):
    def __init__(self, args):
        super(DisEfficient3, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.crop_out = torch.nn.Linear(1280 + 11, 31)
        torch.nn.init.xavier_normal_(self.crop_out.weight)
        

    def forward(self, x, crop_x):
        x = self.backbone(x)
        x = torch.cat((x, crop_x), dim=1)
        crop_out = self.crop_out(x)
        return crop_out