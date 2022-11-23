import timm
import torch.nn as nn

class Regnet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = timm.create_model(args.CODER, pretrained=True,
                                    drop_path_rate=args.drop_path_rate,
                                    )
        
        num_head = self.encoder.head.fc.in_features
        self.encoder.head.fc = nn.Linear(num_head, 30)

    def forward(self, x):
        x = self.encoder(x)
        return x

class Regnet_test(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=False,
                                    drop_path_rate=0,
                                    )
        
        num_head = self.encoder.head.fc.in_features
        self.encoder.head.fc = nn.Linear(num_head, 31)
    
    def forward(self, x):
        x = self.encoder(x)
        return x

class Regnet_test30(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=False,
                                    drop_path_rate=0,
                                    )
        
        num_head = self.encoder.head.fc.in_features
        self.encoder.head.fc = nn.Linear(num_head, 30)
    
    def forward(self, x):
        x = self.encoder(x)
        return x