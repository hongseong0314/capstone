import timm
import torch.nn as nn

class PoolFormer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = timm.create_model(args.CODER, pretrained=True,
                                    drop_path_rate=args.drop_path_rate,
                                    )
        
        num_head = self.encoder.head.in_features
        self.encoder.head = nn.Linear(num_head, 30)

    def forward(self, x):
        x = self.encoder(x)
        return x

class PoolFormer_test(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=False,
                                    drop_path_rate=0,
                                    )
        
        num_head = self.encoder.head.in_features
        self.encoder.head = nn.Linear(num_head, 30)
    
    def forward(self, x):
        x = self.encoder(x)
        return x

class PoolFormer_radius(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = timm.create_model(args.CODER, pretrained=True,
                                    drop_path_rate=args.drop_path_rate,
                                    )
        
        num_head = self.encoder.head.in_features
        self.encoder.head = nn.Linear(num_head, 2)

    def forward(self, x):
        x = self.encoder(x)
        return x

class PoolFormer_radius_test(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=False,
                                    drop_path_rate=0,
                                    )
        
        num_head = self.encoder.head.in_features
        self.encoder.head = nn.Linear(num_head, 2)

    def forward(self, x):
        x = self.encoder(x)
        return x

class PoolFormer_non(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = timm.create_model(args.CODER, pretrained=True,
                                    drop_path_rate=args.drop_path_rate,
                                    )
        
        num_head = self.encoder.head.in_features
        self.encoder.head = nn.Linear(num_head, 5)

    def forward(self, x):
        x = self.encoder(x)
        return x

class PoolFormer_non_test(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=False,
                                    drop_path_rate=0,
                                    )
        
        num_head = self.encoder.head.in_features
        self.encoder.head = nn.Linear(num_head, 5)

    def forward(self, x):
        x = self.encoder(x)
        return x
