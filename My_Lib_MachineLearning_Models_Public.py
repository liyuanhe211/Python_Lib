import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class ResNet1D_Regressor(nn.Module):
    def __init__(self, input_len=171, input_channels=1):
        super(ResNet1D_Regressor, self).__init__()
        
        self.in_channels = 64
        
        # Initial Conv: Kernel size 7, Stride 1, Padding 3
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual Blocks
        # 3 stacked blocks as suggested
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Regression Heads
        # Position Head: Outputs a scalar representing the jump index (normalized to [0, 1])
        # Using Sigmoid to constrain to [0, 1]
        self.fc_pos = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )
        
        # Size Head: Outputs a scalar representing the jump magnitude
        # No activation or maybe Tanh/ReLU depending on sign? Jump size is likely positive?
        # User said "add constant size", "size ... 0~50%".
        # We'll assume linear output for size.
        self.fc_size = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _make_layer(self, in_channels, out_channels, stride):
        layer = ResidualBlock1D(in_channels, out_channels, stride)
        return layer

    def forward(self, x):
        # x shape: (Batch, 1, 171)
        x = self.initial_conv(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # Flatten: (Batch, 256)
        
        pos = self.fc_pos(x)   # (Batch, 1)
        size = self.fc_size(x) # (Batch, 1)
        
        # Concatenate outputs: (Batch, 2) -> [Position, Size]
        return torch.cat((pos, size), dim=1)

class MLP_LeakyReLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, depth):
        super(MLP_LeakyReLU, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.model(x)
        return out



class OffsetCorrectionResNet(nn.Module):
    def __init__(self, input_len=171, input_channels=1):
        super(OffsetCorrectionResNet, self).__init__()
        
        self.in_channels = 64
        self.input_len = input_len
        
        # Initial Conv: Kernel size 7, Stride 1, Padding 3
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual Blocks
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Regression Heads
        # Position Head
        self.fc_pos = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )
        
        # Size Head - Dependent on features and pos
        self.fc_size = nn.Sequential(
            nn.Linear(256 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _make_layer(self, in_channels, out_channels, stride):
        layer = ResidualBlock1D(in_channels, out_channels, stride)
        return layer

    def forward(self, x):
        # x shape: (Batch, 1, 171) or (Batch, 171)
        if x.dim() == 2:
            self.last_input = x
            x = x.unsqueeze(1)
        else:
            self.last_input = x.squeeze(1) # (Batch, 171)
        
        out = self.initial_conv(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.global_pool(out)
        features = out.view(out.size(0), -1) # Flatten: (Batch, 256)
        
        # Predict Position (0~1)
        pos = self.fc_pos(features)   # (Batch, 1)
        
        # Predict Size
        # Concatenate features and pos
        feature_and_pos = torch.cat((features, pos), dim=1) # (Batch, 257)
        size = self.fc_size(feature_and_pos) # (Batch, 1)
        
        # Return only pos and size (Batch, 2)
        return torch.cat((pos, size), dim=1)
