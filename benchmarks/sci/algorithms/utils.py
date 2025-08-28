""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, radius=1, base_channels=16):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.radius = radius
        self.base_channels = base_channels
        self.input_size = 2 * radius + 1  # radius=1 -> 3x3, radius=2 -> 5x5, etc.
        
        # Calculate how many downsampling layers we can use based on input size
        # Each down layer reduces size by 2x, so we need input_size >= 2^(n+1) to avoid 1x1 after n downs
        # For safe upsampling, we need at least 2x2 feature maps
        max_downs = max(0, int(torch.log2(torch.tensor(self.input_size / 2, dtype=torch.float32)).floor()))
        self.n_downs = min(max_downs, 4)  # Cap at 4 as in original
        
        # For very small inputs, use no downsampling
        if self.input_size <= 4:
            self.n_downs = 0
        
        # Initial convolution
        self.inc = DoubleConv(n_channels, base_channels)
        
        # Build downsampling path dynamically
        self.downs = nn.ModuleList()
        in_ch = base_channels
        for i in range(self.n_downs):
            out_ch = min(in_ch * 2, 256)  # Cap channels at 256
            self.downs.append(Down(in_ch, out_ch))
            in_ch = out_ch
        
        # Factor for bilinear upsampling
        factor = 2 if bilinear else 1
        
        # Build upsampling path dynamically
        self.ups = nn.ModuleList()
        
        # Only build upsampling layers if we have downsampling layers
        if self.n_downs > 0:
            # Store channel sizes from down path for proper skip connections
            down_channels = [base_channels]  # Initial channel count
            temp_ch = base_channels
            for i in range(self.n_downs):
                temp_ch = min(temp_ch * 2, 256)
                down_channels.append(temp_ch)
            
            # Build up path in reverse
            # Start with the deepest feature map
            current_ch = down_channels[-1]  # Channels at bottom of U-Net
            
            for i in range(self.n_downs):
                # Skip connection comes from the corresponding down layer
                skip_ch = down_channels[-(i+2)]  # Skip connection channels
                
                # For your Up class:
                # - x1 (current_ch) gets upsampled 
                # - If bilinear: channels stay the same, so x1 becomes current_ch channels
                # - If ConvTranspose: x1 becomes current_ch // 2 channels
                # - Then concat with x2 (skip_ch channels)
                # - Total after concat: current_ch + skip_ch (bilinear) or (current_ch//2) + skip_ch (transpose)
                
                if bilinear:
                    concat_channels = current_ch + skip_ch
                else:
                    concat_channels = (current_ch // 2) + skip_ch
                
                # Output channels for this layer
                if i == self.n_downs - 1:  # Last up layer
                    out_ch = base_channels
                else:
                    out_ch = skip_ch  # Match the skip connection size
                
                self.ups.append(Up(concat_channels, out_ch, bilinear))
                current_ch = out_ch
        
        # Global pooling + classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # [B, C, H, W] -> [B, C, 1, 1]
            nn.Flatten(),              # [B, C, 1, 1] -> [B, C]
            nn.Linear(base_channels, n_classes)   # [B, base_channels] -> [B, n_classes]
        )
    
    def forward(self, x):
        # Store original size
        original_size = x.shape[-1]
        
        # Validate input size
        expected_size = self.input_size
        if x.shape[-1] != expected_size or x.shape[-2] != expected_size:
            raise ValueError(f"Expected input size {expected_size}x{expected_size} for radius={self.radius}, "
                           f"got {x.shape[-2]}x{x.shape[-1]}")
        
        # Calculate padding needed to make size divisible by 2^n_downs
        pad_total = 0
        if self.n_downs > 0:
            # Find the smallest size >= input_size that's divisible by 2^n_downs
            divisor = 2 ** self.n_downs
            padded_size = ((self.input_size + divisor - 1) // divisor) * divisor
            
            # Calculate padding
            pad_total = padded_size - self.input_size
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            pad_top = pad_total // 2  
            pad_bottom = pad_total - pad_top
            
            # Apply padding
            if pad_total > 0:
                x = nn.functional.pad(x, [pad_left, pad_right, pad_top, pad_bottom], mode='reflect')
        
        # Initial convolution
        features = [self.inc(x)]
        
        # Downsampling path
        for down in self.downs:
            features.append(down(features[-1]))
        
        # If no downsampling layers, just use the initial features
        if self.n_downs == 0:
            logits = self.head(features[0])
            return logits
        
        # Start with the bottom-most feature
        x = features[-1]
        
        # Upsampling path with skip connections
        for i, up in enumerate(self.ups):
            skip_idx = -(i + 2)  # Get corresponding skip connection
            x = up(x, features[skip_idx])
        
        # Remove padding to restore original size
        if pad_total > 0:
            pad_left = pad_total // 2
            pad_top = pad_total // 2
            x = x[:, :, pad_top:pad_top+original_size, pad_left:pad_left+original_size]
        
        # Classification head
        logits = self.head(x)
        return logits



##############
# UNet Utils #
##############
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, p=0.1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout(p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConvMultiChannel(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, mid_channels, out_channels, p=0.1):
        super(DoubleConvMultiChannel, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout(p),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

######################
# UNetConvLSTM Utils #
######################
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x):
        batch_size, seq_len, _, height, width = x.size()
        h, c = self.cell.init_hidden(batch_size, (height, width))

        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :, :, :], (h, c))
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)
    
# Attention Gate
class AttentionBlock(nn.Module):
    """
    Attention block as proposed in 'Attention U-Net: Learning Where to Look for the Pancreas'
    (https://arxiv.org/pdf/1804.03999).
    This block takes:
      - gating signal g (decoder feature)
      - skip connection x (encoder feature)
    to produce a spatial attention map, which is then multiplied by the skip connection.
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of channels in the gating signal (decoder)
            F_l: Number of channels in the skip connection (encoder)
            F_int: Number of intermediate channels (usually smaller than F_g and F_l)
        """
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: gating signal (decoder feature)
        x: skip connection (encoder feature)
        Returns:
            scaled skip-connection: x * att_weight
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Combine gating & skip signals
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # attention coefficients in [0,1]

        # Multiply attention coefficients by skip connection
        out = x * psi
        return out
    
class UpAtt(nn.Module):
    """
    Up-sampling + Attention + DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpAtt, self).__init__()
        
        # If bilinear, reduce channels with a 1x1 convolution after upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv_reduce = None
        
        # Attention block
        #   F_g = in_channels//2 (decoder after up-sampling)
        #   F_l = out_channels (skip connection channels from encoder)
        #   F_int = out_channels // 2 (arbitrary choice, can be changed)
        self.attention = AttentionBlock(F_g=in_channels // 2, 
                                        F_l=out_channels, 
                                        F_int=out_channels // 2)
        
        # Final convolution after concatenation
        #   We'll end up with (decoder + skip) = out_channels + in_channels//2
        #   That total feeds into DoubleConv(...).
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: decoder feature (from the previous up block or bottom of the UNet)
        x2: encoder feature (skip connection)
        """
        # Upsample
        x1 = self.up(x1)
        if self.conv_reduce is not None:
            x1 = self.conv_reduce(x1)

        # Attention gating on the skip connection
        x2 = self.attention(x1, x2)

        # Concatenate
        x = torch.cat([x2, x1], dim=1)

        # Convolve
        return self.conv(x)

#######################
# GSTUNet Utils #
#######################
class GHead(nn.Module):
    """Used by GUNet"""

    def __init__(self, hr_size, fc_layer_sizes, dim_treatments, dim_outcome):
        super().__init__()

        self.hr_size = hr_size
        self.fc_layer_sizes = fc_layer_sizes
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome

        layers = []

        in_features = self.hr_size + self.dim_treatments
        for size in self.fc_layer_sizes:
            layers.append(nn.Linear(in_features, size))
            layers.append(nn.ELU())  # Activation function
            in_features = size
        
        # Add the final layer to produce the outcome
        layers.append(nn.Linear(in_features, self.dim_outcome))
        
        # Store as a sequential model
        self.network = nn.Sequential(*layers)

        # Initialize weights
        #self._initialize_weights()

    def forward(self, hr, current_treatment):
        x = torch.cat((hr, current_treatment), dim=-1)
        return self.network(x)
    
    def _initialize_weights(self):
        """Apply Kaiming initialization to linear layers."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                #nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    
def get_k_hop_neighbors(graph, node, k):
    """
    Find all neighbors within k hops of a given node.
    
    Args:
        graph: NetworkX graph object
        node: The starting node
        k: Maximum number of hops to consider
    
    Returns:
        set: All nodes within k hops of the given node (excluding the node itself)
    """
    
    # Get all nodes within k hops (including the starting node)
    ego_subgraph = nx.ego_graph(graph, node, radius=k)
    
    # Return all nodes except the starting node itself
    return set(ego_subgraph.nodes()) - {node}