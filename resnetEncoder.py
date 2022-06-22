import torch.nn
from torch import nn

from models.unet3d.buildingblocks import create_encoders, ExtResNetBlock
from models.unet3d.utils import number_of_features_per_level


class resnetEncoder(nn.Module):
    def __init__(self, in_channels=1,f_maps=64,num_levels=4):
        super(resnetEncoder, self).__init__()
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        self.num_levels = num_levels
        self.encoders = nn.Sequential(*create_encoders(in_channels,
                                        f_maps=f_maps ,
                                        basic_module=ExtResNetBlock,
                                        conv_kernel_size=3,
                                        conv_padding=1,
                                        layer_order='cbl',
                                        num_groups=8,

                                        pool_kernel_size=2))
        self.fc =  nn.Sequential(nn.Linear(512 * 2, 2))
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1))).cuda()
    def forward(self,x):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            
            # reverse the encoder outputs to be aligned with the decoder
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #这里出bug就是fc的层数不对，比如说输入是(128，128，128)，fc输出层数就要为 8192， （64，128，128）fc输出层数就要为 4096，就是(shape[0]/16)*1024
        return  x
        return  torch.tensor(encoders_features)