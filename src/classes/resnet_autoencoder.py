import torch.nn as nn

from classes.resnet_using_basic_block_encoder import Encoder, BasicBlockEnc
from classes.resnet_using_basic_block_decoder import Decoder, BasicBlockDec
from classes.resnet_using_light_basic_block_encoder import LightEncoder, LightBasicBlockEnc
from classes.resnet_using_light_basic_block_decoder import LightDecoder, LightBasicBlockDec

class AE(nn.Module):
    """Construction of resnet autoencoder.

    Attributes:
        network (str): the architectural type of the network. There are 2 choices:
            - 'default' (default), related with the original resnet-18 architecture
            - 'light', a samller network implementation of resnet-18 for smaller input images.
        num_layers (int): the number of layers to be created. Implemented for 18 layers (default) for both types 
            of network, 34 layers for default only network and 20 layers for light network. 
    """

    def __init__(self, network='default', num_layers=18):
        """Initialize the autoencoder.

        Args:
            network (str): a flag to efine the network version. Choices ['default' (default), 'light'].
             num_layers (int): the number of layers to be created. Choices [18 (default), 34 (only for 
                'default' network), 20 (only for 'light' network).
        """
        super().__init__()
        self.network = network
        if self.network == 'default':
            if num_layers==18:
                # resnet 18 encoder
                self.encoder = Encoder(BasicBlockEnc, [2, 2, 2, 2]) 
                # resnet 18 decoder
                self.decoder = Decoder(BasicBlockDec, [2, 2, 2, 2]) 
            elif num_layers==34:
                # resnet 34 encoder
                self.encoder = Encoder(BasicBlockEnc, [3, 4, 6, 3]) 
                # resnet 34 decoder
                self.decoder = Decoder(BasicBlockDec, [3, 4, 6, 3]) 
            else:
                raise NotImplementedError("Only resnet 18 & 34 autoencoder have been implemented for images size >= 64x64.")
        elif self.network == 'light':
            if num_layers==18:
                # resnet 18 encoder
                self.encoder = LightEncoder(LightBasicBlockEnc, [2, 2, 2]) 
                # resnet 18 decoder
                self.decoder = LightDecoder(LightBasicBlockDec, [2, 2, 2]) 
            elif num_layers==20:
                # resnet 18 encoder
                self.encoder = LightEncoder(LightBasicBlockEnc, [3, 3, 3]) 
                # resnet 18 decoder
                self.decoder = LightDecoder(LightBasicBlockDec, [3, 3, 3]) 
            else:
                raise NotImplementedError("Only resnet 18 & 20 autoencoder have been implemented for images size < 64x64.")
        else:
                raise NotImplementedError("Only default and light resnet have been implemented. Th light version corresponds to input datasets with size less than 64x64.")

    def forward(self, x):
        """The forward functon of the model.

        Args:
            x (torch.tensor): the batched input data

        Returns:
            x (torch.tensor): encoder result
            z (torch.tensor): decoder result
        """
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z
    