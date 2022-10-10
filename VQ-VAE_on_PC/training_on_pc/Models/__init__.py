from .cae import ConvolutionalAutoEncoder
from .vae import VariationalAutoEncoder
from .unconventional_cae import Unconven_CAE
from .vq_vae import VQ_VAE


def get_model(name, og_dim):
    if name == 'cae':
        model = ConvolutionalAutoEncoder()
    elif name == 'vae':
        model = VariationalAutoEncoder(og_dim)
    elif name == 'unconventional_cae':
        model = Unconven_CAE()
    elif name == 'vq_vae':
        model = VQ_VAE()
    else:
        return 0

    return model
