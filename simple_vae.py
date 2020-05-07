from __future__ import print_function

from .reparameterizers import get_reparameterizer
from .abstract_vae import AbstractVAE


class SimpleVAE(AbstractVAE):
    def __init__(self, input_shape, **kwargs):
        """ Implements a standard simple VAE.

        :param input_shape: the input shape
        :returns: an object of AbstractVAE
        :rtype: AbstractVAE

        """
        super(SimpleVAE, self).__init__(input_shape, **kwargs)
        self.reparameterizer = get_reparameterizer(self.config['reparam_type'])(config=self.config)

        # build the encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
