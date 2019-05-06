from __future__ import print_function

from .abstract_vae import AbstractVAE
from .reparameterizers.sequential_reparameterizer import SequentialReparameterizer


class SequentiallyReparameterizedVAE(AbstractVAE):
    def __init__(self, input_shape, reparameterizer_strs=["discrete", "isotropic_gaussian"], **kwargs):
        """ Implements a set of latent reparameterized variables, eg, disc --> NN --> gauss.

        :param input_shape: the input tensor shape
        :param reparameterizer_strs: the type of reparameterizers to use
        :returns: SequentiallyReparameterizedVAE object
        :rtype: nn.Module

        """
        super(SequentiallyReparameterizedVAE, self).__init__(input_shape, **kwargs)
        self.reparameterizer_strs = reparameterizer_strs
        self.reparameterizer = SequentialReparameterizer(reparameterizer_strs, self.config)

        # build the encoder and decoder here because of sizing
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def has_discrete(self):
        """ Returns true if there is a discrete reparameterization
            for the 0th layer.

        :returns: True/False
        :rtype: bool

        """
        return isinstance(self.reparameterizer.reparameterizers[0], GumbelSoftmax)

    def loss_function(self, recon_x, x, params):
        """ The -ELBO.

        :param recon_x: the (unactivated) reconstruction logits
        :param x: the original tensor
        :param params: the reparam dict
        :returns: loss dict
        :rtype: dict

        """
        mut_info = self.mut_info(params)
        return super(SequentiallyReparameterizedVAE, self).loss_function(recon_x, x, params,
                                                                         mut_info=mut_info)
