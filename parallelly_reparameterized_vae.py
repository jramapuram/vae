from __future__ import print_function

from .abstract_vae import AbstractVAE
from .reparameterizers.concat_reparameterizer import ConcatReparameterizer


class ParallellyReparameterizedVAE(AbstractVAE):
    def __init__(self, input_shape, reparameterizer_strs=["discrete", "isotropic_gaussian"], **kwargs):
        """ Implements a parallel (in the case of mixture-reparam) VAE

        :param input_shape: the input shape
        :returns: an object of AbstractVAE
        :rtype: AbstractVAE

        """
        super(ParallellyReparameterizedVAE, self).__init__(input_shape, **kwargs)
        self.reparameterizer_strs = reparameterizer_strs
        self.reparameterizer = ConcatReparameterizer(reparameterizer_strs, self.config)

        # build the encoder and decoder here because of sizing
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()


    def has_discrete(self):
        """ Returns true if there is a discrete reparameterization.

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
        return super(ParallellyReparameterizedVAE, self).loss_function(recon_x, x, params,
                                                                       mut_info=mut_info)
