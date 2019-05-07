from __future__ import print_function

from functools import partial

from helpers.layers import str_to_activ_module
from .reparameterizers.gumbel import GumbelSoftmax
from .reparameterizers.mixture import Mixture
from .reparameterizers.beta import Beta
from .reparameterizers.bernoulli import Bernoulli
from .reparameterizers.isotropic_gaussian import IsotropicGaussian
from .abstract_vae import AbstractVAE


class SimpleVAE(AbstractVAE):
    def __init__(self, input_shape, **kwargs):
        """ Implements a parallel (in the case of mixture-reparam) VAE

        :param input_shape: the input shape
        :returns: an object of AbstractVAE
        :rtype: AbstractVAE

        """
        super(SimpleVAE, self).__init__(input_shape, **kwargs)
        reparam_dict = {
            'beta': Beta,
            'bernoulli': Bernoulli,
            'discrete': GumbelSoftmax,
            'isotropic_gaussian': IsotropicGaussian,
            'mixture': partial(Mixture, num_discrete=self.config['discrete_size'],
                               num_continuous=self.config['continuous_size'])
        }
        self.reparameterizer = reparam_dict[self.config['reparam_type']](config=self.config)

        # build the encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def has_discrete(self):
        """ Returns true if there is a discrete reparameterization.

        :returns: True/False
        :rtype: bool

        """
        return isinstance(self.reparameterizer, GumbelSoftmax)

    def loss_function(self, recon_x, x, params):
        """ The -ELBO.

        :param recon_x: the (unactivated) reconstruction logits
        :param x: the original tensor
        :param params: the reparam dict
        :returns: loss dict
        :rtype: dict

        """
        mut_info = self.mut_info(params)
        return super(SimpleVAE, self).loss_function(recon_x, x, params,
                                                    mut_info=mut_info)
