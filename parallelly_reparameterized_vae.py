from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn

from helpers.layers import str_to_activ_module
from .reparameterizers.gumbel import GumbelSoftmax
from .reparameterizers.mixture import Mixture
from .reparameterizers.beta import Beta
from .reparameterizers.isotropic_gaussian import IsotropicGaussian
from .abstract_vae import AbstractVAE


class ParallellyReparameterizedVAE(AbstractVAE):
    def __init__(self, input_shape, **kwargs):
        """ Implements a parallel (in the case of mixture-reparam) VAE

        :param input_shape: the input shape
        :returns: an object of AbstractVAE
        :rtype: AbstractVAE

        """
        super(ParallellyReparameterizedVAE, self).__init__(input_shape, **kwargs)

        # build the reparameterizer
        if self.config['reparam_type'] == "isotropic_gaussian":
            print("using isotropic gaussian reparameterizer")
            self.reparameterizer = IsotropicGaussian(self.config)
        elif self.config['reparam_type'] == "discrete":
            print("using gumbel softmax reparameterizer")
            self.reparameterizer = GumbelSoftmax(self.config)
        elif self.config['reparam_type'] == "beta":
            print("using beta reparameterizer")
            self.reparameterizer = Beta(self.config)
        elif self.config['reparam_type'] == "mixture":
            print("using mixture reparameterizer")
            self.reparameterizer = Mixture(num_discrete=self.config['discrete_size'],
                                           num_continuous=self.config['continuous_size'],
                                           config=self.config)
        else:
            raise Exception("unknown reparameterization type")

        # build the encoder and decoder
        self.encoder = self.build_encoder()
        if not 'lazy_init_decoder' in kwargs:
            self.decoder = self.build_decoder()

    def get_reparameterizer_scalars(self):
        """ Returns any scalars used in reparameterization.

        :returns: dict of scalars
        :rtype: dict

        """
        reparam_scalar_map = {}
        if isinstance(self.reparameterizer, GumbelSoftmax):
            reparam_scalar_map['tau_scalar'] = self.reparameterizer.tau
        elif isinstance(self.reparameterizer, Mixture):
            reparam_scalar_map['tau_scalar'] = self.reparameterizer.discrete.tau

        return reparam_scalar_map

    def decode(self, z):
        """ Decode a latent z back to x.

        :param z: the latent tensor.
        :returns: decoded logits (unactivated).
        :rtype: torch.Tensor

        """
        return self.decoder(z.contiguous())

    def posterior(self, x):
        """ Encodes, reparmeterizes and returns dict.

        :param x: the input tensor
        :returns: an encode dict
        :rtype: dict

        """
        z_logits = self.encode(x)
        return self.reparameterize(z_logits)

    def reparameterize(self, logits):
        """ Reparameterize the logits and returns a dict.

        :param logits: unactivated encoded logits.
        :returns: reparam dict
        :rtype: dict

        """
        return self.reparameterizer(logits)

    def encode(self, x):
        """ Encodes a tensor x to a set of logits.

        :param x: the input tensor
        :returns: logits
        :rtype: torch.Tensor

        """
        return self.encoder(x)

    def kld(self, dist_a):
        """ KL-Divergence of the distribution dict and the prior of that distribution.

        :param dist_a: the distribution dict.
        :returns: tensor that is of dimension batch_size
        :rtype: torch.Tensor

        """
        return self.reparameterizer.kl(dist_a)

    def mut_info(self, dist_params):
        """ Returns mutual information between z <-> x

        :param dist_params: the distribution dict
        :returns: tensor of dimension batch_size
        :rtype: torch.Tensor

        """
        mut_info = None

        # only grab the mut-info if the scalars above are set
        if (self.config['continuous_mut_info'] > 0
             or self.config['discrete_mut_info'] > 0):
            mut_info = self.reparameterizer.mutual_info(dist_params)

        return mut_info

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
