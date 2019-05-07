from __future__ import print_function
import pprint
import copy
import warnings
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from helpers.utils import float_type, ones_like
from .beta import Beta
from .gumbel import GumbelSoftmax
from .isotropic_gaussian import IsotropicGaussian


class Mixture(nn.Module):
    ''' continuous + discrete reparaterization '''
    def __init__(self, num_discrete, num_continuous, config, is_beta=False):
        super(Mixture, self).__init__()
        warnings.warn("\n\nMixture is depricated, use concat_reparam or sequential_reparam.\n")
        self.config = config
        self.is_beta = is_beta
        self.num_discrete_input = num_discrete
        self.num_continuous_input = num_continuous

        # setup the continuous & discrete reparameterizer
        self.continuous = IsotropicGaussian(config) if not is_beta else Beta(config)
        self.discrete = GumbelSoftmax(config)

        self.input_size = num_continuous + num_discrete
        self.output_size = self.discrete.output_size + self.continuous.output_size

    def get_reparameterizer_scalars(self):
        """ Returns any scalars used in reparameterization.

        :returns: dict of scalars
        :rtype: dict

        """
        return self.discrete.get_reparameterizer_scalars()

    def prior(self, batch_size, **kwargs):
        disc = self.discrete.prior(batch_size, **kwargs)
        cont = self.continuous.prior(batch_size, **kwargs)
        return torch.cat([cont, disc], 1)

    def mutual_info(self, params):
        dinfo = self.config['discrete_mut_info'] * self.discrete.mutual_info(params)
        cinfo = self.config['continuous_mut_info'] * self.continuous.mutual_info(params)
        return dinfo - cinfo

    def log_likelihood(self, z, params):
        cont = self.continuous.log_likelihood(z[:, 0:self.continuous.output_size], params)
        disc = self.discrete.log_likelihood(z[:, self.continuous.output_size:], params)
        return torch.cat([cont, disc], 1)

    def reparmeterize(self, logits):
        continuous_logits = logits[:, 0:self.num_continuous_input]
        discrete_logits = logits[:, self.num_continuous_input:]

        continuous_reparam, continuous_params = self.continuous(continuous_logits)
        discrete_reparam, disc_params = self.discrete(discrete_logits)
        merged = torch.cat([continuous_reparam, discrete_reparam], -1)

        # use a separate key for gaussian or beta
        continuous_value = continuous_params['gaussian'] if not self.is_beta else continuous_params['beta']
        continuous_key = 'gaussian' if not self.is_beta else 'beta'
        params = {continuous_key: continuous_value,
                  'discrete': disc_params['discrete'],
                  'z': merged}
        return merged, params

    def kl(self, dist_a, prior=None):
        continuous_kl = self.continuous.kl(dist_a, prior)
        disc_kl = self.discrete.kl(dist_a, prior)
        assert continuous_kl.shape == disc_kl.shape, "need to reduce kl to [#batch] before mixture"
        return continuous_kl + disc_kl

    def forward(self, logits):
        return self.reparmeterize(logits)
