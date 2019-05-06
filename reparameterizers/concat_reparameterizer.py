from __future__ import print_function
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from torch.autograd import Variable
from functools import partial

from helpers.utils import float_type, zeros, get_dtype
from helpers.layers import get_encoder, str_to_activ_module
from .beta import Beta
from .bernoulli import Bernoulli
from .gumbel import GumbelSoftmax
from .mixture import Mixture
from .isotropic_gaussian import IsotropicGaussian


class ConcatReparameterizer(nn.Module):
    def __init__(self, reparam_strs, config):
        """ Concatenates a set of reparameterizations.

        :param config: argparse config
        :returns: ConcatReparameterizer object
        :rtype: object

        """
        super(ConcatReparameterizer, self).__init__()
        self.config = config
        self.reparam_strs = reparam_strs
        self.reparameterizers = nn.ModuleList(self._generate_reparameterizers())
        print(self.reparameterizers)
        self.input_size = np.sum([r.input_size for r in self.reparameterizers])
        self.output_size = np.sum([r.output_size for r in self.reparameterizers])

        # to enumerate over for reparameterization
        self._input_sizing = [0] + list(np.cumsum([r.input_size for r in self.reparameterizers]))

    def _generate_reparameterizers(self):
        """ Helper to generate all the required reparamterizers

        :returns: a list of reparameterizers
        :rtype: list

        """
        reparam_dict = {
            'beta': Beta,
            'bernoulli': Bernoulli,
            'discrete': GumbelSoftmax,
            'isotropic_gaussian': IsotropicGaussian,
            'mixture': partial(Mixture, num_discrete=self.config['discrete_size'],
                               num_continuous=self.config['continuous_size'])
        }

        # build the base reparameterizers
        return [reparam_dict[reparam](config=self.config) for reparam in self.reparam_strs]

    def get_reparameterizer_scalars(self):
        """ Return all scalars from the reparameterizers (eg: tau in gumbel)

        :returns: dict of scalars
        :rtype: dict

        """
        reparam_scalar_map = {}
        for i, reparam in enumerate(self.reparameterizers):
            if isinstance(reparam, GumbelSoftmax):
                reparam_scalar_map['tau%d_scalar'%i] = reparam.tau
            elif isinstance(reparam, Mixture):
                reparam_scalar_map['tau%d_scalar'%i] = reparam.discrete.tau

        return reparam_scalar_map


    def mutual_info(self, dists):
        """ concatenates all the mutual infos together and returns.

        :param dists: a list of distribution params
        :param priors: a list (or None) of priors, used only in VRNN currently
        :returns: mut-info of shape [batch_size]
        :rtype: torch.Tensor

        """
        mi = None
        for param, reparam in zip(dists, self.reparameterizers):
            mi = reparam.mutual_info(param) if mi is None else mi + reparam.mutual_info(param)

        return mi


    def kl(self, dists, priors=None):
        """ concatenates all the KLs together and returns

        :param dists: a list of distribution params
        :param priors: a list (or None) of priors, used only in VRNN currently
        :returns: kl-divergence of shape [batch_size]
        :rtype: torch.Tensor

        """
        priors = [None for _ in range(len(dists))] if priors is None else priors
        assert len(priors) == len(dists)

        kl = None
        for param, prior, reparam in zip(dists, priors, self.reparameterizers):
            kl = reparam.kl(param, prior) if kl is None else kl + reparam.kl(param, prior)

        return kl

    def reparameterize(self, logits):
        """ execute the reparameterization layer-by-layer returning ALL params and last reparam logits.

        :param logits: the input logits
        :returns: concat reparam, list of params
        :rtype: torch.Tensor, list

        """
        params_list = []
        reparameterized = []

        for i, (begin, end) in enumerate(zip(self._input_sizing, self._input_sizing[1:])):
            # print("reparaming from {} to {} for {}-th reparam which is a {} with shape {}".format(
            #     begin, end, i, self.reparameterizers[i], logits[:, begin:end].shape))
            reparameterized_i, params = self.reparameterizers[i](logits[:, begin:end])
            reparameterized.append(reparameterized_i)
            params_list.append(params)

        return torch.cat(reparameterized, -1), params_list

    def forward(self, logits):
        return self.reparameterize(logits)

    def prior(self, batch_size, **kwargs):
        """ Gen the first prior.

        :param batch_size: the batch size to generate
        :returns: prior sample
        :rtype: torch.Tensor

        """
        return torch.cat([r.prior(batch_size) for r in self.reparameterizers], -1)
