from __future__ import print_function
import pprint
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from torch.autograd import Variable
from functools import partial

from helpers.utils import float_type, ones_like
from helpers.layers import get_encoder, str_to_activ_module
from .beta import Beta
from .gumbel import GumbelSoftmax
from .mixture import Mixture
from .isotropic_gaussian import IsotropicGaussian


class SequentialReparameterizer(nn.Module):
    def __init__(self, reparam_strs, config):
        """ A simple struct to hold the input and output size

        :param config: argparse config
        :returns: SequentialReparameterizer object
        :rtype: object

        """
        super(SequentialReparameterizer, self).__init__()
        self.config = config
        self.reparam_strs = reparam_strs
        self.reparameterizers = nn.ModuleList(self._generate_reparameterizers())
        print(self.reparameterizers)
        self.input_size = self.reparameterizers[0].input_size
        self.output_size = self.reparameterizers[-1][-1].output_size \
            if len(self.reparameterizers) > 1 else self.reparameterizers[-1].output_size
        print("input = ", self.input_size, " |out = ", self.output_size)

    def _get_dense_net_map(self, name='dense'):
        """ Internal helper to build a dense network

        :param name: name of the network
        :returns: a nn.Sequential Dense net
        :rtype: nn.Sequential

        """
        config = deepcopy(self.config)
        config['encoder_layer_type'] = 'dense'
        return get_encoder(config, name=name)

    def _append_inter_projection(self, reparam, input_size, output_size, name):
        return nn.Sequential(
            self._get_dense_net_map(name=name)(
                input_size, output_size, nlayers=2,
                activation_fn=str_to_activ_module(self.config['activation'])),
            reparam
        )

    def _generate_reparameterizers(self):
        """ Helper to generate all the required reparamterizers

        :returns: a list of reparameterizers
        :rtype: list

        """
        reparam_dict = {
            'beta': Beta,
            'discrete': GumbelSoftmax,
            'isotropic_gaussian': IsotropicGaussian,
            'mixture': partial(Mixture, num_discrete=self.config['discrete_size'],
                               num_continuous=self.config['continuous_size'])
        }

        # build the base reparameterizers
        reparam_list = [reparam_dict[reparam](config=self.config) for reparam in self.reparam_strs]

        # tack on dense networks between them
        input_size = reparam_list[0].output_size
        for i in range(1, len(reparam_list)):
            reparam_list[i] = self._append_inter_projection(reparam_list[i],
                                                            input_size=input_size,
                                                            output_size=reparam_list[i].input_size,
                                                            name='reparam_proj{}'.format(i))
            input_size = reparam_list[i][-1].output_size

        return reparam_list

    def get_reparameterizer_scalars(self):
        """ Return all scalars from the reparameterizers (eg: tau in gumbel)

        :returns: dict of scalars
        :rtype: dict

        """
        reparam_scalar_map = {}
        for i, reparam in enumerate(self.reparameterizers):
            reparam_obj = reparam[-1] if isinstance(reparam, nn.Sequential) else reparam
            if isinstance(reparam_obj, GumbelSoftmax):
                reparam_scalar_map['tau%d_scalar'%i] = reparam_obj.tau
            elif isinstance(reparam_obj, Mixture):
                reparam_scalar_map['tau%d_scalar'%i] = reparam_obj.discrete.tau

        return reparam_scalar_map


    def mutual_info(self, params):
        raise NotImplementedError("Not implemented mut-info for sequential reparam")

    def kl(self, dists, priors=None):
        """ Adds all the KLs together and returns

        :param dists: a list of distribution params
        :param priors: a list (or None) of priors, used only in VRNN currently
        :returns: kl-divergence of shape [batch_size]
        :rtype: torch.Tensor

        """
        priors = [None for _ in range(len(dists))] if priors is None else priors
        assert len(priors) == len(dists)

        kl = None
        for param, prior, reparam in zip(dists, priors, self.reparameterizers):
            reparam_obj = reparam[-1] if isinstance(reparam, nn.Sequential) else reparam
            kl = reparam_obj.kl(param, prior) if kl is None else kl + reparam_obj.kl(param, prior)

        return kl

    def reparameterize(self, logits):
        """ execute the reparameterization layer-by-layer returning ALL params and last logits.

        :param logits: the input logits
        :returns: last logits, list of params
        :rtype: torch.Tensor, list

        """
        params_list = []
        for reparam in self.reparameterizers:
            logits, params = reparam(logits)
            params_list.append(params)

        return logits, params_list

    def forward(self, logits):
        return self.reparameterize(logits)

    def prior(self, batch_size, **kwargs):
        """ Gen the first prior.

        :param batch_size: the batch size to generate
        :returns: prior sample
        :rtype: torch.Tensor

        """
        return self.reparameterizers[0].prior(batch_size)
