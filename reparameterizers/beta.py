# coding: utf-8

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import pyro.distributions as PD
import torch.nn.functional as F
from torch.autograd import Variable

from helpers.utils import zeros_like, ones_like, same_type
from helpers.utils import eps as eps_fn


class Beta(nn.Module):
    def __init__(self, config):
        """ Beta distribution.

        :param config: argparse
        :returns: Beta module
        :rtype: nn.Module

        """
        super(Beta, self).__init__()
        self.config = config
        self.input_size = self.config['continuous_size']
        assert self.config['continuous_size'] % 2 == 0
        self.output_size = self.config['continuous_size'] // 2

    def prior(self, batch_size, **kwargs):
        """ Returns a Kerman beta prior.

        Kerman, J. (2011). Neutral noninformative and informative
        conjugate beta and gamma prior distributions. Electronic
        Journal of Statistics, 5, 1450-1470.

        :param batch_size: the number of prior samples
        :returns: prior
        :rtype: torch.Tensor

        """
        conc1 = Variable(
            same_type(self.config['half'], self.config['cuda'])(
                batch_size, self.output_size
            ).zero_() + 1/3
        )
        conc2 = Variable(
            same_type(self.config['half'], self.config['cuda'])(
                batch_size, self.output_size
            ).zero_() + 1/3
        )
        return PD.Beta(conc1, conc2).sample()

    def _reparametrize_beta(self, conc1, conc2, force=False):
        """ Internal function to reparameterize beta distribution using concentrations.

        :param conc1: concentration 1
        :param conc2: concentration 2
        :returns: reparameterized sample, distribution params
        :rtype: torch.Tensor, dict

        """
        if self.training or force:
            # rsample is CPU only ¯\_(ツ)_/¯, see https://tinyurl.com/y9e8mtcd
            # thus use pyro which DOES have a GPU version
            beta = PD.Beta(conc1, conc2).rsample()
            return beta, {'conc1': conc1, 'conc2': conc2}

        # can't use mean like in gaussian because beta mean can be > 1.0
        return PD.Beta(conc1, conc2).sample(), {'conc1': conc1, 'conc2': conc2}

    def reparmeterize(self, logits, force=False):
        """ Given logits reparameterize to a beta using
            first half of features for mean and second half for std.

        :param logits: unactivated logits
        :returns: reparameterized tensor (if training), param dict
        :rtype: torch.Tensor, dict

        """
        eps = eps_fn(self.config['half'])
        feature_size = logits.size(-1)
        assert feature_size % 2 == 0 and feature_size // 2 == self.output_size
        if logits.dim() == 2:
            conc1 = torch.sigmoid(logits[:, 0:int(feature_size/2)] + eps)
            conc2 = torch.sigmoid(logits[:, int(feature_size/2):] + eps)
        elif logits.dim() == 3:
            conc1 = torch.sigmoid(logits[:, :, 0:int(feature_size/2)] + eps)
            conc2 = torch.sigmoid(logits[:, :, int(feature_size/2):] + eps)
        else:
            raise Exception("unknown number of dims for isotropic gauss reparam")

        return self._reparametrize_beta(conc1, conc2, force=force)

    def _kld_beta_kerman_prior(self, conc1, conc2):
        """ Internal function to do a KL-div against the prior.

        :param conc1: concentration 1.
        :param conc2: concentration 2.
        :returns: batch_size tensor of kld against prior.
        :rtype: torch.Tensor

        """
        prior = PD.Beta(zeros_like(conc1) + 1/3,
                        zeros_like(conc2) + 1/3)
        beta = PD.Beta(conc1, conc2)
        return torch.sum(D.kl_divergence(beta, prior), -1)

    def kl(self, dist_a, prior=None):
        if prior == None:  # use standard reparamterizer
            return self._kld_beta_kerman_prior(
                dist_a['beta']['conc1'], dist_a['beta']['conc2']
            )

        # we have two distributions provided (eg: VRNN)
        return torch.sum(D.kl_divergence(
            PD.Beta(dist_a['beta']['conc1'], dist_a['beta']['conc2']),
            PD.Beta(prior['beta']['conc1'], prior['beta']['conc2'])
        ), -1)


    def mutual_info(self, params, eps=1e-9):
        """ I(z_d; x) ~ H(z_prior, z_d) + H(z_prior)

        :param params: parameters of distribution
        :param eps: tolerance
        :returns: batch_size mutual information (prop-to) tensor.
        :rtype: torch.Tensor

        """
        z_true = PD.Beta(params['beta']['conc1'],
                         params['beta']['conc2'])
        z_match = PD.Beta(params['q_z_given_xhat']['beta']['conc1'],
                          params['q_z_given_xhat']['beta']['conc2'])
        kl_proxy_to_xent = torch.sum(D.kl_divergence(z_match, z_true), dim=-1)
        return self.config['continuous_mut_info'] * kl_proxy_to_xent

    def log_likelihood(self, z, params):
        """ Log-likelihood of z induced under params.

        :param z: inferred latent z
        :param params: the params of the distribution
        :returns: log-likelihood
        :rtype: torch.Tensor

        """
        return PD.Beta(params['beta']['conc1'],
                       params['beta']['conc2']).log_prob(z)

    def forward(self, logits, force=False):
        """ Returns a reparameterized gaussian and it's params.

        :param logits: unactivated logits.
        :returns: reparam tensor and params.
        :rtype: torch.Tensor, dict

        """
        z, beta_params = self.reparmeterize(logits, force=force)
        beta_params['conc1_mean'] = torch.mean(beta_params['conc1'])
        beta_params['conc2_mean'] = torch.mean(beta_params['conc2'])
        return z, { 'z': z, 'logits': logits, 'beta':  beta_params }
