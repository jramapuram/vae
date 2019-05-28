from __future__ import print_function
import math
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import Variable

from helpers.utils import float_type, one_hot, ones_like, long_type


class Bernoulli(nn.Module):
    def __init__(self, config, dim=-1):
        super(Bernoulli, self).__init__()
        self.dim = dim
        self.iteration = 0
        self.config = config
        self.input_size = self.config['discrete_size']
        self.output_size = self.config['discrete_size']
        self._setup_anneal_params()

    def _prior_distribution(self, batch_size):
        uniform_probs = float_type(self.config['cuda'])(batch_size, self.output_size).zero_()
        #uniform_probs += 1.0 / self.output_size
        uniform_probs += 0.5
        return torch.distributions.Bernoulli(probs=uniform_probs)

    def prior(self, batch_size, **kwargs):
        return self._prior_distribution(batch_size).sample()

    def _setup_anneal_params(self):
        """setup the annealing parameters; TODO: parameterize

        :returns: None
        :rtype: None

        """
        self.min_temp = 0.3
        self.max_temp = 1.0
        self.tau = 1.0
        self.last_epoch = self.config['epochs']
        self.clip_min = 1e-8

    def cosine_anneal(self):
        """ consine-anneals the temperature

        :returns:
        :rtype:

        """
        if self.training and self.iteration > 0:
            updated_tau = self.min_temp + (self.tau - self.min_temp) \
                * (1 + math.cos(math.pi * -1 / self.last_epoch)) / 2
            self.tau = np.clip(updated_tau, self.clip_min, self.max_temp)

    def reparmeterize(self, logits):
        """ reparamterize the logits

        :param logits: non-activated logits
        :returns: reparameterized and hard outputs
        :rtype: torch.Tensor, torch.Tensor

        """
        relaxed = D.RelaxedBernoulli(temperature=self.tau, logits=logits).rsample()
        hard = relaxed.clone()
        hard[relaxed < 0.5] = 0.0
        hard[relaxed >= 0.5] = 1.0
        return relaxed, hard

    def mutual_info_analytic(self, params, eps=1e-9):
        raise NotImplementedError

    def mutual_info_monte_carlo(self, params, eps=1e-9):
        raise NotImplementedError

    def mutual_info(self, params, eps=1e-9):
        raise NotImplementedError

    @staticmethod
    def _kld_bern_uniform(q_z, dim=-1, eps=1e-9):
        return torch.mul(q_z, torch.log(q_z + eps) - np.log(0.5)) + \
            torch.mul(1.0 - q_z, torch.log(1 - q_z + eps) - np.log(0.5))

    @staticmethod
    def _kld_bern_bern(q_z, p_z, dim=-1, eps=1e-9):
        return torch.mul(q_z, torch.log(q_z + eps) - torch.log(p_z + eps)) + \
            torch.mul(1.0 - q_z, torch.log(1 - q_z + eps) - torch.log(p_z + eps))

    def kl(self, dist_a, prior=None, eps=1e-9):
        if prior is None:  # use standard uniform prior
            # prior = self._prior_distribution(dist_a['discrete']['logits'].shape[0])
            # bern = torch.distributions.Bernoulli(logits=dist_a['discrete']['logits'])
            # return torch.sum(D.kl_divergence(bern, prior), -1)

            return torch.sum(Bernoulli._kld_bern_uniform(
                #torch.log(F.sigmoid(dist_a['discrete']['logits'] + eps) + eps)), -1)
                F.sigmoid(dist_a['discrete']['logits'])), -1)

            # bern = torch.distributions.RelaxedBernoulli(temperature=self.tau, logits=dist_a['discrete']['logits'])
            # return torch.sum(F.kl_div(bern.rsample(), prior.sample(), reduction='none'), -1)

        # we have two distributions provided (eg: VRNN)
        kld_elem = Bernoulli._kld_bern_bern(F.sigmoid(dist_a['discrete']['logits']),
                                            F.sigmoid(prior['discrete']['logits']))
        return torch.sum(kld_elem, -1)

        raise NotImplementedError

    def log_likelihood(self, z, params):
        return D.Bernoulli(logits=params['discrete']['logits']).log_prob(z)

    def forward(self, logits):
        self.cosine_anneal()  # anneal first
        z, z_hard = self.reparmeterize(logits)
        params = {
            'z_hard': z_hard,
            'logits': logits,
            'tau_scalar': self.tau
        }
        self.iteration += 1

        if self.training:
            # return the reparameterization
            # and the params of gumbel
            return z, { 'z': z, 'logits': logits, 'discrete': params }

        return z_hard, { 'z': z, 'logits': logits, 'discrete': params }
