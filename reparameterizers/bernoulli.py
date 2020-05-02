from __future__ import print_function
import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

from helpers.utils import same_type


class Bernoulli(nn.Module):
    def __init__(self, config, dim=-1):
        super(Bernoulli, self).__init__()
        self.is_discrete = True
        self.dim = dim
        self.iteration = 0
        self.config = config
        self.input_size = self.config['discrete_size']
        self.output_size = self.config['discrete_size']
        self._setup_anneal_params()

    def prior_params(self, batch_size, **kwargs):
        """ Helper to get prior parameters

        :param batch_size: the size of the batch
        :returns: a dictionary of parameters
        :rtype: dict

        """
        uniform_probs = same_type(self.config['half'], self.config['cuda'])(
            batch_size, self.output_size).zero_()
        uniform_probs += 0.5
        return {
            'discrete': {
                'logits': D.Bernoulli(probs=uniform_probs).logits
            }
        }

    def prior_distribution(self, batch_size, **kwargs):
        """ get a torch distrbiution prior

        :param batch_size: size of the prior
        :returns: uniform categorical
        :rtype: torch.distribution

        """
        params = self.prior_params(batch_size, **kwargs)
        return torch.distributions.Bernoulli(logits=params['discrete']['logits'])

    def prior(self, batch_size, **kwargs):
        return self.prior_distribution(batch_size, **kwargs).sample()

    def _setup_anneal_params(self):
        """setup the annealing parameters; TODO: parameterize

        :returns: None
        :rtype: None

        """
        # for regular annealing
        self.anneal_rate = 3e-6
        self.tau0 = 1.0

        # for cos anneal
        self.min_temp = 1e-6
        self.max_temp = 1.0
        self.tau = 1.0
        self.last_epoch = self.config['epochs']
        self.clip_min = 1e-8

    def get_reparameterizer_scalars(self):
        """ Returns any scalars used in reparameterization.

        :returns: dict of scalars
        :rtype: dict

        """
        return {'tau_scalar': self.tau}

    def anneal(self, anneal_interval=10):
        """ Helper to anneal the temperature.

        :param anneal_interval: the interval to employ annealing.
        :returns: None
        :rtype: None

        """
        if self.training \
           and self.iteration > 0 \
           and self.iteration % anneal_interval == 0:

            # smoother annealing
            rate = -self.anneal_rate * self.iteration
            self.tau = np.maximum(self.tau0 * np.exp(rate),
                                  self.min_temp)
            # hard annealing
            # self.tau = np.maximum(0.9 * self.tau, self.min_temp)

    def cosine_anneal(self):
        """ consine-anneals the temperature

        :returns:
        :rtype:

        """
        if self.training and self.iteration > 0:
            updated_tau = self.min_temp + (self.tau - self.min_temp) \
                * (1 + math.cos(math.pi * -1 / self.last_epoch)) / 2
            self.tau = np.clip(updated_tau, self.clip_min, self.max_temp)

    def _compute_hard(self, relaxed):
        """ helper to get the hard version of relaxed

        :param relaxed: the relaxed estimate
        :returns: hard bernoulli with 0s and 1s
        :rtype: torch.Tensor

        """

        hard = relaxed.clone()
        hard[relaxed < 0.5] = 0.0
        hard[relaxed >= 0.5] = 1.0
        hard_diff = hard - relaxed  # sub the relaxed tensor backprop path
        return hard_diff.detach() + relaxed  # add back for 0 effect, keeping bp path

    def reparmeterize(self, logits):
        """ reparamterize the logits

        :param logits: non-activated logits
        :returns: reparameterized and hard outputs
        :rtype: torch.Tensor, torch.Tensor

        """
        relaxed = D.RelaxedBernoulli(temperature=self.tau, logits=logits).rsample()
        hard = self._compute_hard(relaxed)
        return relaxed, hard

    def mutual_info_analytic(self, params, eps=1e-9):
        raise NotImplementedError

    def mutual_info_monte_carlo(self, params, eps=1e-9):
        raise NotImplementedError

    def mutual_info(self, params, eps=1e-9):
        raise NotImplementedError

    @staticmethod
    def _kld_bern_bern(q_dist, p_dist, eps=1e-6):
        """ Stable bern-bern KL from: https://github.com/pytorch/pytorch/issues/15288

        :param q_dist: the D.Bernoulli q distribution
        :param p_dist: the D.Bernoulli p distribution
        :param eps: numerical precision
        :returns: kl-divergence (non-reduced)
        :rtype: torch.Tensor

        """
        q1 = q_dist.probs
        q0 = 1 - q1
        p1 = p_dist.probs
        p0 = 1 - p1

        logq1 = (q1 + eps).log()
        logq0 = (q0 + eps).log()
        logp1 = (p1 + eps).log()
        logp0 = (p0 + eps).log()

        kldiv_1 = q1*(logq1 - logp1)
        kldiv_0 = q0*(logq0 - logp0)
        return kldiv_1 + kldiv_0

    def kl(self, dist_a, prior=None, eps=1e-6):
        """ Evaluates KL between two distributions or against a naive prior if not provided

        :param dist_a: the dist in KL(dist_a, prior)
        :param prior: the (optional) prior in KL(dist_a, prior)
        :param eps: tolerance for numerical precision
        :returns: [batch_size] size tensor
        :rtype: torch.Tensor

        """
        if prior is None:  # use standard uniform prior
            prior_probs = torch.zeros_like(dist_a['discrete']['logits']) + 0.5
            prior = D.Bernoulli(probs=prior_probs)
            bern = D.Bernoulli(logits=dist_a['discrete']['logits'])
            kld_elem = self._kld_bern_bern(bern, prior)
            return torch.sum(kld_elem, -1)

        kld_elem = self._kld_bern_bern(D.Bernoulli(logits=dist_a['discrete']['logits']),
                                       D.Bernoulli(logits=prior['discrete']['logits']))
        return torch.sum(kld_elem, -1)

    def log_likelihood(self, z, params):
        return D.Bernoulli(logits=params['discrete']['logits']).log_prob(z)

    def forward(self, logits, force=False):
        # self.cosine_anneal()  # anneal first
        self.anneal()           # anneal first

        z, z_hard = self.reparmeterize(logits)
        params = {
            'z_hard': z_hard,
            'logits': logits,
            'tau_scalar': self.tau
        }
        self.iteration += 1

        if self.training or force:
            # return the reparameterization and the params of gumbel
            return z, {'z': z, 'logits': logits, 'discrete': params}

        return z_hard, {'z': z, 'logits': logits, 'discrete': params}
