from __future__ import print_function

import torch
import torch.nn as nn
from functools import partial

from helpers.layers import str_to_activ_module, get_decoder
from .reparameterizers.gumbel import GumbelSoftmax
from .reparameterizers.mixture import Mixture
from .reparameterizers.beta import Beta
from .reparameterizers.bernoulli import Bernoulli
from .reparameterizers.isotropic_gaussian import IsotropicGaussian
from .abstract_vae import AbstractVAE


class MSGVAE(AbstractVAE):
    def __init__(self, input_shape, **kwargs):
        """ Implements a VAE which decodes many samples and averages outputs.

        :param input_shape: the input shape
        :returns: an object of MSG-VAE
        :rtype: MSGVAE

        """
        super(MSGVAE, self).__init__(input_shape, **kwargs)
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

        # build the gates
        self.gates = nn.ModuleList([get_decoder(self.config, reupsample=True, name='gate_{}'.format(i))(
                                                input_size=self.reparameterizer.output_size,
                                                output_shape=self.input_shape,
                                                activation_fn=self.activation_fn)
                                    for i in range(self.config['max_time_steps'])])

        # over-ride the reparam prior
        self.single_prior = self.reparameterizer.prior
        self.reparameterizer.prior = self._prior_override

    def _prior_override(self, batch_size, **kwargs):
        """ Helper to generate many samples from the true prior

        :param batch_size: the batch size to generate samples for
        :returns: a list of priors
        :rtype: [torch.Tensor]

        """
        return [self.single_prior(batch_size, **kwargs) for _ in range(self.config['max_time_steps'])]

    def has_discrete(self):
        """ Returns true if there is a discrete reparameterization.

        :returns: True/False
        :rtype: bool

        """
        return isinstance(self.reparameterizer, (GumbelSoftmax, Mixture))

    def kld(self, dist_list):
        """ KL-Divergence of the distribution dict and the prior of that distribution.
            NOTE: we use the last one because we calculate the analytical KL divergence
                  which only necessisitates the parameters of the distribution.

        :param dist_list: the list of distributions.
        :returns: tensor that is of dimension batch_size
        :rtype: torch.Tensor

        """
        return self.reparameterizer.kl(dist_list[-1])

    def reparameterize(self, logits):
        """ Reparameterize the logits and returns a dict.

        :param logits: unactivated encoded logits.
        :returns: reparam dict
        :rtype: dict

        """
        z_list, params_list = [], []
        for _ in range(self.config['max_time_steps']):
            z, params = self.reparameterizer(logits)
            z_list.append(z); params_list.append(params)

        return z_list, params_list

        # return zip(*[self.reparameterizer(logits)
        #              for _ in range(self.config['max_time_steps'])])

    def decode(self, z):
        """ Decode a set of latent z back to x_mean.

        :param z: the latent tensor.
        :returns: decoded logits (unactivated).
        :rtype: torch.Tensor

        """
        assert isinstance(z, (list, tuple)), "expecting a tuple or list"
        gate_encodes = [torch.sigmoid(g(z_i)) for g, z_i in zip(self.gates, z)]
        return torch.mean(torch.cat([(g_i * self.decoder(z_i.contiguous())).unsqueeze(0)
                                     for z_i, g_i in zip(z, gate_encodes)], 0), 0)
        # if self.training:
        #     return torch.mean(torch.cat([(g_i * self.decoder(z_i.contiguous())).unsqueeze(0)
        #                                  for z_i, g_i in zip(z, self.gate_encodes)], 0), 0)


        # return torch.mean(torch.cat([self.decoder(z_i.contiguous()).unsqueeze(0) for z_i in z], 0), 0)
