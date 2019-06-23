from __future__ import print_function

import torch
import torch.distributions as D
import torch.nn.functional as F

from functools import partial

from helpers.distributions import nll as nll_fn
from helpers.layers import str_to_activ_module
from helpers.utils import nan_check_and_break
from .reparameterizers.gumbel import GumbelSoftmax
from .reparameterizers.mixture import Mixture
from .reparameterizers.beta import Beta
from .reparameterizers.bernoulli import Bernoulli
from .reparameterizers.isotropic_gaussian import IsotropicGaussian
from .abstract_vae import AbstractVAE


class SymmetricSimpleVAE(AbstractVAE):
    def __init__(self, input_shape, **kwargs):
        """ Implements a parallel (in the case of mixture-reparam) VAE

        :param input_shape: the input shape
        :returns: an object of AbstractVAE
        :rtype: AbstractVAE

        """
        super(SymmetricSimpleVAE, self).__init__(input_shape, **kwargs)
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
        """ Produces ELBO, handles mutual info and proxy loss terms too.

        :param recon_x: the unactivated reconstruction preds.
        :param x: input tensor.
        :param params: the dict of reparameterization.
        :param mut_info: the calculated mutual info.
        :returns: loss dict
        :rtype: dict

        """
        if self.config['decoder_layer_type'] == 'pixelcnn':
            x = (x - .5) * 2.

        nll = nll_fn(x, recon_x, self.config['nll_type'])
        nan_check_and_break(nll, "nll")
        prior = {
            'gaussian': {'mu': torch.zeros_like(params['gaussian']['mu']),
                         'logvar': torch.ones_like(params['gaussian']['logvar'])}
        }
        kld = self.kld(params)
        nan_check_and_break(kld, "kld")
        elbo = nll + kld  # save the base ELBO, but use the beta-vae elbo for the full loss

        # add the proxy loss if it exists
        proxy_loss = self.reparameterizer.proxy_layer.loss_function() \
            if hasattr(self.reparameterizer, 'proxy_layer') else torch.zeros_like(elbo)

        # handle the mutual information term
        mut_info = self.mut_info(params, x.size(0))

        # calculate the JSD (symmetric KL) and final loss
        jsd = 0.5 * self.config['kl_beta'] * (kld + self.reparameterizer.kl(prior, params))
        loss = (nll + jsd) - mut_info

        return {
            'loss': loss,
            'loss_mean': torch.mean(loss),
            'elbo_mean': torch.mean(elbo),
            'nll_mean': torch.mean(nll),
            'kld_mean': torch.mean(kld),
            'proxy_mean': torch.mean(proxy_loss),
            'mut_info_mean': torch.mean(mut_info)
        }
