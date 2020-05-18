from __future__ import print_function

import torch
from copy import deepcopy

from .simple_vae import SimpleVAE
import helpers.utils as utils
import helpers.layers as layers
import helpers.distributions as distributions


class VAENoKL(SimpleVAE):
    def loss_function(self, recon_x, x, reparam_map):
        """ VAE with no KL objective. Still uses reparam.

        :param recon_x: the unactivated reconstruction preds.
        :param x: input tensor.
        :returns: loss dict
        :rtype: dict

        """
        nll = distributions.nll(x, recon_x, self.config['nll_type'])
        utils.nan_check_and_break(nll, "nll")
        return {
            'loss': nll,
            'loss_mean': torch.mean(nll),
            'elbo_mean': torch.mean(torch.zeros_like(nll)),
            'nll_mean': torch.mean(nll),
            'kld_mean': torch.mean(torch.zeros_like(nll)),
            'proxy_mean': torch.mean(torch.zeros_like(nll)),
            'mut_info_mean': torch.mean(torch.zeros_like(nll)),
        }


class Autoencoder(SimpleVAE):
    def reparameterize(self, logits, force=False):
        """ No reparameterization for autoencoders.

        :param logits: unactivated encoded logits.
        :param force: unused, kept for API reasons.
        :returns: reparam dict
        :rtype: dict

        """
        return logits, {}

    def generate_synthetic_samples(self, batch_size, **kwargs):
        """ Generates samples with VAE.

        :param batch_size: the number of samples to generate.
        :returns: decoded logits
        :rtype: torch.Tensor

        """
        z_samples = utils.same_type(self.config['half'], self.config['cuda'])(
            batch_size, self.config['continuous_size']
        ).normal_(mean=0.0, std=1.0)
        return self.nll_activation(self.decode(z_samples))

    def build_encoder(self):
        """ helper to build the encoder type

        :returns: an encoder
        :rtype: nn.Module

        """
        return layers.get_encoder(**self.config)(
            output_size=self.config['continuous_size']
        )

    def build_decoder(self, reupsample=True):
        """ helper function to build convolutional or dense decoder

        :returns: a decoder
        :rtype: nn.Module

        """
        dec_conf = deepcopy(self.config)
        if dec_conf['nll_type'] == 'pixel_wise':
            dec_conf['input_shape'][0] *= 256

        decoder = layers.get_decoder(output_shape=dec_conf['input_shape'], **dec_conf)(
            input_size=self.config['continuous_size']
        )

        # append the variance as necessary
        return self._append_variance_projection(decoder)

    def loss_function(self, recon_x, x, **unused_kwargs):
        """ Autoencoder is simple the NLL term in the VAE.

        :param recon_x: the unactivated reconstruction preds.
        :param x: input tensor.
        :returns: loss dict
        :rtype: dict

        """
        nll = distributions.nll(x, recon_x, self.config['nll_type'])
        utils.nan_check_and_break(nll, "nll")
        return {
            'loss': nll,
            'elbo': torch.zeros_like(nll),
            'loss_mean': torch.mean(nll),
            'elbo_mean': 0,
            'nll_mean': torch.mean(nll),
            'kld_mean': 0,
            'kl_beta_scalar': 0,
            'proxy_mean': 0,
            'mut_info_mean': 0,
        }
