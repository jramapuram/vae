from __future__ import print_function

import torch

from .simple_vae import SimpleVAE
from helpers.utils import nan_check_and_break, zeros, same_type, is_half
from helpers.distributions import nll as nll_fn
from helpers.layers import get_decoder, get_encoder, str_to_activ_module


class VAENoKL(SimpleVAE):
    def loss_function(self, recon_x, x, reparam_map):
        """ VAE with no KL objective. Still uses reparam.

        :param recon_x: the unactivated reconstruction preds.
        :param x: input tensor.
        :returns: loss dict
        :rtype: dict

        """
        nll = nll_fn(x, recon_x, self.config['nll_type'])
        nan_check_and_break(nll, "nll")
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
    def reparameterize(self, logits):
        """ Reparameterize the logits and returns a dict.

        :param logits: unactivated encoded logits.
        :returns: reparam dict
        :rtype: dict

        """
        return logits, {}


    def build_encoder(self):
        """ helper to build the encoder type

        :returns: an encoder
        :rtype: nn.Module

        """
        conv_layer_types = ['conv', 'coordconv', 'resnet']
        input_shape = [self.input_shape[0], 0, 0] if self.config['encoder_layer_type'] \
            in conv_layer_types else self.input_shape

        # return the encoder
        return get_encoder(self.config)(input_shape=input_shape,
                                        output_size=self.config['latent_size'],
                                        activation_fn=self.activation_fn)

    def build_decoder(self, reupsample=True):
        """ helper function to build convolutional or dense decoder

        :returns: a decoder
        :rtype: nn.Module

        """
        if self.config['decoder_layer_type'] == "pixelcnn":
            assert self.config['nll_type'] == "disc_mix_logistic" \
                or self.config['nll_type'] == "log_logistic_256", \
                "pixelcnn only works with disc_mix_logistic or log_logistic_256"

        decoder = get_decoder(self.config, reupsample)(input_size=self.config['latent_size'],
                                                       output_shape=self.input_shape,
                                                       activation_fn=self.activation_fn)
        # append the variance as necessary
        return self._append_variance_projection(decoder)

    def generate_synthetic_samples(self, batch_size, **kwargs):
        """ Generates samples with VAE.

        :param batch_size: the number of samples to generate.
        :returns: decoded logits
        :rtype: torch.Tensor

        """
        # return zeros([batch_size] + self.input_shape, cuda=self.config['cuda'])
        z_mu = self.aggregate_posterior.ema_val
        z_logvar = same_type(is_half(z_mu), z_mu.is_cuda)(
            z_mu.size()
        ).normal_()
        z_samples = z_mu + z_logvar
        if self.config['decoder_layer_type'] == "pixelcnn":
            # decode the synthetic samples through the base decoder
            decoded = self.decoder(z_samples)
            return self.generate_pixel_cnn(batch_size, decoded)

        # in the normal case just decode and activate
        return self.nll_activation(self.decode(z_samples))

    def loss_function(self, recon_x, x, reparam_map):
        """ Autoencoder is simple the NLL term in the VAE.

        :param recon_x: the unactivated reconstruction preds.
        :param x: input tensor.
        :returns: loss dict
        :rtype: dict

        """
        nll = nll_fn(x, recon_x, self.config['nll_type'])
        nan_check_and_break(nll, "nll")
        return {
            'loss': nll,
            'loss_mean': torch.mean(nll),
            'elbo_mean': torch.mean(torch.zeros_like(nll)),
            'nll_mean': torch.mean(nll),
            'kld_mean': torch.mean(torch.zeros_like(nll)),
            'proxy_mean': torch.mean(torch.zeros_like(nll)),
            'mut_info_mean': torch.mean(torch.zeros_like(nll)),
        }
