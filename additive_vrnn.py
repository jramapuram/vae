import torch
import functools
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from copy import deepcopy

from .abstract_vae import AbstractVAE
from .vrnn import VRNN, VRNNMemory
from .reparameterizers.gumbel import GumbelSoftmax
from .reparameterizers.mixture import Mixture
from .reparameterizers.beta import Beta
from .reparameterizers.isotropic_gaussian import IsotropicGaussian
from helpers.distributions import nll_activation as nll_activation_fn
from helpers.distributions import nll as nll_fn
from helpers.layers import get_encoder, get_decoder, Identity, EMA
from helpers.utils import eps as eps_fn, add_noise_to_imgs, float_type
from helpers.utils import same_type, zeros_like, expand_dims, \
    zeros, nan_check_and_break


class AdditiveVRNN(VRNN):
    def forward(self, input_t):
        """ Multi-step forward pass for AdditiveVRNN.

        :param input_t: input tensor or list of tensors
        :returns: final output tensor
        :rtype: torch.Tensor

        """
        decoded, params = [], []
        batch_size = input_t.shape[0] if isinstance(input_t, torch.Tensor) else input_t[0].shape[0]

        self.memory.init_state(batch_size, input_t.is_cuda) # always re-init state at first step.
        for i in range(self.config['max_time_steps']):
            if isinstance(input_t, list):  # if we have many inputs as a list
                decode_t, params_t = self.step(input_t[i])
                decode_activated_t = nll_activation_fn(decode_t, self.config['nll_type'])
                input_t[-1] = decode_activated_t if i == 0 else decode_activated_t + input_t[i]
            else:                          # single input encoded many times
                decode_t, params_t = self.step(input_t)
                decode_activated_t = nll_activation_fn(decode_t, self.config['nll_type'])
                input_t = decode_activated_t if i == 0 else decode_activated_t + input_t

            if self.training and i == 0:
                self.aggregate_posterior['rnn_hidden_state_h'](self.memory.get_state()[0])
                self.aggregate_posterior['rnn_hidden_state_c'](self.memory.get_state()[1])

            # append mutual information if requested
            params_t = self._compute_mi_params(decode_t, params_t)

            # add the params and the input to the list
            decoded.append(decode_t)
            params.append(params_t)

        self.memory.clear()                # clear memory to prevent perennial growth
        return decoded, params

    def generate_synthetic_samples(self, batch_size, **kwargs):
        """ generate batch_size samples.

        :param batch_size: the size of the batch to generate
        :returns: generated tensor
        :rtype: torch.Tensor

        """
        if 'reset_state' in kwargs and kwargs['reset_state']:
            self.memory.init_state(batch_size, cuda=self.config['cuda'],
                                   override_noisy_state=True)

        final_state, z_prior_t = self._get_prior_and_state(batch_size, **kwargs)

        # grab final state and prior samples & encode them through feature extractor
        final_state, z_prior_t = self._get_prior_and_state(batch_size, **kwargs)
        phi_z_t = self.phi_z(z_prior_t)

        # run the first step of the decoding process using the prior
        dec_input_t = torch.cat([phi_z_t, final_state], -1)
        dec_output_t = self._decode_pixelcnn_or_normal_and_activate(dec_input_t)
        decoded_list = [dec_output_t.clone()]

        # iterate max_time_steps -1  times using the output from above
        for _ in range(self.config['max_time_steps'] - 1):
            dec_output_tp1, _ = self.step(dec_output_t, **kwargs)
            dec_output_tp1 = nll_activation_fn(dec_output_tp1, self.config['nll_type'])
            dec_output_t = dec_output_t + dec_output_tp1
            decoded_list.append(dec_output_t.clone())

        return torch.cat(decoded_list, 0)

    def loss_function(self, recon_x_container, x_container, params_map):
        """ evaluates the loss of the model by simply summing individual losses

        :param recon_x_container: the reconstruction container
        :param x_container: the input container
        :param params_map: the params dict
        :returns: the mean-reduced aggregate dict
        :rtype: dict

        """
        assert len(recon_x_container) == len(params_map)

        # case where only 1 data sample, but many posteriors
        # the additive VRNN differs here in the sense that it uses only the last reconstr
        if not isinstance(x_container, list) and len(x_container) != len(recon_x_container):
            x_container = [x_container.clone() for _ in range(len(recon_x_container))]
            recon_x_container = [recon_x_container[-1].clone() for _ in range(len(recon_x_container))]

        # aggregate the loss many and return the mean of the map
        loss_aggregate_map = None
        for recon_x, x, params in zip(recon_x_container, x_container, params_map):
            loss_t = super(VRNN, self).loss_function(recon_x, x, params)
            loss_aggregate_map = self._add_loss_map(loss_t, loss_aggregate_map)

        return self._mean_map(loss_aggregate_map)
