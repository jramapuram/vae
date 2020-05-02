from __future__ import print_function

from .abstract_vae import AbstractVAE
from .reparameterizers.concat_reparameterizer import ConcatReparameterizer


class ParallellyReparameterizedVAE(AbstractVAE):
    def __init__(self, input_shape, reparameterizer_strs=["bernoulli", "isotropic_gaussian"], **kwargs):
        """ Implements a parallel (in the case of mixture-reparam) VAE

        :param input_shape: the input shape
        :returns: an object of AbstractVAE
        :rtype: AbstractVAE

        """
        super(ParallellyReparameterizedVAE, self).__init__(input_shape, **kwargs)
        self.reparameterizer_strs = reparameterizer_strs
        self.reparameterizer = ConcatReparameterizer(reparameterizer_strs, self.config)

        # build the encoder and decoder here because of sizing
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def _compute_mi_params(self, recon_x_logits, params_list):
        """ Internal helper to compute the MI params and append to full params

        :param recon_x: reconstruction
        :param params: the original params
        :returns: original params OR param + MI_params
        :rtype: dict

        """
        if self.config['continuous_mut_info'] > 0 or self.config['discrete_mut_info'] > 0:
            _, q_z_given_xhat_params_list = self.posterior(self.nll_activation(recon_x_logits))
            for param, q_z_given_xhat in zip(params_list, q_z_given_xhat_params_list):
                param['q_z_given_xhat'] = q_z_given_xhat

            return params_list

        # base case, no MI
        return params_list
