import torch
import torch.nn as nn
import numpy as np

import models.vae.flows as flows


class FlowReparameterizer(nn.Module):
    """"A fully flow based reparameterizer."""

    def __init__(self, config):
        """Builds a continuous --> flow reparameterizer.

        :param config: argparse config
        :returns: reparameterizer
        :rtype: nn.Module

        """
        super(FlowReparameterizer, self).__init__()
        flow_type = config['flow_type']
        continuous_size = config['continuous_size']
        latent_size = config['latent_size']

        # Build the flow
        self.num_reads = config['memory_read_steps']
        if flow_type == 'maf':
            self.flow = flows.build_maf_flow(num_inputs=continuous_size, num_hidden=latent_size,
                                             num_cond_inputs=None, num_blocks=5, activation_str='tanh')
        elif flow_type == 'maf_split':
            self.flow = flows.build_maf_split_flow(num_inputs=continuous_size, num_hidden=latent_size,
                                                   num_cond_inputs=None, num_blocks=5,
                                                   s_activation_str='tanh', t_activation_str='relu')
        elif flow_type == 'maf_split_glow':
            self.flow = flows.build_maf_split_glow_flow(num_inputs=continuous_size, num_hidden=latent_size,
                                                        num_cond_inputs=None, num_blocks=5,
                                                        s_activation_str='tanh', t_activation_str='relu')
        elif flow_type == 'glow':
            self.flow = flows.build_glow_flow(num_inputs=continuous_size, num_hidden=latent_size,
                                              num_cond_inputs=None, num_blocks=5,
                                              activation_str=config['encoder_activation'],
                                              normalization_str=config['dense_normalization'],
                                              layer_modifier=config['encoder_layer_modifier'],
                                              cuda=config['cuda'])
        elif flow_type == 'realnvp':
            self.flow = flows.build_realnvp_flow(num_inputs=continuous_size, num_hidden=latent_size,
                                                 num_cond_inputs=None, num_blocks=5,
                                                 s_activation_str='tanh', t_activation_str='relu',
                                                 normalization_str=config['dense_normalization'],
                                                 cuda=config['cuda'])
        self.input_size = continuous_size
        self.output_size = continuous_size

    def prior(self, batch_size, scale_var=1.0, **kwargs):
        noise = torch.Tensor(batch_size, self.flow.num_inputs).normal_(std=scale_var)
        gen = self.flow.sample(num_samples=batch_size, noise=noise)
        return gen

    def get_reparameterizer_scalars(self):
        """ Returns any scalars used in reparameterization.

        :returns: dict of scalars
        :rtype: dict

        """
        return {}

    def kl(self, dist_a, prior=None):
        recon, logdet = dist_a['recon_x'], dist_a['logdet']
        # print('recon = {} | logdet = {}'.format(recon.shape, logdet.shape))
        nll = (-0.5 * recon.pow(2) - 0.5 * np.log(2 * np.pi)).sum(-1)
        # batch_size = dist_a['recon_x'].shape[0]
        # nll = torch.sum(F.mse_loss(input=dist_a['recon_x'].view(batch_size, num_reads, -1),
        #                            target=torch.zeros_like(dist_a['recon_x']).view(batch_size, num_reads, -1),
        #                            reduction='none'), dim=-1)
        # mean = dist_a['recon_x']
        # nll = - torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean)).log_prob(mean).sum(-1)
        # print("[POST]nll = {} | logdet = {}".format(nll.shape, logdet.shape))
        return torch.mean(-nll - logdet, -1)

    def forward(self, logits, force=False):
        batch_size, num_reads, input_size = logits.shape
        flow, logdet = self.flow(logits.contiguous().view(-1, input_size))
        flow = flow.view([-1, num_reads, input_size])
        logdet = logdet.view([-1, num_reads])
        return flow, {'recon_x': flow, 'logdet': logdet}
