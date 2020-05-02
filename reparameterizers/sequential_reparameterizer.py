from __future__ import print_function
import torch.nn as nn

from copy import deepcopy

import models.vae.reparameterizers as reparameterizers
from helpers.layers import get_encoder


class SequentialReparameterizer(nn.Module):
    def __init__(self, reparam_strs, config):
        """ Sequentially attaches a set of reparameterizations.

        :param config: argparse config
        :returns: SequentialReparameterizer object
        :rtype: object

        """
        super(SequentialReparameterizer, self).__init__()
        self.config = config
        self.reparam_strs = reparam_strs
        self.reparameterizers = nn.ModuleList(self._generate_reparameterizers())
        print('\nreparameterizers: {}\n'.format(self.reparameterizers))
        self.input_size = self.reparameterizers[0].input_size
        self.output_size = self.reparameterizers[-1][-1].output_size \
            if len(self.reparameterizers) > 1 else self.reparameterizers[-1].output_size

    def _get_dense_net(self, input_size, name='dense'):
        """ Internal helper to build a dense network

        :param name: name of the network
        :returns: a nn.Sequential Dense net
        :rtype: nn.Sequential

        """
        config = deepcopy(self.config)
        config['encoder_layer_type'] = 'dense'
        config['input_shape'] = [input_size]
        return get_encoder(name=name, **config)

    def _append_inter_projection(self, reparam, input_size, output_size, name):
        return nn.Sequential(
            self._get_dense_net(input_size=input_size, name=name)(
                output_size=output_size
            ),
            reparam
        )

    def _generate_reparameterizers(self):
        """ Helper to generate all the required reparamterizers

        :returns: a list of reparameterizers
        :rtype: list

        """
        # build the base reparameterizers
        reparam_list = [reparameterizers.get_reparameterizer(reparam)(config=self.config)
                        for reparam in self.reparam_strs]

        # tack on dense networks between them
        input_size = reparam_list[0].output_size
        for i in range(1, len(reparam_list)):
            reparam_list[i] = self._append_inter_projection(reparam_list[i],
                                                            input_size=input_size,
                                                            output_size=reparam_list[i].input_size,
                                                            name='reparam_proj{}'.format(i))
            input_size = reparam_list[i][-1].output_size

        return reparam_list

    def get_reparameterizer_scalars(self):
        """ Return all scalars from the reparameterizers (eg: tau in gumbel)

        :returns: dict of scalars
        :rtype: dict

        """
        reparam_scalar_map = {}
        for i, reparam in enumerate(self.reparameterizers):
            reparam_obj = reparam[-1] if isinstance(reparam, nn.Sequential) else reparam
            current_scalars_map = reparam_obj.get_reparameterizer_scalars()
            for k, v in current_scalars_map.items():
                key_update = "{}{}".format(i, k)
                reparam_scalar_map[key_update] = v

        return reparam_scalar_map

    def mutual_info(self, dists):
        """ Adds all the mutual infos together and returns.

        :param dists: a list of distribution params
        :param priors: a list (or None) of priors, used only in VRNN currently
        :returns: mut-info of shape [batch_size]
        :rtype: torch.Tensor

        """
        dist_params, mi = dists['concat'], None
        for param, reparam in zip(dist_params, self.reparameterizers):
            reparam_obj = reparam[-1] if isinstance(reparam, nn.Sequential) else reparam
            mi = reparam_obj.mutual_info(param) if mi is None else mi + reparam_obj.mutual_info(param)

        return mi

    def kl(self, dists, priors=None):
        """ Adds all the KLs together and returns

        :param dists: a list of distribution params
        :param priors: a list (or None) of priors, used only in VRNN currently
        :returns: kl-divergence of shape [batch_size]
        :rtype: torch.Tensor

        """
        dist_params = dists['sequential']
        priors = [None for _ in range(len(dist_params))] if priors is None else priors
        assert len(priors) == len(dist_params)

        kl = None
        for param, prior, reparam in zip(dist_params, priors, self.reparameterizers):
            reparam_obj = reparam[-1] if isinstance(reparam, nn.Sequential) else reparam
            kl = reparam_obj.kl(param, prior) if kl is None else kl + reparam_obj.kl(param, prior)

        return kl

    def forward(self, logits, force=False):
        """ execute the reparameterization layer-by-layer returning ALL params and last logits.

        :param logits: the input logits
        :returns: last logits, list of params
        :rtype: torch.Tensor, list

        """
        assert force is False, "force not implemented for sequential reparameterizer"
        original_logits = logits

        # Do the first reparam separately because it doesnt have a dense layer
        logits, params = self.reparameterizers[0](logits, force=force)
        params_list = [{**params, 'logits': logits}]

        for reparam in self.reparameterizers[1:]:
            for layer in reparam:  # iterate over the [repram, dense]
                if reparameterizers.is_module_a_reparameterizer(layer):
                    logits, params = layer(logits, force=force)
                else:
                    logits = layer(logits)

            params_list.append({**params, 'logits': logits})

        return logits, {'logits': original_logits, 'sequential': params_list}

    def prior(self, batch_size, **kwargs):
        """ Gen the first prior.

        :param batch_size: the batch size to generate
        :returns: prior sample
        :rtype: torch.Tensor

        """
        prior = self.reparameterizers[0].prior(batch_size)
        if len(self.reparameterizers) > 1:
            for reparam in self.reparameterizers[1:]:
                prior, _ = reparam(prior)

        return prior
