from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn

import models.vae.reparameterizers as reparameterizers


class ConcatReparameterizer(nn.Module):
    def __init__(self, reparam_strs, config):
        """ Concatenates a set of reparameterizations.

        :param config: argparse config
        :returns: ConcatReparameterizer object
        :rtype: object

        """
        super(ConcatReparameterizer, self).__init__()
        self.config = config
        self.reparam_strs = reparam_strs
        self.reparameterizers = nn.ModuleList(self._generate_reparameterizers())
        print('\nreparameterizers: {}\n'.format(self.reparameterizers))
        self.input_size = int(np.sum([r.input_size for r in self.reparameterizers]))
        self.output_size = int(np.sum([r.output_size for r in self.reparameterizers]))

        # to enumerate over for reparameterization
        self._input_sizing = [0] + list(np.cumsum([r.input_size for r in self.reparameterizers]))

    def _generate_reparameterizers(self):
        """ Helper to generate all the required reparamterizers

        :returns: a list of reparameterizers
        :rtype: list

        """
        # build the base reparameterizers
        return [reparameterizers.get_reparameterizer(reparam)(config=self.config)
                for reparam in self.reparam_strs]

    def get_reparameterizer_scalars(self):
        """ Return all scalars from the reparameterizers (eg: tau in gumbel)

        :returns: dict of scalars
        :rtype: dict

        """
        reparam_scalar_map = {}
        for i, reparam in enumerate(self.reparameterizers):
            current_scalars_map = reparam.get_reparameterizer_scalars()
            for k, v in current_scalars_map.items():
                key_update = "{}{}".format(i, k)
                reparam_scalar_map[key_update] = v

        return reparam_scalar_map

    def mutual_info(self, dists):
        """ concatenates all the mutual infos together and returns.

        :param dists: a list of distribution params
        :param priors: a list (or None) of priors, used only in VRNN currently
        :returns: mut-info of shape [batch_size]
        :rtype: torch.Tensor

        """
        mi = None
        for param, reparam in zip(dists, self.reparameterizers):
            mi = reparam.mutual_info(param) if mi is None else mi + reparam.mutual_info(param)

        return mi

    def kl(self, dists, priors=None):
        """ concatenates all the KLs together and returns

        :param dists: a list of distribution params
        :param priors: a list (or None) of priors, used only in VRNN currently
        :returns: kl-divergence of shape [batch_size]
        :rtype: torch.Tensor

        """
        priors = [None for _ in range(len(dists))] if priors is None else priors
        assert len(priors) == len(dists)

        kl = None
        for param, prior, reparam in zip(dists, priors, self.reparameterizers):
            kl = reparam.kl(param, prior) if kl is None else kl + reparam.kl(param, prior)

        return kl

    def forward(self, logits, force=False):
        """ execute the reparameterization layer-by-layer returning ALL params and last reparam logits.

        :param logits: the input logits
        :returns: concat reparam, list of params
        :rtype: torch.Tensor, list

        """
        params_list = []
        reparameterized = []

        # determine which dimension we slice over
        dim_map = {
            2: -1,  # [B, F]
            3: -1,  # [B, T, F] --> TODO: do we want to slice time or feature?
            4: 1    # [B, C, H, W]
        }
        assert logits.dim() in dim_map, "unknown number of dims for concat reparam"
        dim2slice = dim_map[logits.dim()]

        for i, (begin, end) in enumerate(zip(self._input_sizing, self._input_sizing[1:])):
            # print("reparaming from {} to {} for {}-th reparam which is a {} with shape {}".format(
            #     begin, end, i, self.reparameterizers[i], logits[:, begin:end].shape))
            logits_i = torch.narrow(logits, dim2slice, begin, end-begin)
            reparameterized_i, params = self.reparameterizers[i](logits_i, force=force)
            reparameterized.append(reparameterized_i.clone())
            params_list.append({**params, 'logits': logits[:, begin:end]})

        return torch.cat(reparameterized, -1), params_list

    def prior(self, batch_size, **kwargs):
        """ Gen the first prior.

        :param batch_size: the batch size to generate
        :returns: prior sample
        :rtype: torch.Tensor

        """
        return torch.cat([r.prior(batch_size) for r in self.reparameterizers], -1)
