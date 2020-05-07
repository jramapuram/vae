from __future__ import print_function
import torch
from copy import deepcopy
import torch.nn.functional as F

import helpers.layers as layers
from .simple_vae import SimpleVAE


class ClassConditionedPriorVAE(SimpleVAE):
    def __init__(self, input_shape, **kwargs):
        """ Implements a Simple VAE with a conditioned latent variable

        :param input_shape: the input shape
        :returns: an object of AbstractVAE
        :rtype: AbstractVAE

        """
        super(ClassConditionedPriorVAE, self).__init__(input_shape, **kwargs)
        assert 'output_size' in self.config, "need to specify label size in argparse"

        # build a projector for the labels to the learned prior
        conf = deepcopy(self.config)
        conf['encoder_layer_type'] = 'dense'
        conf['input_shape'] = [conf['output_size']]  # the size of the labels from the dataset
        self.prior_mlp = layers.get_encoder(**conf)(
            output_size=self.reparameterizer.input_size
        )

    def generate_synthetic_samples_from_labels(self, batch_size, labels, **kwargs):
        """ Generates samples using provided labels

        :param batch_size: the number of samples to generate.
        :returns: decoded logits
        :rtype: torch.Tensor

        """
        prior_logits = self._process_labels(labels, torch.float32)
        z_samples, _ = self.reparameterizer(prior_logits, force=kwargs.get('force', False))

        # in the normal case just decode and activate
        return self.nll_activation(self.decode(z_samples))

    def _clamp_variance(self, logits):
        """ clamp the variance when using a gaussian dist.

        :param logits: the un-activated logits
        :returns: the logits, clamped
        :rtype: torch.Tensor

        """
        if self.config['reparam_type'] == 'isotropic_gaussian':
            feat_size = logits.size(-1)
            return torch.cat(
                [logits[:, 0:feat_size//2],
                 torch.sigmoid(logits[:, feat_size//2:])],
                -1)
        elif self.config['reparam_type'] == 'mixture':
            feat_size = self.reparameterizer.num_continuous_input
            return torch.cat(
                [logits[:, 0:feat_size//2],                         # mean
                 torch.sigmoid(logits[:, feat_size//2:feat_size]),  # clamped var
                 logits[:, feat_size:]],                       # discrete
                -1)
        else:
            return logits

    def generate_synthetic_samples(self, batch_size, **kwargs):
        """ Generates samples with VAE.

        :param batch_size: the number of samples to generate.
        :returns: decoded logits
        :rtype: torch.Tensor

        """
        gen = super(ClassConditionedPriorVAE, self).generate_synthetic_samples(batch_size, **kwargs)
        if 'labels' in kwargs:  # Add label based generations
            label_gen = self.generate_synthetic_samples_from_labels(batch_size, **kwargs)
        else:  # enumerate batch_size possible one-hot positions
            enumerated_labels = torch.arange(
                self.config['output_size'], device='cuda:0' if self.config['cuda'] else 'cpu')
            label_gen = self.generate_synthetic_samples_from_labels(batch_size,
                                                                    labels=enumerated_labels,
                                                                    **kwargs)

        gen = torch.cat([gen, label_gen], 0)
        return gen

    def _process_labels(self, labels, dtype):
        """Simple helper to process the labels, converting them to one-hot vectors and running the DNN.

        :param labels: int labels [B] or feature labels [B, F]
        :param dtype: output dtype of labels (useful for fp16)
        :param device: target device, since one_hot
        :returns: prior_logits
        :rtype: torch.Tensor

        """
        device = labels.device
        assert labels.dim() <= 2, "unknown labels with dimension {} received".format(labels.dim())
        if labels.dim() == 1:
            labels = F.one_hot(labels, self.config['output_size'])

        # convert to appropriate type and place on correct device
        labels = labels.type(dtype).to(device)

        # handle the prior projection and reparam
        prior_logits = self.prior_mlp(labels)
        prior_logits = self._clamp_variance(prior_logits)  # XXX
        return prior_logits

    def reparameterize(self, logits, labels, force=False):
        """ Reparameterize the logits and returns a dict.

        :param logits: unactivated encoded logits.
        :param force: force reparameterize the distributions
        :returns: reparam dict
        :rtype: dict

        """
        z_posterior, params_posterior = self.reparameterizer(logits, force=force)
        assert labels is not None, "labels needed for class-conditioned vae."

        # One hot labels if we need to.
        prior_logits = self._process_labels(labels, logits.dtype)
        z_prior, params_prior = self.reparameterizer(prior_logits, force=force)

        # add both to the params dict and return
        updated_params = {
            'labels': labels,
            'logits': logits,
            'prior': params_prior,
            'posterior': params_posterior,
        }
        return z_posterior, updated_params

    def kld(self, dist):
        """ KL divergence between dist_a and prior as well as constrain prior to hyper-prior

        :param dist: the distribution map
        :returns: kl divergence
        :rtype: torch.Tensor

        """
        prior_kl = self.reparameterizer.kl(dist['prior'])  \
            if self.config['use_prior_kl'] is True else 0
        return self.reparameterizer.kl(dist['posterior'], dist['prior']) + prior_kl
