from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.autograd import Variable
from collections import OrderedDict

import helpers.utils as utils
import helpers.layers as layers
import helpers.distributions as distributions


class VarianceProjector(nn.Module):
    def __init__(self, nll_type_str):
        """ A single scalar (learnable) variance.

        :param nll_type_str: string describing negative log-likelihood type.
        :returns: object
        :rtype: object

        """
        super(VarianceProjector, self).__init__()

        # build the sequential layer
        if distributions.nll_has_variance(nll_type_str):
            self.register_parameter(
                "variance_scalar",
                nn.Parameter(torch.zeros(1))
            )

    def forward(self, x):
        if hasattr(self, 'variance_scalar'):
            if x.dim() == 4:
                # Expand the scalar variance to a matrix diagonal covariance and broadcast to 3 channels.
                covariance = torch.eye(x.shape[-2], x.shape[-1], device=x.device) * self.variance_scalar
                return torch.cat([x, covariance.expand_as(x)], 1)
            elif x.dim() == 2:
                # [B, F], expand to F
                return torch.cat([x, self.variance_scalar.expand_as(x)], 1)

            raise Exception("unknown dims for Variance projector")

        return x


class AbstractVAE(nn.Module):
    def __init__(self, input_shape, **kwargs):
        """ Abstract base class for VAE.

        :param input_shape: the input tensor shape
        :returns: instantiation of object
        :rtype: object

        """
        super(AbstractVAE, self).__init__()
        self.input_shape = input_shape
        self.is_color = input_shape[0] > 1
        self.chans = 3 if self.is_color else 1
        self.config = kwargs['kwargs']

        # keep track of ammortized posterior
        self.aggregate_posterior = layers.EMA(0.9)
        self.polyak_ema = layers.EMA(self.config['polyak_ema']) if self.config['polyak_ema'] > 0 else None

        # Setup the cyclic annealing object if required.
        self.kl_annealer = self.build_kl_annealer()

    def get_reparameterizer_scalars(self):
        """ return the reparameterization scalars (eg: tau in gumbel)

        :returns: a dict of scalars
        :rtype: dict

        """
        return self.reparameterizer.get_reparameterizer_scalars()

    def build_kl_annealer(self):
        """Helper to build a KL annealer (if requred in argparse)."""
        kl_annealer = None
        klc = self.config['kl_annealing_cycles']
        if klc is not None and klc > 0:
            ten_percent_of_epochs_as_steps = int(self.config['epochs'] * 0.1) * self.config['steps_per_train_epoch']
            total_cycles = self.config['total_train_steps'] / self.config['kl_annealing_cycles']
            # print("steps_per_epoch = {} | total_steps = {} | total_cycles = {} | 10% steps = {}".format(
            #     self.config['steps_per_epoch'],
            #     self.config['total_steps'],
            #     total_cycles, ten_percent_of_epochs_as_steps))
            # Linear warmup with fixed rate; generally performs worse than cosine-annealing below.
            # self.kl_annealer = layers.LinearWarmupWithFixedInterval(
            #     fixed_steps=int(np.ceil((total_cycles + 1) * 0.3)),  # Use 90% for base kl-beta
            #     warmup_steps=int(np.floor((total_cycles + 1) * 0.7))  # Use 10% for linear warmup
            # )
            kl_annealer = layers.LinearWarmupWithCosineAnnealing(
                decay_steps=int(total_cycles * 0.9),                        # Use 90% for cos-anneal.
                warmup_steps=int(total_cycles * 0.1),                       # Use 10% for linear warmup.
                total_steps=self.config['total_train_steps'],                     # Total steps for model.
                constant_for_last_k_steps=ten_percent_of_epochs_as_steps    # Constant steps at end.
            )
            print("\nKL-Annealer: {}\n".format(kl_annealer))

        return kl_annealer

    def build_encoder(self):
        """ helper to build the encoder type

        :returns: an encoder
        :rtype: nn.Module

        """
        encoder = layers.get_encoder(**self.config)(
            output_size=self.reparameterizer.input_size
        )
        print('encoder has {} parameters\n'.format(utils.number_of_parameters(encoder) / 1e6))
        return torch.jit.script(encoder) if self.config['jit'] else encoder

    def build_decoder(self, reupsample=True):
        """ helper function to build convolutional or dense decoder

        :returns: a decoder
        :rtype: nn.Module

        """
        dec_conf = deepcopy(self.config)
        if dec_conf['nll_type'] == 'pixel_wise':
            dec_conf['input_shape'][0] *= 256

        decoder = layers.get_decoder(output_shape=dec_conf['input_shape'], **dec_conf)(
            input_size=self.reparameterizer.output_size
        )
        print('decoder has {} parameters\n'.format(utils.number_of_parameters(decoder) / 1e6))

        # append the variance as necessary
        decoder = self._append_variance_projection(decoder)
        return torch.jit.script(decoder) if self.config['jit'] else decoder

    def _append_variance_projection(self, decoder):
        """ Appends a decoder variance for gaussian, etc.

        :param decoder: the nn.Module
        :returns: appended variance projector to decoder
        :rtype: nn.Module

        """
        if distributions.nll_has_variance(self.config['nll_type']):
            # add the variance projector (if we are in that case for the NLL)
            # warnings.warn("\nCurrently variance is not being added to p(x|z)\ --> using mean. \n")
            print("adding variance projector for {} log-likelihood".format(self.config['nll_type']))
            decoder = nn.Sequential(
                decoder,
                VarianceProjector(self.config['nll_type'])
            )

        return decoder

    def compile_full_model(self):
        """ Takes all the submodules and module-lists
            and returns one gigantic sequential_model

        :returns: None
        :rtype: None

        """
        full_model_list, _ = layers.flatten_layers(self)
        return nn.Sequential(OrderedDict(full_model_list))

    def reparameterize_aggregate_posterior(self):
        """ Gets reparameterized aggregate posterior samples

        :returns: reparameterized tensor
        :rtype: torch.Tensor

        """
        training_tmp = self.reparameterizer.training
        self.reparameterizer.train(True)
        z_samples, _ = self.reparameterize(self.aggregate_posterior.ema_val)
        self.reparameterizer.train(training_tmp)
        return z_samples

    def generate_synthetic_samples(self, batch_size, **kwargs):
        """ Generates samples with VAE.

        :param batch_size: the number of samples to generate.
        :returns: decoded logits
        :rtype: torch.Tensor

        """
        if 'use_aggregate_posterior' in kwargs and kwargs['use_aggregate_posterior']:
            z_samples = self.reparameterize_aggregate_posterior()
        else:
            z_samples = self.reparameterizer.prior(
                batch_size, scale_var=self.config['generative_scale_var'], **kwargs
            )

        # in the normal case just decode and activate
        return self.nll_activation(self.decode(z_samples))

    def generate_synthetic_sequential_samples(self, num_original_discrete, num_rows=8):
        """ Iterates over all discrete positions and generates samples (for mix or disc only).

        :param num_original_discrete: The original discrete size (useful for LLVAE).
        :param num_rows: for visdom
        :returns: decoded logits
        :rtype: torch.Tensor

        """
        assert self.has_discrete()

        # create a grid of one-hot vectors for displaying in visdom
        # uses one row for original dimension of discrete component
        discrete_indices = np.array([np.random.randint(begin, end, size=num_rows) for begin, end in
                                     zip(range(0, self.reparameterizer.config['discrete_size'],
                                               num_original_discrete),
                                         range(num_original_discrete,
                                               self.reparameterizer.config['discrete_size'] + 1,
                                               num_original_discrete))])
        discrete_indices = discrete_indices.reshape(-1)

        self.eval()  # lock BN / Dropout, etc
        with torch.no_grad():
            z_samples = Variable(
                torch.from_numpy(utils.one_hot_np(self.reparameterizer.config['discrete_size'],
                                                  discrete_indices))
            )
            z_samples = z_samples.type(utils.same_type(self.config['half'], self.config['cuda']))

            if self.config['reparam_type'] == 'mixture' and self.config['vae_type'] != 'sequential':
                ''' add in the gaussian prior '''
                z_cont = self.reparameterizer.continuous.prior(z_samples.size(0))
                z_samples = torch.cat([z_cont, z_samples], dim=-1)

            # the below is to handle the issues with BN
            # pad the z to be full batch size
            number_to_return = z_samples.shape[0]  # original generate number
            number_batches_z = int(max(1, np.ceil(
                float(self.config['batch_size']) / float(number_to_return))))
            z_padded = torch.cat(
                [z_samples for _ in range(number_batches_z)], 0
            )[0:self.config['batch_size']]

            # generate and return the requested number
            number_batches_to_generate = int(max(1, np.ceil(
                float(number_to_return) / float(self.config['batch_size']))))
            generated = torch.cat([self.generate_synthetic_samples(
                self.config['batch_size'], z_samples=z_padded
            ) for _ in range(number_batches_to_generate)], 0)
            return generated[0:number_to_return]  # only return num_requested

    def nll_activation(self, logits):
        """ Activates the logits

        :param logits: the unactivated logits
        :returns: activated logits.
        :rtype: torch.Tensor

        """
        return distributions.nll_activation(logits,
                                            self.config['nll_type'],
                                            chans=self.chans)

    def forward(self, x, labels=None):
        """Accepts input (and optionally labels), gets posterior and latent and decodes.

        :param x: input tensor.
        :param labels: (optional) labels
        :returns: decoded logits and reparam dict
        :rtype: torch.Tensor, dict

        """
        z, params = self.posterior(x, labels=labels)
        decoded_logits = self.decode(z)
        params = self._compute_mi_params(decoded_logits, params)
        return decoded_logits, params

    def importance_weighted_elbo_analytic(self, x, labels=None, K=5000):
        """ ELBO using importance samples.

        :param x: input data.
        :param labels: (optional) labels
        :param K: number of importance samples.
        :returns: negative ELBO
        :rtype: float32

        """
        elbo_mu = torch.zeros(x.shape[0], device=x.device)

        for _ in range(K):
            z, params = self.posterior(x, labels=labels, force=True)
            decoded_logits, params = self.decode(z)
            loss_t = self.loss_function(decoded_logits, x, params)
            elbo_mu += loss_t['elbo']

        # log-sum exp and return
        return torch.logsumexp(elbo_mu / K, dim=0)

    def importance_weighted_elbo_monte_carlo(self, x, labels=None, K=500):
        """ ELBO using importance samples.

        :param x: input data.
        :param labels: (optional) labels.
        :param K: number of importance samples.
        :returns: negative ELBO
        :rtype: float32

        """
        elbo_mu = torch.zeros(x.shape[0], device=x.device)
        from tqdm import tqdm

        for _ in tqdm(range(K)):
            z, params = self.posterior(x, labels=labels, force=True)
            decoded_logits = self.decode(z)

            # grab log likelihood of posterior and prior over z
            log_q_z_given_x = self.reparameterizer.log_likelihood(z, params)
            prior_params = self.reparameterizer.prior_params(x.shape[0])
            log_p_z = self.reparameterizer.log_likelihood(z, prior_params)

            # compute the NLL and the log_q_z_given_x - log_p_z
            nll = distributions.nll(x, decoded_logits, self.config['nll_type'])
            kl_z = torch.sum(log_q_z_given_x - log_p_z, -1)
            elbo_mu += nll - kl_z

        # log-sum exp and return
        return torch.logsumexp(elbo_mu / K, dim=0)

    def compute_kl_beta(self, kl_beta_list):
        """Compute the KL-beta term using an annealer or just returns.

        :param kl_beta_list: a list of kl-beta values to scale
        :returns: scalar float32
        :rtype: float32

        """
        if self.kl_annealer is not None:
            kl_beta_list = self.kl_annealer(kl_beta_list)

        return kl_beta_list

    def loss_function(self, recon_x, x, params, K=1):
        """ Produces ELBO, handles mutual info and proxy loss terms too.

        :param recon_x: the unactivated reconstruction preds.
        :param x: input tensor.
        :param params: the dict of reparameterization.
        :param K: number of monte-carlo samples to use.
        :returns: loss dict
        :rtype: dict

        """
        nll = distributions.nll(x, recon_x, self.config['nll_type'])

        # multiple monte-carlo samples for the decoder.
        if self.training:
            for k in range(1, K):
                z_k, params_k = self.reparameterize(logits=params['logits'],
                                                    labels=params.get('labels', None))
                recon_x_i = self.decode(z_k)
                nll = nll + distributions.nll(x, recon_x_i, self.config['nll_type'])

            nll = nll / K

        kld = self.kld(params)
        elbo = nll + kld  # save the base ELBO, but use the beta-vae elbo for the full loss

        # add the proxy loss if it exists
        proxy_loss = self.reparameterizer.proxy_layer.loss_function() \
            if hasattr(self.reparameterizer, 'proxy_layer') else torch.zeros_like(elbo)

        # handle the mutual information term
        mut_info = self.mut_info(params, x.size(0))

        # get the kl-beta from the annealer or just set to fixed value
        kl_beta = self.compute_kl_beta([self.config['kl_beta']])[0]

        # sanity checks only dont in fp32 due to too much fp16 magic
        if not self.config['half']:
            utils.nan_check_and_break(nll, "nll")
            if kl_beta > 0:  # only check if we have a KLD
                utils.nan_check_and_break(kld, "kld")

        # compute full loss to use for optimization
        loss = (nll + kl_beta * kld) - mut_info
        return {
            'loss': loss,
            'elbo': elbo,
            'loss_mean': torch.mean(loss),
            'elbo_mean': torch.mean(elbo),
            'nll_mean': torch.mean(nll),
            'kld_mean': torch.mean(kld),
            'kl_beta_scalar': kl_beta,
            'proxy_mean': torch.mean(proxy_loss),
            'mut_info_mean': torch.mean(mut_info)
        }

    def has_discrete(self):
        """ returns True if the model has a discrete
            as it's first (in the case of parallel) reparameterizer

        :returns: True/False
        :rtype: bool

        """
        return self.reparameterizer.is_discrete

    def reparameterize(self, logits, labels=None, force=False):
        """ Reparameterize the logits and returns a dict.

        :param logits: unactivated encoded logits.
        :param labels: (optional) labels
        :param force: force reparameterize the distributions
        :returns: reparam dict
        :rtype: dict

        """
        return self.reparameterizer(logits, force=force)

    def decode(self, z):
        """ Decode a latent z back to x.

        :param z: the latent tensor.
        :returns: decoded logits (unactivated).
        :rtype: torch.Tensor

        """
        decoded_logits = self.decoder(z.contiguous())
        return decoded_logits

    def posterior(self, x, labels=None, force=False):
        """ get a reparameterized Q(z|x) for a given x

        :param x: input tensor
        :param labels: (optional) labels
        :param force:  force reparameterization
        :returns: reparam dict
        :rtype: torch.Tensor

        """
        z_logits = self.encode(x)                          # encode logits
        self.aggregate_posterior(z_logits)                 # aggregate posterior EMA
        return self.reparameterize(z_logits, labels=labels, force=force)  # return reparameterized value

    def encode(self, x):
        """ Encodes a tensor x to a set of logits.

        :param x: the input tensor
        :returns: logits
        :rtype: torch.Tensor

        """
        encoded = self.encoder(x).squeeze()
        if encoded.dim() < 2:
            return encoded.unsqueeze(-1)

        return encoded

    def kld(self, dist_a):
        """ KL-Divergence of the distribution dict and the prior of that distribution.

        :param dist_a: the distribution dict.
        :returns: tensor that is of dimension batch_size
        :rtype: torch.Tensor

        """
        return self.reparameterizer.kl(dist_a)

    def _clamp_mut_info(self, mut_info):
        """ helper to clamp the mutual information according to a predefined strategy

        :param mut_info: the tensor of mut-info
        :returns: clamped mut-info
        :rtype: torch.Tensor

        """
        mut_clamp_strategy_map = {                # Clamping strategies
            'none': lambda mut_info: mut_info,
            'norm': lambda mut_info: mut_info / torch.norm(mut_info, p=2),
            'clamp': lambda mut_info: torch.clamp(mut_info,
                                                  min=-self.config['mut_clamp_value'],
                                                  max=self.config['mut_clamp_value'])
        }
        return mut_clamp_strategy_map[self.config['mut_clamp_strategy'].strip().lower()](mut_info)

    def _compute_mi_params(self, recon_x_logits, params):
        """ Internal helper to compute the MI params and append to full params

        :param recon_x: reconstruction
        :param params: the original params
        :returns: original params OR param + MI_params
        :rtype: dict

        """
        if self.config['continuous_mut_info'] > 0 or self.config['discrete_mut_info'] > 0:
            _, q_z_given_xhat_params = self.posterior(self.nll_activation(recon_x_logits))
            return {**params, 'q_z_given_xhat': q_z_given_xhat_params}

        # base case, no MI
        return params

    def mut_info(self, dist_params, batch_size):
        """ Returns mutual information between z <-> x

        :param dist_params: the distribution dict
        :returns: tensor of dimension batch_size
        :rtype: torch.Tensor

        """
        mut_info = utils.same_type(self.config['half'], self.config['cuda'])(batch_size).zero_()

        # only grab the mut-info if the scalars above are set
        if self.config['continuous_mut_info'] > 0 or self.config['discrete_mut_info'] > 0:
            mut_info = self._clamp_mut_info(self.reparameterizer.mutual_info(dist_params))

        return mut_info

    def get_activated_reconstructions(self, reconstr):
        """ Returns activated reconstruction

        :param reconstr: unactivated reconstr logits
        :returns: activated reconstr
        :rtype: torch.Tensor

        """
        return {'reconstruction_imgs': self.nll_activation(reconstr)}
