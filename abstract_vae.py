from __future__ import print_function
import pprint
import functools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict, Counter

from helpers.pixel_cnn.model import PixelCNN
from helpers.utils import float_type, zeros, nan_check_and_break
from helpers.layers import View, flatten_layers, Identity, \
    build_pixelcnn_decoder, add_normalization, str_to_activ_module, get_decoder, get_encoder
from helpers.distributions import nll_activation as nll_activation_fn
from helpers.distributions import nll as nll_fn
from helpers.distributions import nll_has_variance


class VarianceProjector(nn.Module):
    ''' simple helper to project to 2 * chans if we have variance;
        or do nothing otherwise :) '''
    def __init__(self, output_shape, activation_fn, config):
        super(VarianceProjector, self).__init__()
        chans = output_shape[0]
        self.config = config

        # build the sequential layer
        if nll_has_variance(config['nll_type']):
            if config['decoder_layer_type'] == 'conv':
                self.decoder_projector = nn.Sequential(
                    #TODO: caused error with groupnorm w/ 32
                    #add_normalization(Identity(), config['normalization'], 2, self.chans, num_groups=32),
                    activation_fn(),
                    nn.ConvTranspose2d(chans, chans*2, 1, stride=1, bias=False)
                )
            else: # dense projector
                input_flat = int(np.prod(output_shape))
                self.decoder_projector = nn.Sequential(
                    View([-1, input_flat]),
                    add_normalization(Identity(), config['dense_normalization'], 1, input_flat),
                    activation_fn(),
                    nn.Linear(input_flat, input_flat*2, bias=True),
                    View([-1, chans*2, *output_shape[1:]])
                )

    def forward(self, x):
        if hasattr(self, 'decoder_projector'):
            return self.decoder_projector(x)

        return x


class AbstractVAE(nn.Module):
    ''' abstract base class for VAE, both sequentialVAE and parallelVAE inherit this '''
    def __init__(self, input_shape, **kwargs):
        super(AbstractVAE, self).__init__()
        self.input_shape = input_shape
        self.is_color = input_shape[0] > 1
        self.chans = 3 if self.is_color else 1

        # grab the meta config and print for
        self.config = kwargs['kwargs']
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)

        # grab the activation nn.Module from the string
        self.activation_fn = str_to_activ_module(self.config['activation'])

    def get_name(self, reparam_str):
        ''' helper to get the name of the model '''
        es_str = "es" + str(int(self.config['early_stop'])) if self.config['early_stop'] \
                 else "epochs" + str(self.config['epochs'])
        full_hash_str = """_{}{}_{}act{}_cr{}_dr{}_klr{}_gsv{}_mcig{}_mcs{}{}_input{}_batch{}_mut{}d{}c_filter{}_nll{}_lr{}_{}_{}_ngpu{}""".format(
            str(self.config['encoder_layer_type']),
            str(self.config['decoder_layer_type']),
            reparam_str,
            str(self.activation_fn.__name__),
            self.config['conv_normalization'],
            self.config['dense_normalization'],
            str(self.config['kl_reg']),
            str(self.config['generative_scale_var']),
            str(int(self.config['monte_carlo_infogain'])),
            str(self.config['mut_clamp_strategy']),
            "{}".format(str(self.config['mut_clamp_value'])) if self.config['mut_clamp_strategy'] == 'clamp' else "",
            str(self.input_shape),
            str(self.config['batch_size']),
            str(self.config['discrete_mut_info']),
            str(self.config['continuous_mut_info']),
            str(self.config['filter_depth']),
            str(self.config['nll_type']),
            str(self.config['lr']),
            es_str,
            str(self.config['optimizer']),
            str(self.config['ngpu'])
        )
        full_hash_str = full_hash_str.strip().lower().replace('[', '')  \
                                                     .replace(']', '')  \
                                                     .replace(' ', '')  \
                                                     .replace('\n', '') \
                                                     .replace('{', '') \
                                                     .replace('}', '') \
                                                     .replace(',', '_') \
                                                     .replace(':', '') \
                                                     .replace('(', '') \
                                                     .replace(')', '') \
                                                     .replace('\'', '')
        task_cleaned = AbstractVAE._clean_task_str(self.config['task'])
        return task_cleaned + full_hash_str


    @staticmethod
    def _clean_task_str(task_str):
        ''' helper to reduce string length.
            eg: mnist+svhn+mnist --> mnist2svhn1 '''
        result_str = ''
        if '+' in task_str:
            splits = Counter(task_str.split('+'))
            for k, v in splits.items():
                result_str += '{}{}'.format(k, v)

            return result_str

        return task_str

    def build_encoder(self):
        return get_encoder(self.config)(input_shape=self.input_shape,
                                        output_size=self.reparameterizer.input_size,
                                        activation_fn=self.activation_fn)

    def lazy_build_decoder(self, input_size):
        ''' lazily build the decoder network '''
        if not hasattr(self, 'decoder'):
            setattr(self, 'decoder', self._build_decoder(input_size))

        return self.decoder

    def build_decoder(self):
        ''' helper function to build convolutional or dense decoder'''
        return self._build_decoder(self.reparameterizer.output_size)

    def _build_decoder(self, input_size, reupsample=True):
        # sanity check, pcnn only works with discrete mixture logistic
        if self.config['decoder_layer_type'] == "pixelcnn":
            assert self.config['nll_type'] == "disc_mix_logistic", \
                "pixelcnn only works with disc_mix_logistic"

        decoder = get_decoder(self.config, reupsample)(input_size=input_size,
                                                       output_shape=self.input_shape,
                                                       activation_fn=self.activation_fn)
        # append the variance as necessary
        return self._append_variance_projection(decoder)

    def _append_variance_projection(self, decoder):
        if self.config['decoder_layer_type'] == "pixelcnn":
            # pixel CNN already accounts for variance internally
            self.pixel_cnn = PixelCNN(input_channels=self.chans,
                                      nr_resnet=2, nr_filters=40,
                                      nr_logistic_mix=10)
            decoder = nn.Sequential(
                decoder,
                self.activation_fn(),
                self.pixel_cnn
            )
        elif nll_has_variance(self.config['nll_type']):
            # add the variance projector (if we are in that case for the NLL)
            print("adding variance projector for {} log-likelihood".format(self.config['nll_type']))
            decoder = nn.Sequential(
                decoder,
                self.activation_fn(),
                VarianceProjector(self.input_shape, self.activation_fn, self.config)
            )

        return decoder

    def fp16(self):
        self.encoder = self.encoder.half()
        if self.config['decoder_layer_type'] == "pixelcnn":
            self.decoder = nn.Sequential(
                self.decoder[0:-1].half(),
                self.decoder[-1].half()
            )
        else:
            self.decoder = self.decoder.half()

    def parallel(self):
        self.encoder = nn.DataParallel(self.encoder)
        if self.config['decoder_layer_type'] == "pixelcnn":
            self.decoder = nn.Sequential(
                nn.DataParallel(self.decoder[0:-1]),
                nn.DataParallel(self.decoder[-1])
            )
        else:
            self.decoder = nn.DataParallel(self.decoder)

    def compile_full_model(self):
        ''' takes all the submodules and module-lists
            and returns one gigantic sequential_model '''
        full_model_list, _ = flatten_layers(self)
        return nn.Sequential(OrderedDict(full_model_list))

    def generate_pixel_cnn(self, batch_size, decoded=None):
        self.pixel_cnn.eval()
        if decoded is None:  # use zeros if no values provided
            # XXX: hardcoded cnn projection size, pre-upsample
            decoded = zeros([batch_size] + [self.config['img_shp'][0], 32, 32],
                            cuda=self.config['cuda'])

        for i in range(decoded.size(2)):
            for j in range(decoded.size(3)):
                logits = self.pixel_cnn(decoded, sample=True)
                out_sample = self.nll_activation(logits)
                decoded[:, :, i, j] = out_sample[:, :, i, j]

        rescaling_inv = lambda x : .5 * x  + .5
        return rescaling_inv(decoded)

    def generate_synthetic_samples(self, batch_size, **kwargs):
        z_samples = self.reparameterizer.prior(
            batch_size, scale_var=self.config['generative_scale_var'], **kwargs
        )

        if self.config['decoder_layer_type'] == "pixelcnn":
            # hot-swap the non-pixel CNN for the decoder
            full_decoder = self.decoder
            trunc_decoder = self.decoder[0:-1]
            self.decoder = trunc_decoder

            # decode the synthetic samples
            decoded = self.decode(z_samples)

            # swap back the decoder and run the pixelcnn
            self.decoder = full_decoder
            return self.generate_pixel_cnn(batch_size, decoded)

        # in the normal case just decode and activate
        return self.nll_activation(self.decode(z_samples))

    def generate_synthetic_sequential_samples(self, num_rows=8):
        ''' iterates over all discrete positions and generates samples '''
        assert self.has_discrete()

        # create a grid of one-hot vectors for displaying in visdom
        # uses one row for original dimension of discrete component
        discrete_indices = np.array([np.random.randint(begin, end, size=num_rows) for begin, end in
                                     zip(range(0, self.reparameterizer.config['discrete_size'],
                                               self.config['discrete_size']),
                                         range(self.config['discrete_size'],
                                               self.reparameterizer.config['discrete_size'] + 1,
                                               self.config['discrete_size']))])
        discrete_indices = discrete_indices.reshape(-1)

        self.eval() # lock BN / Dropout, etc
        with torch.no_grad():
            z_samples = Variable(torch.from_numpy(
                one_hot_np(self.reparameterizer.config['discrete_size'],
                           discrete_indices))
            )
            z_samples = z_samples.type(float_type(self.config['cuda']))

            if self.config['reparam_type'] == 'mixture' and self.config['vae_type'] != 'sequential':
                ''' add in the gaussian prior '''
                z_cont = self.reparameterizer.gaussian.prior(z_samples.size(0))
                z_samples = torch.cat([z_cont, z_samples], dim=-1)

            return self.nll_activation(self.decode(z_samples))

    def nll_activation(self, logits):
        return nll_activation_fn(logits,
                                 self.config['nll_type'],
                                 chans=self.chans)

    def forward(self, x):
        ''' params is a map of the latent variable's parameters'''
        if self.config['use_pixel_cnn_decoder']:
            rescaling = lambda x : (x - .5) * 2.
            x = rescaling(x)

        # encode to posterior and then decode
        z, params = self.posterior(x)
        decoded = self.decode(z)

        # and then tentatively invert for pixel_cnn
        # if self.config['use_pixel_cnn_decoder']:
        #     rescaling_inv = lambda x : .5 * x  + .5
        #     decoded = rescaling_inv(decoded)

        return decoded , params

    def loss_function(self, recon_x, x, params, mut_info=None):
        # elbo = -log_likelihood + latent_kl
        # cost = elbo + consistency_kl - self.mutual_info_reg * mutual_info_regularizer
        nll = nll_fn(x, recon_x, self.config['nll_type'])
        nan_check_and_break(nll, "nll")
        kld = self.config['kl_reg'] * self.kld(params)
        nan_check_and_break(kld, "kld")
        elbo = nll + kld

        # handle the mutual information term
        if mut_info is None:
            mut_info = Variable(
                float_type(self.config['cuda'])(x.size(0)).zero_()
            )
        else:
            # Clamping strategies
            mut_clamp_strategy_map = {
                'none': lambda mut_info: mut_info,
                'norm': lambda mut_info: mut_info / torch.norm(mut_info, p=2),
                'clamp': lambda mut_info: torch.clamp(mut_info,
                                                      min=-self.config['mut_clamp_value'],
                                                      max=self.config['mut_clamp_value'])
            }
            mut_info = mut_clamp_strategy_map[self.config['mut_clamp_strategy'].strip().lower()](mut_info)

        loss = elbo - mut_info
        return {
            'loss': loss,
            'loss_mean': torch.mean(loss),
            'elbo_mean': torch.mean(elbo),
            'nll_mean': torch.mean(nll),
            'kld_mean': torch.mean(kld),
            'mut_info_mean': torch.mean(mut_info)
        }

    def has_discrete(self):
        ''' returns True if the model has a discrete
            as it's first (in the case of parallel) reparameterizer'''
        raise NotImplementedError("has_discrete not implemented")

    def get_reparameterizer_scalars(self):
        ''' returns a map of the scalars of the reparameterizers.
            This is useful for visualization purposes'''
        raise NotImplementedError("get_reparameterizer_scalars not implemented")

    def reparameterize(self, logits):
        ''' reparameterizes the latent logits appropriately '''
        raise NotImplementedError("reparameterize not implemented")

    def decode(self, z):
        '''returns logits '''
        raise NotImplementedError("decode not implemented")

    def posterior(self, x):
        ''' get a reparameterized Q(z|x) for a given x '''
        z_logits = self.encode(x)
        return self.reparameterize(z_logits)

    def encode(self, x):
        ''' encodes via a convolution and returns logits '''
        raise NotImplementedError("encode not implemented")

    def kld(self, dist_params):
        ''' KL divergence between dist_a and prior '''
        raise NotImplementedError("kld not implemented")

    def mut_info(self, dist_params):
        ''' helper to get the mutual info to add to the loss '''
        raise NotImplementedError("mut_info not implemented")
