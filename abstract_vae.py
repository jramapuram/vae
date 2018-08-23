from __future__ import print_function
import pprint
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict, Counter


#from .pixelcnn import PixelCNN
from helpers.pixel_cnn.model import PixelCNN

from helpers.utils import float_type, zeros
from helpers.layers import View, flatten_layers, Identity, \
    build_gated_conv_encoder, build_conv_encoder, build_dense_encoder, build_gated_dense_encoder, \
    build_gated_conv_decoder, build_conv_decoder, build_dense_decoder, build_gated_dense_decoder, \
    build_relational_conv_encoder, build_pixelcnn_decoder, add_normalization, str_to_activ_module
from helpers.distributions import nll_activation as nll_activation_fn
from helpers.distributions import nll as nll_fn


class VarianceProjector(nn.Module):
    ''' simple helper to project to 2 * chans if we have variance;
        or do nothing otherwise :) '''
    def __init__(self, output_shape, activation_fn, config):
        super(VarianceProjector, self).__init__()
        chans = output_shape[0]

        # build the sequential layer
        if config['nll_type'] == 'gaussian' or config['nll_type'] == 'clamp':
            if config['layer_type'] == 'conv':
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
                    add_normalization(Identity(), config['normalization'], 1, input_flat),
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
        full_hash_str = "_{}_{}act{}_pd{}_klr{}_gsv{}_mcig{}_mcs{}{}_input{}_batch{}_mut{}d{}c_filter{}_nll{}_lr{}_{}_ngpu{}".format(
            str(self.config['layer_type']),
            reparam_str,
            str(self.activation_fn.__name__),
            str(int(self.config['use_pixel_cnn_decoder'])),
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
            str(self.config['ngpu'])
        )
        full_hash_str = full_hash_str.strip().lower().replace('[', '')  \
                                                     .replace(']', '')  \
                                                     .replace(' ', '')  \
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
        ''' helper function to build convolutional or dense encoder '''
        if self.config['layer_type'] == 'conv':
            if self.config['use_relational_encoder']:
                encoder = build_relational_conv_encoder(input_shape=self.input_shape,
                                                        filter_depth=self.config['filter_depth'],
                                                        activation_fn=self.activation_fn)
            else:
                conv_builder = build_gated_conv_encoder \
                           if self.config['disable_gated'] is False else build_conv_encoder
                encoder = conv_builder(input_shape=self.input_shape,
                                       output_size=self.reparameterizer.input_size,
                                       filter_depth=self.config['filter_depth'],
                                       activation_fn=self.activation_fn,
                                       normalization_str=self.config['normalization'])
        elif self.config['layer_type'] == 'dense':
            dense_builder = build_gated_dense_encoder \
                if self.config['disable_gated'] is False else build_dense_encoder
            encoder = dense_builder(input_shape=self.input_shape,
                                    output_size=self.reparameterizer.input_size,
                                    latent_size=512,
                                    activation_fn=self.activation_fn,
                                    normalization_str=self.config['normalization'])
        else:
            raise Exception("unknown layer type requested")

        # if self.config['ngpu'] > 1:
        #     encoder = nn.DataParallel(encoder)

        # if self.config['cuda']:
        #     encoder = encoder.cuda()

        return encoder

    def lazy_build_decoder(self, input_size):
        ''' lazily build the decoder network '''
        if not hasattr(self, 'decoder'):
            setattr(self, 'decoder', self._build_decoder(input_size))

        return self.decoder

    def build_decoder(self):
        ''' helper function to build convolutional or dense decoder'''
        return self._build_decoder(self.reparameterizer.output_size)

    def _build_decoder(self, input_size, reupsample=True):
        ''' helper function to build convolutional or dense decoder'''
        if self.config['layer_type'] == 'conv':
            conv_builder = build_gated_conv_decoder \
                           if self.config['disable_gated'] is False else build_conv_decoder
            decoder = conv_builder(input_size=input_size,
                                   output_shape=self.input_shape,
                                   filter_depth=self.config['filter_depth'],
                                   activation_fn=self.activation_fn,
                                   normalization_str=self.config['normalization'],
                                   reupsample=reupsample)
        elif self.config['layer_type'] == 'dense':
            dense_builder = build_gated_dense_decoder \
                if self.config['disable_gated'] is False else build_dense_decoder
            decoder = dense_builder(input_shape=input_size,
                                    output_shape=self.input_shape,
                                    activation_fn=self.activation_fn,
                                    normalization_str=self.config['normalization'])
        else:
            raise Exception("unknown layer type requested")

        if self.config['use_pixel_cnn_decoder']:
            print("adding pixel CNN decoder...")
            # chan_mult = 1 if self.config['nll_type'] == 'bernoulli' else 2
            # input_shape = [chan_mult * self.chans] + self.input_shape[1:]
            decoder = nn.Sequential(
                decoder,
                #PixelCNN(self.input_shape, self.config)
                PixelCNN(input_channels=self.chans)
            )

        # add the variance projector (if we are in that case for the NLL)
        decoder = nn.Sequential(
            decoder,
            #VarianceProjector(self.input_shape, self.activation_fn, self.config)
        )

        # if self.config['ngpu'] > 1:
        #     decoder = nn.DataParallel(decoder)

        # if self.config['cuda']:
        #     decoder = decoder.cuda()

        return decoder

    def parallel(self):
        self.encoder = nn.DataParallel(self.encoder)
        self.decoder = nn.DataParallel(self.decoder)

    def compile_full_model(self):
        ''' takes all the submodules and module-lists
            and returns one gigantic sequential_model '''
        full_model_list, _ = flatten_layers(self)
        return nn.Sequential(OrderedDict(full_model_list))

    def generate_synthetic_samples(self, batch_size, **kwargs):
        z_samples = self.reparameterizer.prior(
            batch_size, scale_var=self.config['generative_scale_var'], **kwargs
        )
        return self.nll_activation(self.decode(z_samples))

    def generate_synthetic_sequential_samples(self, num_rows=8):
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
        with torch.no_grad():
            z_samples = Variable(torch.from_numpy(
                one_hot_np(self.reparameterizer.config['discrete_size'],
                           discrete_indices))
            )
            z_samples = z_samples.type(float_type(self.config['cuda']))

            if self.config['reparam_type'] == 'mixture' and self.config['vae_type'] != 'sequential':
                ''' add in the gaussian prior '''
                z_gauss = self.reparameterizer.gaussian.prior(z_samples.size(0))
                z_samples = torch.cat([z_gauss, z_samples], dim=-1)

            return self.nll_activation(self.decode(z_samples))

    def nll_activation(self, logits):
        return nll_activation_fn(logits, self.config['nll_type'])

    def forward(self, x):
        ''' params is a map of the latent variable's parameters'''
        if self.config['use_pixel_cnn_decoder']:
            rescaling     = lambda x : (x - .5) * 2.
            x = rescaling(x)

        # encode to posterior and then decode
        z, params = self.posterior(x)
        decoded = self.decode(z)

        # and then tentatively invert for pixel_cnn
        if self.config['use_pixel_cnn_decoder']:
            rescaling_inv = lambda x : .5 * x  + .5
            decoded = rescaling_inv(decoded)

        return decoded , params

    def loss_function(self, recon_x, x, params, mut_info=None):
        # elbo = -log_likelihood + latent_kl
        # cost = elbo + consistency_kl - self.mutual_info_reg * mutual_info_regularizer
        # assert x.shape == recon_x.shape, "incompatible sizing for reconstruction {} vs. true data {}".format(
        #     list(recon_x.shape),
        #     list(x.shape)
        # )
        nll = nll_fn(x, recon_x, self.config['nll_type'])
        kld = self.config['kl_reg'] * self.kld(params)
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
