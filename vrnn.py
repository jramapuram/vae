import torch
import functools
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from copy import deepcopy

from .abstract_vae import AbstractVAE
from .reparameterizers.gumbel import GumbelSoftmax
from .reparameterizers.mixture import Mixture
from .reparameterizers.beta import Beta
from .reparameterizers.isotropic_gaussian import IsotropicGaussian
from helpers.distributions import nll_activation as nll_activation_fn
from helpers.distributions import nll as nll_fn
from helpers.layers import get_encoder, Identity
from helpers.utils import eps as eps_fn
from helpers.utils import same_type, zeros_like, expand_dims, \
    zeros, nan_check_and_break

class VRNNMemory(nn.Module):
    ''' Helper object to contain states and outputs for the RNN'''
    def __init__(self, h_dim, n_layers, bidirectional,
                 config, rnn=None, cuda=False):
        super(VRNNMemory, self).__init__()
        self.model = rnn
        self.config = config
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.h_dim = h_dim
        self.use_cuda = cuda
        self.memory_buffer = []

    @staticmethod
    def _state_from_tuple(tpl):
        _, state = tpl
        return state

    @staticmethod
    def _output_from_tuple(tpl):
        output, _ = tpl
        return output

    def _append_to_buffer(self, tpl):
        output_t, state_t = tpl
        self.memory_buffer.append([output_t.clone(), (state_t[0].clone(),
                                                      state_t[1].clone())])

    def clear(self):
        self.memory_buffer.clear()

    def init_state(self, batch_size, cuda=False,
                   override_noisy_state=False):
        def _init(batch_size, cuda):
            ''' return a single initialized state'''
            num_directions = 2 if self.bidirectional else 1
            if override_noisy_state or \
               (self.training and self.config['use_noisy_rnn_state']):
                # add some noise to initial state
                # consider also: nn.init.xavier_uniform_(
                return same_type(self.config['half'], cuda)(
                    num_directions * self.n_layers, batch_size, self.h_dim
                ).normal_(0, 0.01).requires_grad_()


            # return zeros for testing
            return same_type(self.config['half'], cuda)(
                num_directions * self.n_layers, batch_size, self.h_dim
            ).zero_().requires_grad_()

        self.state = ( # LSTM state is (h, c)
            _init(batch_size, cuda),
            _init(batch_size, cuda)
        )

    def init_output(self, batch_size, seqlen, cuda=False):
        self.outputs = same_type(self.config['half'], cuda)(
            seqlen, batch_size, self.h_dim
        ).zero_().requires_grad_()

    def update(self, tpl):
        self._append_to_buffer(tpl)
        self.outputs, self.state = tpl
        # if  self.state[0] is not None and self.state[0].requires_grad:
        #     self.state[0].register_hook(lambda x: x.clamp(min=-10, max=10))

        # if self.state[1] is not None and self.state[1].requires_grad:
        #     self.state[1].register_hook(lambda x: x.clamp(min=-10, max=10))

    def forward(self, input_t, reset_state=False):
        batch_size = input_t.size(0)
        if reset_state:
            self.init_state(batch_size, input_t.is_cuda)

        input_t = input_t.contiguous()

        # if not self.config['half']:
        self.update(self.model(input_t, self.state))
        # else:
        # self.update(self.model(input_t, collect_hidden=True))

        return self.get_output()

    def get_state(self):
        assert hasattr(self, 'state'), "do a forward pass first"
        return self.state

    def get_repackaged_state(self, h=None):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if h is None:
            return self.get_repackaged_state(self.state)

        if isinstance(h, torch.Tensor):
            return h.detach()

        return tuple(self.get_repackaged_state(v) for v in h)

    def get_output(self):
        assert hasattr(self, 'outputs'), "do a forward pass first"
        return self.outputs

    def get_merged_memory(self):
        ''' merges over num_layers of the state which is [nlayer, batch, latent]'''
        assert hasattr(self, 'memory_buffer'), "do a forward pass first"
        mem_concat = torch.cat([self._state_from_tuple(mem)[0]
                                for mem in self.memory_buffer], 0)
        return torch.mean(mem_concat, 0)

    def get_final_memory(self):
        assert hasattr(self, 'memory_buffer'), "do a forward pass first"
        return self._state_from_tuple(self.memory_buffer[-1])[0]


class VRNN(AbstractVAE):
    def __init__(self, input_shape, n_layers=2, bidirectional=False, **kwargs):
        """implementation of the Variational Recurrent
           Neural Network (VRNN) from https://arxiv.org/abs/1506.02216

           params:

             normalization: use a separate normalization for the VRNN [eg: GN doesnt work]
             n_layers: number of RNN layers
             bidirectional: bidirectional RNN
        """
        super(VRNN, self).__init__(input_shape, **kwargs)
        self.bidirectional = bidirectional
        self.n_layers = n_layers

        # build the reparameterizer
        if self.config['reparam_type'] == "isotropic_gaussian":
            print("using isotropic gaussian reparameterizer")
            self.reparameterizer = IsotropicGaussian(self.config)
        elif self.config['reparam_type'] == "discrete":
            print("using gumbel softmax reparameterizer")
            self.reparameterizer = GumbelSoftmax(self.config)
        elif self.config['reparam_type'] == "beta":
            print("using beta reparameterizer")
            self.reparameterizer = Beta(self.config)
        elif "mixture" in self.config['reparam_type']:
            print("using mixture reparameterizer with {} + discrete".format(
                'beta' if 'beta' in self.config['reparam_type'] else 'isotropic_gaussian'
            ))
            self.reparameterizer = Mixture(num_discrete=self.config['discrete_size'],
                                           num_continuous=self.config['continuous_size'],
                                           config=self.config,
                                           is_beta='beta' in self.config['reparam_type'])
        else:
            raise Exception("unknown reparameterization type")

        # build the entire model
        self._build_model()

    def _build_phi_x_model(self):
        ''' simple helper to build the feature extractor for x'''
        return self._lazy_build_phi_x(self.input_shape)

    def _lazy_build_phi_x(self, input_shape):
        return nn.Sequential(
            get_encoder(self.config)(input_shape=input_shape,
                                     output_size=self.config['latent_size'],
                                     activation_fn=self.activation_fn),
            self.activation_fn()
            #nn.SELU()
        )

    def _lazy_rnn_lambda(self, x, state,
                          model_type='lstm',
                          bias=True,
                          dropout=0):
        ''' automagically builds[if it does not exist]
            and returns the output of an RNN lazily '''
        if not hasattr(self, 'rnn'):
            self.rnn = self._build_rnn_memory_model(input_size=x.size(-1),
                                                    model_type=model_type,
                                                    bias=bias,
                                                    dropout=dropout)

        return self.rnn(x, state)

    def _get_dense_net_map(self, name='vrnn'):
        '''helper to pull a dense encoder'''
        config = deepcopy(self.config)
        config['encoder_layer_type'] = 'dense'
        return get_encoder(config, name=name)

    def _build_model(self):
        input_size = int(np.prod(self.input_shape))

        # feature-extracting transformations
        self.phi_x = self._build_phi_x_model()
        self.phi_x_i = []
        self.phi_z = nn.Sequential(
            self._get_dense_net_map('phi_z')(
                self.reparameterizer.output_size, self.config['latent_size'],
                activation_fn=self.activation_fn,
                normalization_str=self.config['dense_normalization'],
                #activation_fn=Identity,     # XXX: hardcode
                #normalization_str='batchnorm',     # XXX: hardcode
                num_layers=2
            ),
            nn.SELU()
            # self.activation_fn()
        )

        # prior
        self.prior = self._get_dense_net_map('prior')(
            self.config['latent_size'], self.reparameterizer.input_size,
            activation_fn=self.activation_fn,
            normalization_str=self.config['dense_normalization'],
            # activation_fn=Identity,
            # normalization_str='batchnorm',
            num_layers=2
        )

        # decoder
        self.decoder = self._build_decoder(input_size=self.config['latent_size']*2,
                                           reupsample=True)

        # memory module that contains the RNN or DNC
        self.memory = VRNNMemory(h_dim=self.config['latent_size'],
                                 n_layers=self.n_layers,
                                 bidirectional=self.bidirectional,
                                 config=self.config,
                                 rnn=self._lazy_rnn_lambda,
                                 cuda=self.config['cuda'])

    def fp16(self):
        self.phi_x = self.phi_x.half()
        self.phi_z = self.phi_z.half()
        self.prior = self.prior.half()
        super(VRNN, self).fp16()
        # RNN should already be half'd

    def parallel(self):
        self.phi_x = nn.DataParallel(self.phi_x)
        self.phi_z = nn.DataParallel(self.phi_z)
        self.prior = nn.DataParallel(self.prior)
        super(VRNN, self).parallel()

        # TODO: try to get this working
        #self.memory.model = nn.DataParallel(self.memory.model)

    def get_name(self):
        if "mixture" in self.config['reparam_type']:
            reparam_str = "mixturecat{}{}{}_".format(
                str(self.config['discrete_size']),
                'beta' if 'beta' in self.config['reparam_type'] else 'gauss',
                str(self.config['continuous_size'])
            )
        elif self.config['reparam_type'] == "isotropic_gaussian" or self.config['reparam_type'] == "beta":
            reparam_str = "cont{}_".format(str(self.config['continuous_size']))
        elif self.config['reparam_type'] == "discrete":
            reparam_str = "disc{}_".format(str(self.config['discrete_size']))
        else:
            raise Exception("unknown reparam type")

        base_str = 'vrnn_{}ts_{}ns_{}pkl'.format(
            self.config['max_time_steps'],
            int(self.config['use_noisy_rnn_state']),
            int(self.config['use_prior_kl'])
        )
        return base_str + super(VRNN, self).get_name(reparam_str)

    def has_discrete(self):
        ''' True is we have a discrete reparameterization '''
        return self.config['reparam_type'] == 'mixture' \
            or self.config['reparam_type'] == 'discrete'

    def get_reparameterizer_scalars(self):
        ''' basically returns tau from reparameterizers for now '''
        reparam_scalar_map = {}
        if isinstance(self.reparameterizer, GumbelSoftmax):
            reparam_scalar_map['tau_scalar'] = self.reparameterizer.tau
        elif isinstance(self.reparameterizer, Mixture):
            reparam_scalar_map['tau_scalar'] = self.reparameterizer.discrete.tau

        return reparam_scalar_map

    def _build_rnn_memory_model(self, input_size, model_type='lstm', bias=True, dropout=0):
        if self.config['half']:
            import apex

        model_fn_map = {
            'gru': torch.nn.GRU if not self.config['half'] else apex.RNN.GRU,
            'lstm': torch.nn.LSTM if not self.config['half'] else apex.RNN.LSTM
        }
        rnn = model_fn_map[model_type](
            input_size=input_size,
            hidden_size=self.config['latent_size'],
            num_layers=self.n_layers,
            bidirectional=self.bidirectional,
            bias=bias, dropout=dropout
        )

        if self.config['cuda'] and not self.config['half']:
            rnn.flatten_parameters()

        return rnn

    def _clamp_variance(self, logits):
        ''' clamp the variance when using a gaussian dist '''
        if self.config['reparam_type'] == 'isotropic_gaussian':
            feat_size = logits.size(-1)
            return torch.cat(
                [logits[:, 0:feat_size//2],
                 torch.sigmoid(logits[:, feat_size//2:])],
                -1)
        elif self.config['reparam_type'] == 'mixture':
            feat_size = self.reparameterizer.num_continuous_input
            return torch.cat(
                [logits[:, 0:feat_size//2],                    # mean
                 torch.sigmoid(logits[:, feat_size//2:feat_size]), # clamped var
                 logits[:, feat_size:]],                       # discrete
                -1)
        else:
            return logits

    def reparameterize(self, logits_map):
        '''reparameterize the encoder output and the prior'''
        # nan_check_and_break(logits_map['encoder_logits'], "enc_logits")
        # nan_check_and_break(logits_map['prior_logits'], "prior_logits")
        z_enc_t, params_enc_t = self.reparameterizer(logits_map['encoder_logits'])

        # XXX: clamp the variance of gaussian priors to not explode
        logits_map['prior_logits'] = self._clamp_variance(logits_map['prior_logits'])

        # reparamterize the prior distribution
        z_prior_t, params_prior_t = self.reparameterizer(logits_map['prior_logits'])

        z = {  # reparameterization
            'prior': z_prior_t,
            'posterior': z_enc_t,
            'x_features': logits_map['x_features']
        }
        params = {  # params of the posterior
            'prior': params_prior_t,
            'posterior': params_enc_t
        }

        return z, params

    def decode(self, z_t, produce_output=False, reset_state=False):
        # grab state from RNN, TODO: evaluate recovery methods below
        # [0] grabs the h from LSTM (as opposed to (h, c))
        final_state = torch.mean(self.memory.get_state()[0], 0)
        # nan_check_and_break(final_state, "final_rnn_output[decode]")

        # feature transform for z_t
        phi_z_t = self.phi_z(z_t['posterior'])
        # nan_check_and_break(phi_z_t, "phi_z_t")

        # concat and run through RNN to update state
        input_t = torch.cat([z_t['x_features'], phi_z_t], -1).unsqueeze(0)
        self.memory(input_t.contiguous(), reset_state=reset_state)

        # decode only if flag is set
        dec_t = None
        if produce_output:
            dec_input_t = torch.cat([phi_z_t, final_state], -1)
            dec_t = self.decoder(dec_input_t)

        return dec_t

    def _extract_features(self, x, *xargs):
        ''' accepts x and any number of extra x items and returns
            each of them projected through it's own NN,
            creating any networks as needed
        '''
        phi_x_t = self.phi_x(x)
        for i, x_item in enumerate(xargs):
            if len(self.phi_x_i) < i + 1:
                # add a new model at runtime if needed
                self.phi_x_i.append(self._lazy_build_phi_x(x_item.size()[1:]))
                print("increased length of feature extractors to {}".format(len(self.phi_x_i)))

            # use the model and concat on the feature dimension
            phi_x_i = self.phi_x_i[i](x_item)
            phi_x_t = torch.cat([phi_x_t, phi_x_i], -1)

        # nan_check_and_break(phi_x_t, "phi_x_t")
        return phi_x_t

    def _lazy_build_encoder(self, input_size):
        ''' lazy build the encoder based on the input size'''
        if not hasattr(self, 'encoder'):
            self.encoder = self._get_dense_net_map('vrnn_enc')(
                input_size, self.reparameterizer.input_size,
                activation_fn=self.activation_fn,
                normalization_str=self.config['dense_normalization'],
                # activation_fn=Identity,
                # normalization_str='batchnorm',
                num_layers=2
            )

        return self.encoder

    def encode(self, x, *xargs):
        # get the memory trace, TODO: evaluate different recovery methods below
        batch_size = x.size(0)
        final_state = torch.mean(self.memory.get_state()[0], 0)
        nan_check_and_break(final_state, "final_rnn_output")

        # extract input data features
        phi_x_t = self._extract_features(x, *xargs)

        # encoder projection
        enc_input_t = torch.cat([phi_x_t, final_state], dim=-1)
        enc_t = self._lazy_build_encoder(enc_input_t.size(-1))(enc_input_t)
        nan_check_and_break(enc_t, "enc_t")

        # prior projection , consider: + eps_fn(self.config['cuda']))
        prior_t = self.prior(final_state.contiguous())
        nan_check_and_break(prior_t, "priot_t")

        return {
            'encoder_logits': enc_t,
            'prior_logits': prior_t,
            'x_features': phi_x_t
        }

    def _decode_pixelcnn_or_normal(self, dec_input_t):
        ''' helper to decode using the pixel-cnn or normal decoder'''
        if self.config['decoder_layer_type'] == "pixelcnn":
            # hot-swap the non-pixel CNN for the decoder
            full_decoder = self.decoder
            trunc_decoder = self.decoder[0:-1]

            # decode the synthetic samples using non-pCNN
            decoded = trunc_decoder(dec_input_t)

            # then decode with the pCNN
            return self.generate_pixel_cnn(dec_input_t.size(0), decoded)

        dec_logits_t = self.decoder(dec_input_t)
        return self.nll_activation(dec_logits_t)

    def generate_synthetic_samples(self, batch_size, **kwargs):
        if 'reset_state' in kwargs and kwargs['reset_state']:
            self.memory.init_state(batch_size, cuda=self.config['cuda'],
                                   override_noisy_state=True)

        # grab the final state
        final_state = torch.mean(self.memory.get_state()[0], 0)

        # reparameterize the prior distribution
        prior_t = self.prior(final_state.contiguous())
        prior_t = self._clamp_variance(prior_t)
        z_prior_t, params_prior_t = self.reparameterizer(prior_t)

        # encode prior sample, this contrasts the decoder where
        # the features are run through this network
        phi_z_t = self.phi_z(z_prior_t)

        # construct decoder inputs and process
        dec_input_t = torch.cat([phi_z_t, final_state], -1)
        dec_output_t = self._decode_pixelcnn_or_normal(dec_input_t)

        # use the decoder output as features to update the RNN
        phi_x_t = self.phi_x(dec_output_t)
        input_t = torch.cat([phi_x_t, phi_z_t], -1).unsqueeze(0)
        self.memory(input_t.contiguous(), reset_state=False)

        # return the activated outputs
        return dec_output_t

    def posterior(self, *x_args):
        logits_map = self.encode(*x_args)
        return self.reparameterize(logits_map)

    def _ensure_same_size(self, prediction_list, target_list):
        ''' helper to ensure that image sizes in both lists match '''
        assert len(prediction_list) == len(target_list), "#preds[{}] != #targets[{}]".format(
            len(prediction_list), len(target_list))
        for i in range(len(target_list)):
            if prediction_list[i].size() != target_list[i].size():
                if prediction_list[i].size() > target_list[i].size():
                    larger_size = prediction_list[i].size()
                    target_list[i] = F.upsample(target_list[i],
                                                size=tuple(larger_size[2:]),
                                                mode='bilinear')

                else:
                    larger_size = target_list[i].size()
                    prediction_list[i] = F.upsample(prediction_list[i],
                                                    size=tuple(larger_size[2:]),
                                                    mode='bilinear')

        return prediction_list, target_list

    def kld(self, dist):
        ''' KL divergence between dist_a and prior as well as constrain prior to hyper-prior'''
        prior_kl = self.reparameterizer.kl(dist['prior'])  \
            if self.config['use_prior_kl'] is True else 0
        return self.reparameterizer.kl(dist['posterior'], dist['prior']) + prior_kl

    def mut_info(self, dist_params_container):
        ''' helper to get mutual info '''
        mut_info = None
        if (self.config['continuous_mut_info'] > 0
            or self.config['discrete_mut_info'] > 0):
            # only grab the mut-info if the scalars above are set
            mut_info = [self.reparameterizer.mutual_info(params['posterior']).unsqueeze(0)
                        for params in dist_params_container]
            mut_info = torch.sum(torch.cat(mut_info, 0), 0)

        return mut_info

    @staticmethod
    def _add_loss_map(loss_t, loss_aggregate_map):
        ''' helper to add two maps and keep counts
            of the total samples for reduction later'''
        if loss_aggregate_map is None:
            return {**loss_t, 'count': 1}

        for (k, v) in loss_t.items():
            loss_aggregate_map[k] += v

        # increment total count
        loss_aggregate_map['count'] += 1
        return loss_aggregate_map

    @staticmethod
    def _mean_map(loss_aggregate_map):
        ''' helper to reduce all values by the key count '''
        for k in loss_aggregate_map.keys():
            if k == 'count':
                continue

            loss_aggregate_map[k] /= loss_aggregate_map['count']

        return loss_aggregate_map

    def loss_function(self, recon_x_container, x_container, params_map):
        ''' evaluates the loss of the model by simply summing individual losses '''
        assert len(recon_x_container) == len(x_container) == len(params_map)
        loss_aggregate_map = None
        for recon_x, x, params in zip(recon_x_container, x_container, params_map):
            mut_info_t = self.mut_info(params)
            loss_t = super(VRNN, self).loss_function(recon_x, x, params,
                                                     mut_info=mut_info_t)
            loss_aggregate_map = self._add_loss_map(loss_t, loss_aggregate_map)

        return self._mean_map(loss_aggregate_map)
