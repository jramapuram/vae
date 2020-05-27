import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from .abstract_vae import AbstractVAE
from .reparameterizers import get_reparameterizer
from helpers.distributions import nll_activation as nll_activation_fn
from helpers.layers import get_encoder, get_decoder, str_to_activ_module, EMA
from helpers.utils import add_noise_to_imgs, float_type
from helpers.utils import same_type, nan_check_and_break


class VRNNMemory(nn.Module):
    def __init__(self, h_dim, n_layers, bidirectional,
                 config, rnn=None, cuda=False):
        """  Helper object to abstract away memory for VRNN.

        :param h_dim: hidden size
        :param n_layers: number of layers for RNN
        :param bidirectional: bidirectional bool flag
        :param config: argparse
        :param rnn: the rnn object
        :param cuda: cuda flag
        :returns: VRNNMemory object
        :rtype: nn.Module

        """
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
        """ Return the state from the tuple.

        :param tpl: the state-output tuple.
        :returns: state
        :rtype: torch.Tensor

        """
        _, state = tpl
        return state

    @staticmethod
    def _output_from_tuple(tpl):
        """ Return the output from the tuple.

        :param tpl: the state-output tuple.
        :returns: output.
        :rtype: torch.Tensor

        """
        output, _ = tpl
        return output

    def _append_to_buffer(self, tpl):
        """ Append the tuple to the running memory buffer.

        :param tpl: the current tuple
        :returns: None
        :rtype: None

        """
        output_t, state_t = tpl
        self.memory_buffer.append([output_t.clone(), (state_t[0].clone(),
                                                      state_t[1].clone())])

    def clear(self):
        """ Clears the memory.

        :returns: None
        :rtype: None

        """
        self.memory_buffer.clear()

    def init_state(self, batch_size, cuda=False, override_noisy_state=False):
        """ Initializes (or re-initializes) the state.

        :param batch_size: size of batch
        :param cuda: bool flag
        :param override_noisy_state: bool flag for setting up noisy state
        :returns: state-tuple
        :rtype: (torch.Tensor, torch.Tensor)

        """
        def _init(batch_size, cuda):
            """ Return a single initialized state

            :param batch_size: batch size
            :param cuda: is on cuda or not
            :returns: a single state init
            :rtype: (torch.Tensor, torch.Tensor)

            """
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

        self.state = (  # LSTM state is (h, c)
            _init(batch_size, cuda),
            _init(batch_size, cuda)
        )

    def update(self, tpl):
        """ Adds tuple to buffer and set current members

        :param tpl: the state-output tuple.
        :returns: None
        :rtype: None

        """
        self._append_to_buffer(tpl)
        self.outputs, self.state = tpl

    def forward(self, input_t, reset_state=False):
        """ Single-step forward pass, encodes with RNN and cache state.

        :param input_t: the current input tensor
        :param reset_state: whether or not to reset the current state
        :returns: the current output
        :rtype: torch.Tensor

        """
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
        """ Returns latest state.

        :returns: state
        :rtype: torch.Tensor

        """
        assert hasattr(self, 'state'), "do a forward pass first"
        return self.state

    def get_repackaged_state(self, h=None):
        """ Wraps hidden states in new Tensors, to detach them from their history.

        :param h: the state tuple to repackage (optional).
        :returns: tuple of repackaged states
        :rtype: (torch.Tensor, torch.Tensor)

        """
        if h is None:
            return self.get_repackaged_state(self.state)

        if isinstance(h, torch.Tensor):
            return h.detach()

        return tuple(self.get_repackaged_state(v) for v in h)

    def get_output(self):
        """ Helper to get the latest output

        :returns: output tensor
        :rtype: torch.Tensor

        """
        assert hasattr(self, 'outputs'), "do a forward pass first"
        return self.outputs

    def get_merged_memory(self):
        """ Merges over num_layers of the state which is [nlayer, batch, latent]

        :returns: merged temporal memory.
        :rtype: torch.Tensor

        """
        assert hasattr(self, 'memory_buffer'), "do a forward pass first"
        mem_concat = torch.cat([self._state_from_tuple(mem)[0]
                                for mem in self.memory_buffer], 0)
        return torch.mean(mem_concat, 0)

    def get_final_memory(self):
        """ Get the final memory state.

        :returns: the final memory state.
        :rtype: torch.Tensor

        """
        assert hasattr(self, 'memory_buffer'), "do a forward pass first"
        return self._state_from_tuple(self.memory_buffer[-1])[0]


class VRNN(AbstractVAE):
    def __init__(self, input_shape, n_layers=2, bidirectional=False, **kwargs):
        """ Implementation of the Variational Recurrent
            Neural Network (VRNN) from https://arxiv.org/abs/1506.02216

        :param input_shape: the input dimension
        :param n_layers: number of RNN / equivalent layers
        :param bidirectional: whether the model is bidirectional or not
        :returns: VRNN object
        :rtype: AbstractVAE

        """
        super(VRNN, self).__init__(input_shape, **kwargs)
        assert self.config['max_time_steps'] > 0, "Need max_time_steps > 0 for VRNN."
        self.activation_fn = str_to_activ_module(self.config['activation'])
        self.bidirectional = bidirectional
        self.n_layers = n_layers

        # build the reparameterizer
        self.reparameterizer = get_reparameterizer(self.config['reparam_type'])(self.config)

        # keep track of ammortized posterior
        self.aggregate_posterior = nn.ModuleDict({
            'encoder_logits': EMA(0.999),
            'prior_logits': EMA(0.999),
            'rnn_hidden_state_h': EMA(0.999),
            'rnn_hidden_state_c': EMA(0.999)
        })

        # build the entire model
        self._build_model()

    def _build_phi_x_model(self):
        """ simple helper to build the feature extractor for x

        :returns: a model for phi_x
        :rtype: nn.Module

        """
        return self._lazy_build_phi_x(self.input_shape)

    def _lazy_build_phi_x(self, input_shape):
        """ Lazily build an encoder to extract features.

        :param input_shape: the input tensor shape
        :returns: an encoder module
        :rtype: nn.Module

        """
        conf = deepcopy(self.config)
        conf['input_shape'] = input_shape
        return nn.Sequential(
            get_encoder(**conf)(output_size=self.config['latent_size']),
            self.activation_fn()
            # nn.SELU()
        )

    def _lazy_rnn_lambda(self, x, state,
                         model_type='lstm',
                         bias=True,
                         dropout=0):
        """ automagically builds[if it does not exist]
            and returns the output of an RNN lazily

        :param x: the input tensor
        :param state: the state tensor
        :param model_type: lstm or gru
        :param bias: whether to use a bias or not
        :param dropout: whether to use dropout or not
        :returns: lazy-inits an RNN and returns the RNN forward pass
        :rtype: (torch.Tensor, torch.Tensor)

        """
        if not hasattr(self, 'rnn'):
            self.rnn = self._build_rnn_memory_model(input_size=x.size(-1),
                                                    model_type=model_type,
                                                    bias=bias,
                                                    dropout=dropout)

        return self.rnn(x, state)

    def _get_dense_net_map(self, input_shape, name='vrnn'):
        """ helper to pull a dense encoder

        :param name: the name of the dense network
        :returns: the dense network
        :rtype: nn.Module

        """
        config = deepcopy(self.config)
        config['encoder_layer_type'] = 'dense'
        config['input_shape'] = input_shape
        return get_encoder(**config, name=name)

    def _build_model(self):
        """ Helper to build the entire model as members of this class.

        :returns: None
        :rtype: None

        """

        # feature-extracting transformations
        self.phi_x = self._build_phi_x_model()
        self.phi_x_i = []
        self.phi_z = nn.Sequential(
            self._get_dense_net_map(self.reparameterizer.output_size, 'phi_z')(
                output_size=self.config['latent_size']
            ),
            nn.SELU()
            # self.activation_fn()
        )

        # prior
        self.prior = self._get_dense_net_map(self.config['latent_size'], 'prior')(
            output_size=self.reparameterizer.input_size
        )

        # decoder
        self.decoder = self.build_decoder()

        # memory module that contains the RNN or DNC
        self.memory = VRNNMemory(h_dim=self.config['latent_size'],
                                 n_layers=self.n_layers,
                                 bidirectional=self.bidirectional,
                                 config=self.config,
                                 rnn=self._lazy_rnn_lambda,
                                 cuda=self.config['cuda'])

    def build_decoder(self, reupsample=True):
        """ helper function to build convolutional or dense decoder

        :returns: a decoder
        :rtype: nn.Module

        """
        dec_conf = deepcopy(self.config)
        if dec_conf['nll_type'] == 'pixel_wise':
            dec_conf['input_shape'][0] *= 256

        decoder = get_decoder(output_shape=dec_conf['input_shape'], **dec_conf)(
            input_size=self.config['latent_size']*2
        )

        # append the variance as necessary
        return self._append_variance_projection(decoder)

    def _build_rnn_memory_model(self, input_size, model_type='lstm', bias=True, dropout=0):
        """ Builds an RNN Memory Model. Currently restricted to LSTM.

        :param input_size: input size to RNN
        :param model_type: lstm or gru
        :param bias: add a bias term?
        :param dropout: dropout lstm?
        :returns: rnn module
        :rtype: nn.Module

        """
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

    def reparameterize(self, logits_map):
        """ reparameterize the encoder output and the prior

        :param logits_map: the map of logits
        :returns: a dict of reparameterized things
        :rtype: dict

        """
        nan_check_and_break(logits_map['encoder_logits'], "enc_logits")
        nan_check_and_break(logits_map['prior_logits'], "prior_logits")
        z_enc_t, params_enc_t = self.reparameterizer(logits_map['encoder_logits'])

        # XXX: clamp the variance of gaussian priors to not explode
        logits_map['prior_logits'] = self._clamp_variance(logits_map['prior_logits'])

        # reparamterize the prior distribution
        z_prior_t, params_prior_t = self.reparameterizer(logits_map['prior_logits'])

        z = {      # reparameterization
            'prior': z_prior_t,
            'posterior': z_enc_t,
            'x_features': logits_map['x_features']
        }
        params = {  # params of the reparameterization
            'prior': params_prior_t,
            'posterior': params_enc_t
        }

        return z, params

    def forward(self, input_t, **unused_kwargs):
        """ Multi-step forward pass for VRNN.

        :param input_t: input tensor or list of tensors
        :returns: final output tensor
        :rtype: torch.Tensor

        """
        decoded, params = [], []
        batch_size = input_t.shape[0] if isinstance(input_t, torch.Tensor) else input_t[0].shape[0]

        self.memory.init_state(batch_size, input_t.is_cuda)  # always re-init state at first step.
        for i in range(self.config['max_time_steps']):
            if isinstance(input_t, list):  # if we have many inputs as a list
                decode_t, params_t = self.step(input_t[i])
            else:                          # single input encoded many times
                decode_t, params_t = self.step(input_t)
                input_t = nll_activation_fn(decode_t, self.config['nll_type'])

            if i == 0:  # TODO: only use the hidden state from t=0?
                self.aggregate_posterior['rnn_hidden_state_h'](self.memory.get_state()[0])
                self.aggregate_posterior['rnn_hidden_state_c'](self.memory.get_state()[1])

            # append mutual information if requested
            params_t = self._compute_mi_params(decode_t, params_t)

            # add the params and the input to the list
            decoded.append(decode_t.clone())
            params.append(params_t)

        self.memory.clear()                # clear memory to prevent perennial growth
        return decoded, params

    def step(self, x_t, inference_only=False, **kwargs):
        """ Single step forward pass.

        :param x_i: the input tensor for time t
        :param inference_only: whether or not to run the decoding process
        :returns: decoded for current time and params for current time
        :rtype: torch.Tensor, torch.Tensor

        """
        x_t_inference = add_noise_to_imgs(x_t) \
            if self.config['add_img_noise'] else x_t     # add image quantization noise
        z_t, params_t = self.posterior(x_t_inference)

        # sanity checks
        nan_check_and_break(x_t_inference, "x_related_inference")
        nan_check_and_break(z_t['prior'], "prior")
        nan_check_and_break(z_t['posterior'], "posterior")
        nan_check_and_break(z_t['x_features'], "x_features")

        if not inference_only:                           # decode the posterior
            decoded_t = self.decode(z_t, produce_output=True)
            nan_check_and_break(decoded_t, "decoded_t")
            return decoded_t, params_t

        return None, params_t

    def decode(self, z_t, produce_output=False, update_memory=True):
        """ decodes using VRNN

        :param z_t: the latent sample
        :param produce_output: produce output or just update stae
        :returns: decoded logits
        :rtype: torch.Tensor

        """
        # grab state from RNN, TODO: evaluate recovery methods below
        # [0] grabs the h from LSTM (as opposed to (h, c))
        final_state = torch.mean(self.memory.get_state()[0], 0)

        # feature transform for z_t
        phi_z_t = self.phi_z(z_t['posterior'])

        # sanity checks
        nan_check_and_break(final_state, "final_rnn_output[decode]")
        nan_check_and_break(phi_z_t, "phi_z_t")

        if update_memory:  # concat and run through RNN to update state
            input_t = torch.cat([z_t['x_features'], phi_z_t], -1).unsqueeze(0)
            self.memory(input_t.contiguous())

        # decode only if flag is set
        dec_t = None
        if produce_output:
            dec_input_t = torch.cat([phi_z_t, final_state], -1)
            dec_t = self.decoder(dec_input_t)

        return dec_t

    def _extract_features(self, x, *xargs):
        """ accepts x and any number of extra x items and returns
            each of them projected through it's own NN,
            creating any networks as needed

        :param x: the input tensor
        :returns: the extracted features
        :rtype: torch.Tensor

        """
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
        """ lazy build the encoder based on the input size

        :param input_size: the input tensor size
        :returns: the encoder
        :rtype: nn.Module

        """
        if not hasattr(self, 'encoder'):
            self.encoder = self._get_dense_net_map(input_size, 'vrnn_enc')(
                output_size=self.reparameterizer.input_size
            )

        return self.encoder

    def encode(self, x, *xargs):
        """ single sample encode using x

        :param x: the input tensor
        :returns: dict of encoded logits
        :rtype: dict

        """
        if self.config['decoder_layer_type'] == 'pixelcnn':
            x = (x - .5) * 2.

        # get the memory trace, TODO: evaluate different recovery methods below
        final_state = torch.mean(self.memory.get_state()[0], 0)
        nan_check_and_break(final_state, "final_rnn_output")

        # extract input data features
        phi_x_t = self._extract_features(x, *xargs).squeeze()

        # encoder projection
        enc_input_t = torch.cat([phi_x_t, final_state], dim=-1)
        enc_t = self._lazy_build_encoder(enc_input_t.size(-1))(enc_input_t)

        # prior projection , consider: + eps_fn(self.config['cuda']))
        prior_t = self.prior(final_state.contiguous())

        # sanity checks
        nan_check_and_break(enc_t, "enc_t")
        nan_check_and_break(prior_t, "priot_t")

        return {
            'encoder_logits': enc_t,
            'prior_logits': prior_t,
            'x_features': phi_x_t
        }

    def _decode_and_activate(self, dec_input_t):
        """ helper to decode using the pixel-cnn or normal decoder

        :param dec_input_t: input decoded tensor (unactivated)
        :returns: activated tensor
        :rtype: torch.Tensor

        """
        return self.nll_activation(self.decoder(dec_input_t))

    def _get_prior_and_state(self, batch_size, **kwargs):
        """ Internal helper to get the final memory state and prior samples.

        :param batch_size: the number of samples to generate
        :returns: state and prior
        :rtype: (torch.Tensor, torch.Tensor)

        """
        if 'use_aggregate_posterior' in kwargs and kwargs['use_aggregate_posterior']:
            final_state = torch.mean(self.aggregate_posterior['rnn_hidden_state_h'].ema_val, 0)
            self.memory.state = (self.aggregate_posterior['rnn_hidden_state_h'].ema_val,
                                 self.aggregate_posterior['rnn_hidden_state_c'].ema_val)
            # XXX: over-ride training to get some stochasticity
            training_tmp = self.reparameterizer.training
            self.reparameterizer.train(True)

            # grab the prior using the exponential moving average.
            z_prior_t, _ = self.reparameterizer(self.aggregate_posterior['prior_logits'].ema_val)

            # XXX: reset training state to reparameterizer
            self.reparameterizer.train(training_tmp)
        else:
            final_state = torch.mean(self.memory.get_state()[0], 0)
            z_prior_t = self.reparameterizer.prior(
                batch_size, scale_var=self.config['generative_scale_var'], **kwargs
            )

        return final_state, z_prior_t

    def generate_synthetic_samples(self, batch_size, **kwargs):
        """ generate batch_size samples.

        :param batch_size: the size of the batch to generate
        :returns: generated tensor
        :rtype: torch.Tensor

        """
        if 'reset_state' in kwargs and kwargs['reset_state']:
            override_noisy_state = kwargs.get('override_noisy_state', False)
            self.memory.init_state(batch_size, cuda=self.config['cuda'],
                                   override_noisy_state=override_noisy_state)

        # grab final state and prior samples & encode them through feature extractor
        final_state, z_prior_t = self._get_prior_and_state(batch_size, **kwargs)
        phi_z_t = self.phi_z(z_prior_t)

        # run the first step of the decoding process using the prior
        dec_input_t = torch.cat([phi_z_t, final_state], -1)
        dec_output_t = self._decode_and_activate(dec_input_t)
        decoded_list = [dec_output_t.clone()]

        # iterate max_time_steps -1  times using the output from above
        for _ in range(self.config['max_time_steps'] - 1):
            dec_output_t, _ = self.step(dec_output_t, **kwargs)
            dec_output_t = nll_activation_fn(dec_output_t, self.config['nll_type'])
            decoded_list.append(dec_output_t.clone())

        return torch.cat(decoded_list, 0)

    def posterior(self, *x_args):
        """ encode the set of input tensor args

        :returns: reparam dict
        :rtype: dict

        """
        logits_map = self.encode(*x_args)
        self.aggregate_posterior['encoder_logits'](logits_map['encoder_logits'])
        self.aggregate_posterior['prior_logits'](logits_map['prior_logits'])

        return self.reparameterize(logits_map)

    def _ensure_same_size(self, prediction_list, target_list):
        """ helper to ensure that image sizes in both lists match

        :param prediction_list: the list of predictions
        :param target_list:  the list of targers
        :returns: None
        :rtype: None

        """
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
        """ KL divergence between dist_a and prior as well as constrain prior to hyper-prior

        :param dist: the distribution map
        :returns: kl divergence
        :rtype: torch.Tensor

        """
        prior_kl = self.reparameterizer.kl(dist['prior'])  \
            if self.config['use_prior_kl'] is True else 0
        return self.reparameterizer.kl(dist['posterior'], dist['prior']) + prior_kl

    def _compute_mi_params(self, recon_x_logits, params):
        """ Internal helper to compute the MI params and append to full params

        :param recon_x: reconstruction
        :param params: the original params
        :returns: original params OR param + MI_params
        :rtype: dict

        """
        if self.config['continuous_mut_info'] > 0 or self.config['discrete_mut_info'] > 0:
            _, q_z_given_xhat_params = self.posterior(self.nll_activation(recon_x_logits))
            params['posterior']['q_z_given_xhat'] = q_z_given_xhat_params['posterior']

        # base case, no MI
        return params

    def mut_info(self, dist_params, batch_size):
        """ Returns mutual information between z <-> x

        :param dist_params: the distribution dict
        :returns: tensor of dimension batch_size
        :rtype: torch.Tensor

        """
        mut_info = float_type(self.config['cuda'])(batch_size).zero_()

        # only grab the mut-info if the scalars above are set
        if (self.config['continuous_mut_info'] > 0 or self.config['discrete_mut_info'] > 0):
            mut_info = self._clamp_mut_info(self.reparameterizer.mutual_info(dist_params['posterior']))

        return mut_info

    @staticmethod
    def _add_loss_map(loss_t, loss_aggregate_map):
        """ helper to add two maps and keep counts
            of the total samples for reduction later

        :param loss_t: the loss dict
        :param loss_aggregate_map: the aggregator dict
        :returns: aggregate dict
        :rtype: dict

        """
        if loss_aggregate_map is None:
            return {**loss_t, 'count': 1}

        for (k, v) in loss_t.items():
            loss_aggregate_map[k] += v

        # increment total count
        loss_aggregate_map['count'] += 1
        return loss_aggregate_map

    @staticmethod
    def _mean_map(loss_aggregate_map):
        """ helper to reduce all values by the key count

        :param loss_aggregate_map: the aggregate dict
        :returns: count reduced dict
        :rtype: dict

        """
        for k in loss_aggregate_map.keys():
            if k == 'count':
                continue

            loss_aggregate_map[k] /= loss_aggregate_map['count']

        return loss_aggregate_map

    def loss_function(self, recon_x, x, params, K=1):
        """ evaluates the loss of the model by simply summing individual losses

        :param recon_x: the reconstruction container
        :param x: the input container
        :param params: the params dict
        :param K: number of monte-carlo samples to use.
        :returns: the mean-reduced aggregate dict
        :rtype: dict

        """
        assert len(recon_x) == len(params)
        assert K == 1, "Monte carlo sampling for decoding not implemented for VRNN"

        # case where only 1 data sample, but many posteriors
        if not isinstance(x, list) and len(x) != len(recon_x):
            x = [x.clone() for _ in range(len(recon_x))]

        # aggregate the loss many and return the mean of the map
        loss_aggregate_map = None
        for recon_x, x, p in zip(recon_x, x, params):
            loss_t = super(VRNN, self).loss_function(recon_x, x, p, K=K)
            loss_aggregate_map = self._add_loss_map(loss_t, loss_aggregate_map)

        return self._mean_map(loss_aggregate_map)

    def get_activated_reconstructions(self, reconstr_container):
        """ Returns activated reconstruction

        :param reconstr: unactivated reconstr logits list
        :returns: activated reconstr
        :rtype: dict

        """
        recon_dict = {}
        for i, recon in enumerate(reconstr_container):
            recon_dict['reconstruction{}_imgs'.format(i)] = self.nll_activation(recon)

        return recon_dict
