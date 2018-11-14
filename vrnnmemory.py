import torch
import functools
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

        self.state = (  # LSTM state is (h, c)
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
