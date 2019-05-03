from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy

from helpers.utils import float_type
from helpers.layers import get_encoder
from .reparameterizers.sequential_reparameterizer import SequentialReparameterizer
from .parallelly_reparameterized_vae import ParallellyReparameterizedVAE


class SequentiallyReparameterizedVAE(ParallellyReparameterizedVAE):
    def __init__(self, input_shape, reparameterizer_strs=["discrete", "isotropic_gaussian"], **kwargs):
        """ Implements a set of latent reparameterized variables, eg, disc --> NN --> gauss.

        :param input_shape: the input tensor shape
        :param reparameterizer_strs: the type of reparameterizers to use
        :returns: SequentiallyReparameterizedVAE object
        :rtype: nn.Module

        """
        kwargs['lazy_init_reparameterizer'] = True
        kwargs['lazy_init_decoder'] = True
        kwargs['lazy_init_encoder'] = True
        super(SequentiallyReparameterizedVAE, self).__init__(input_shape, **kwargs)
        self.reparameterizer_strs = reparameterizer_strs
        self.reparameterizer = SequentialReparameterizer(reparameterizer_strs, self.config)

        # build the encoder and decoder here because of sizing
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def has_discrete(self):
        ''' True is our FIRST parameterize is discrete '''
        return isinstance(self.reparameterizer.reparameterizers[0], GumbelSoftmax)

    def get_reparameterizer_scalars(self):
        """ return the reparameterization scalars (eg: tau in gumbel)

        :returns: a dict of scalars
        :rtype: dict

        """
        return self.reparameterizer.get_reparameterizer_scalars()
