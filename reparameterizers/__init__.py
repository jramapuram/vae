from .beta import Beta
from .mixture import Mixture
from .bernoulli import Bernoulli
from .gumbel import GumbelSoftmax
from .isotropic_gaussian import IsotropicGaussian
from .concat_reparameterizer import ConcatReparameterizer
from .sequential_reparameterizer import SequentialReparameterizer

from .flow import FlowReparameterizer


reparam_dict = {
    'flow': FlowReparameterizer,
    'beta': Beta,
    'bernoulli': Bernoulli,
    'discrete': GumbelSoftmax,
    'isotropic_gaussian': IsotropicGaussian,
    'mixture': Mixture,
    'concat': ConcatReparameterizer,
    'sequential': SequentialReparameterizer
}


def get_reparameterizer(reparam_type_str):
    """ Returns a reparameterizer type based on the string

    :param reparam_type_str: the type of reparam
    :returns: a reparam object
    :rtype: nn.Module

    """
    assert reparam_type_str in reparam_dict, "Unknown reparameterizer requested: {}".format(
        reparam_type_str)
    return reparam_dict[reparam_type_str]


def is_module_a_reparameterizer(module):
    """Returns true if the provided torch module is a reparamterizer

    :param module: nn.Module, etc.
    :returns: true or false
    :rtype: bool

    """
    module_types = tuple(reparam_dict.values())
    return isinstance(module, module_types)
