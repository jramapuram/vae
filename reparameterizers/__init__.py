from .beta import Beta
from .mixture import Mixture
from .bernoulli import Bernoulli
from .gumbel import GumbelSoftmax
from .isotropic_gaussian import IsotropicGaussian
from .concat_reparameterizer import ConcatReparameterizer
from .sequential_reparameterizer import SequentialReparameterizer


def get_reparameterizer(reparam_type_str):
    """ Returns a reparameterizer type based on the string

    :param reparam_type_str: the type of reparam
    :returns: a reparam object
    :rtype: nn.Module

    """
    reparam_dict = {
        'beta': Beta,
        'bernoulli': Bernoulli,
        'discrete': GumbelSoftmax,
        'isotropic_gaussian': IsotropicGaussian,
        'mixture': Mixture,
        'concat': ConcatReparameterizer,
        'sequential': SequentialReparameterizer
    }
    return reparam_dict[reparam_type_str]
