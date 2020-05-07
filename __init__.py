import functools

from .msg import MSGVAE
from .vrnn import VRNN
from .additive_vrnn import AdditiveVRNN
from .simple_vae import SimpleVAE
from .conditioned import ClassConditionedPriorVAE
from .autoencoder import Autoencoder, VAENoKL
from .parallelly_reparameterized_vae import ParallellyReparameterizedVAE
from .sequentially_reparameterized_vae import SequentiallyReparameterizedVAE


def build_vae(vae_type_str):
    """ Simple helper to return the correct VAE class.

    :param vae_type_str: the string vae type
    :returns: an AbstractVAE object
    :rtype: AbstractVAE

    """
    vae_type_map = {
        'autoencoder': Autoencoder,
        'vaenokl': VAENoKL,
        'simple': SimpleVAE,
        'msg': MSGVAE,
        'vrnn': VRNN,
        'additive_vrnn': AdditiveVRNN,
        'parallel': ParallellyReparameterizedVAE,
        'sequential': SequentiallyReparameterizedVAE,
        'class_conditioned': ClassConditionedPriorVAE,
    }

    # Special logic to split our the reparams for the parallel or sequential vae models.
    if 'parallel' in vae_type_str or 'sequential' in vae_type_str:
        assert '+' in vae_type_str, "Specify sequential+bernoulli+discrete+... (or parallel+) as type."
        vae_base_type = vae_type_str.split('+')[0]
        reparams = vae_type_str.split('+')[1:]
        return functools.partial(vae_type_map[vae_base_type], reparameterizer_strs=reparams)

    return vae_type_map[vae_type_str]
