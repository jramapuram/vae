from .msg import MSGVAE
from .vrnn import VRNN
from .additive_vrnn import AdditiveVRNN
from .simple_vae import SimpleVAE
from .symmetric_simple_vae import SymmetricSimpleVAE
from .parallelly_reparameterized_vae import ParallellyReparameterizedVAE
from .sequentially_reparameterized_vae import SequentiallyReparameterizedVAE

def build_vae(vae_type_str):
    """ Simple helper to return the correct VAE class.

    :param vae_type_str: the string vae type
    :returns: an AbstractVAE object
    :rtype: AbstractVAE

    """
    vae_type_map = {
        'simple': SimpleVAE,
        'msg': MSGVAE,
        'vrnn': VRNN,
        'additive_vrnn': AdditiveVRNN,
        'symmetric_simple': SymmetricSimpleVAE,
        'parallel': ParallellyReparameterizedVAE,
        'sequential': SequentiallyReparameterizedVAE
    }
    return vae_type_map[vae_type_str]
