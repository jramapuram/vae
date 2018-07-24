# VAE Pool

This repository is to be used as a submodule in projects where VAE's are required.
It contains:

  1. Sequentially reparameterized VAE (i.e. z0 --> z1 --> .. )
  2. Parallelly reparameterized VAE (i.e. x --> [z0, z1] --> ..)
  3. VRNN

All of the VAE implementations inherit from `AbstractVAE.py` which acts as a base abstraction.

## Things to get a basic version working

You will need the helpers submodule from https://github.com/jramapuram/helpers.git
This is needed as it provides basic functions such as utils and network builders.
VAE's expect this helpers submodule to be at the baselevel of the repository.
