import torch
import torch.nn as nn

from helpers.layers import build_pixelcnn_decoder
from helpers.distributions import nll_activation, \
     nll_activation_with_variance


class PixelCNN(nn.Module):
    def __init__(self, input_shape, config):
        super(PixelCNN, self).__init__()
        self.input_shape = input_shape
        self.config = config
        self.chans = input_shape[0]

        # use the variance in the likelihood if it exists
        self.nll_fn = nll_activation_with_variance if self.config['nll_type'] == 'gaussian' \
            or self.config['nll_type'] == 'clamp' else nll_activation

        # build the pixel-cnn, input is x_dec decoded from z,
        # concated with the auto-regressive x , i.e. P(x_i | x_{i-1}, x_dec)
        self.pixel_cnn = build_pixelcnn_decoder(input_size=self.chans,
                                                output_shape=self.input_shape,
                                                normalization_str=self.config['normalization'])

    def forward(self, x_decoded_logits):
        batch_size = x_decoded_logits.size(0)
        dtype = 'float32' if not self.config['half'] else 'float16'

        # activate the logits with the proper likelihood
        x_decoded_activated = self.nll_fn(x_decoded_logits,
                                          self.config['nll_type'])

        # create the pixel cnn buffer
        x_buffer = torch.zeros_like(x_decoded_activated)

        # what we want to return is actually the mean generally
        # but use the variance for the generation of the buffer
        mean = None

        # now loop over the columns and rows
        for w in range(self.input_shape[-2]):
            for h in range(self.input_shape[-1]):
                # concat on channel with standard decoder output
                pixel_inputs = torch.cat([x_decoded_activated, x_buffer], 1)
                pixel_output_logits = self.pixel_cnn(pixel_inputs)
                pixel_output_nll = self.nll_fn(
                    pixel_output_logits,
                    self.config['nll_type']
                )
                x_buffer[:, :, w, h] = pixel_output_nll[:, :, w, h]

                if self.config['nll_type'] == 'gaussian' \
                   or self.config['nll_type'] == 'clamp':
                    num_half_chans = pixel_output_logits.size(1) // 2
                    mean = pixel_output_logits[:, 0:num_half_chans, :, :]
                else:
                    mean = pixel_output_logits

        return mean
