"""
All code is from https://github.com/speechbrain/speechbrain

speechbrain/nnet/RNN/RNN
speechbrain/nnet/linear/Linear

Library implementing recurrent neural networks.

Authors
 * Mirco Ravanelli 2020
 * Ju-Chieh Chou 2020
 * Jianyuan Zhong 2020
 * Loren Lugosch 2020
"""

import torch
import torch.nn

class Linear(torch.nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape: tuple
        It is the shape of the input tensor.
    input_size: int
        Size of the input tensor.
    bias : bool
        If True, the additive bias b is adopted.
    combine_dims : bool
        If True and the input is 4D, combine 3rd and 4th dimensions of input.

    Example
    -------
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
    >>> output = lin_t(inputs)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(
        self,
        n_neurons,
        input_shape=None,
        input_size=None,
        bias=True,
        combine_dims=False,
    ):
        super().__init__()
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]

        # Weights are initialized following pytorch approach
        self.w = torch.nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly.
        """
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        wx = self.w(x)

        return wx

class RNN(torch.nn.Module):
    """This function implements a vanilla RNN.

    It accepts in input tensors formatted as (batch, time, fea).
    In the case of 4d inputs like (batch, time, fea, channel) the tensor is
    flattened as (batch, time, fea*channel).

    Arguments
    ---------
    hidden_size : int
        Number of output neurons (i.e, the dimensionality of the output).
        values (i.e, time and frequency kernel sizes respectively).
    input_shape : tuple
        The shape of an example input. Alternatively, use ``input_size``.
    input_size : int
        The size of the input. Alternatively, use ``input_shape``.
    nonlinearity : str
        Type of nonlinearity (tanh, relu).
    num_layers : int
        Number of layers to employ in the RNN architecture.
    bias : bool
        If True, the additive bias b is adopted.
    dropout : float
        It is the dropout factor (must be between 0 and 1).
    re_init : bool
        If True, orthogonal initialization is used for the recurrent weights.
        Xavier initialization is used for the input connection weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both
        right-to-left and left-to-right is used.

    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = RNN(hidden_size=5, input_shape=inp_tensor.shape)
    >>> out_tensor, _ = net(inp_tensor)
    >>>
    torch.Size([4, 10, 5])
    """

    def __init__(
        self,
        hidden_size,
        input_shape=None,
        input_size=None,
        nonlinearity="relu",
        num_layers=1,
        bias=True,
        dropout=0.0,
        re_init=True,
        bidirectional=False,
    ):
        super().__init__()
        self.reshape = False

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size.")

        # Computing the feature dimensionality
        if input_size is None:
            if len(input_shape) > 3:
                self.reshape = True
            input_size = torch.prod(torch.tensor(input_shape[2:]))

        self.rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            bias=bias,
            batch_first=True,
            nonlinearity=nonlinearity,
        )

        if re_init:
            rnn_init(self.rnn)

    def forward(self, x, hx=None, lengths=None):
        """Returns the output of the vanilla RNN.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        hx : torch.Tensor
            Starting hidden state.
        lengths : torch.Tensor
            Relative lengths of the input signals.
        """
        # Reshaping input tensors for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # Flatten params for data parallel
        self.rnn.flatten_parameters()

        # Pack sequence for proper RNN handling of padding
        if lengths is not None:
            x = pack_padded_sequence(x, lengths)

        # Support custom initial state
        if hx is not None:
            output, hn = self.rnn(x, hx=hx)
        else:
            output, hn = self.rnn(x)

        # Unpack the packed sequence
        if lengths is not None:
            output = pad_packed_sequence(output)

        return output, hn


def pad_packed_sequence(inputs):
    """Returns speechbrain-formatted tensor from packed sequences.

    Arguments
    ---------
    inputs : torch.nn.utils.rnn.PackedSequence
        An input set of sequences to convert to a tensor.
    """
    outputs, lengths = torch.nn.utils.rnn.pad_packed_sequence(
        inputs, batch_first=True
    )
    return outputs


def pack_padded_sequence(inputs, lengths):
    """Returns packed speechbrain-formatted tensors.

    Arguments
    ---------
    inputs : torch.Tensor
        The sequences to pack.
    lengths : torch.Tensor
        The length of each sequence.
    """
    lengths = (lengths * inputs.size(1)).cpu()
    return torch.nn.utils.rnn.pack_padded_sequence(
        inputs, lengths, batch_first=True, enforce_sorted=False
    )


def rnn_init(module):
    """This function is used to initialize the RNN weight.
    Recurrent connection: orthogonal initialization.

    Arguments
    ---------
    module: torch.nn.Module
        Recurrent neural network module.

    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = RNN(hidden_size=5, input_shape=inp_tensor.shape)
    >>> out_tensor = net(inp_tensor)
    >>> rnn_init(net)
    """
    for name, param in module.named_parameters():
        if "weight_hh" in name or ".u.weight" in name:
            torch.nn.init.orthogonal_(param)


