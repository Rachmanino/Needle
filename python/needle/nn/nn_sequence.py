"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module, ReLU, Tanh


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(-x)) ** -1
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.activation = ReLU() if nonlinearity == 'relu' else Tanh()
        self.W_ih = Parameter(init.rand(input_size,
                                        hidden_size, 
                                        low=-np.sqrt(1/hidden_size),
                                        high=np.sqrt(1/hidden_size), 
                                        device=device, 
                                        requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size,
                                        hidden_size, 
                                        low=-np.sqrt(1/hidden_size),
                                        high=np.sqrt(1/hidden_size), 
                                        device=device, 
                                        requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, 
                                               low=-np.sqrt(1/hidden_size),
                                               high=np.sqrt(1/hidden_size), 
                                               device=device, 
                                               requires_grad=True))
            self.bias_hh = Parameter(init.rand(hidden_size,  
                                               low=-np.sqrt(1/hidden_size),
                                               high=np.sqrt(1/hidden_size), 
                                               device=device, 
                                               requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None: #!这里不能h=None时就不参加计算,否则有Parameter无梯度
            h = init.zeros(X.shape[0], self.hidden_size, 
                           device=X.device, 
                           dtype=X.dtype, 
                           requires_grad=True)
        out = X @ self.W_ih + h @ self.W_hh
        if self.bias:
            out += self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(out.shape) + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(out.shape)
        return self.activation(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype
        assert num_layers > 0
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)]
        for _ in range(num_layers-1):
            self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        output = []
        X_seq = ops.split(X, 0)
        if h0:
            h_seq = ops.split(h0, 0)
        else:
            h_seq = [None] * self.num_layers
        for x in X_seq:
            hidden = []
            for j, cell in enumerate(self.rnn_cells):
                x = cell(x, h_seq[j])
                hidden.append(x)
            output.append(x)
            h_seq = hidden
        return ops.stack(output, 0), ops.stack(hidden, 0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.W_ih = Parameter(init.rand(input_size,
                                        4*hidden_size, 
                                        low=-np.sqrt(1/hidden_size),
                                        high=np.sqrt(1/hidden_size), 
                                        device=device, 
                                        requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size,
                                        4*hidden_size, 
                                        low=-np.sqrt(1/hidden_size),
                                        high=np.sqrt(1/hidden_size), 
                                        device=device, 
                                        requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(4*hidden_size, 
                                               low=-np.sqrt(1/hidden_size),
                                               high=np.sqrt(1/hidden_size), 
                                               device=device, 
                                               requires_grad=True))
            self.bias_hh = Parameter(init.rand(4*hidden_size,  
                                               low=-np.sqrt(1/hidden_size),
                                               high=np.sqrt(1/hidden_size), 
                                               device=device, 
                                               requires_grad=True))
        self.bias = bias
        self.sig = Sigmoid()
        self.tanh = Tanh()
        self.hidden_size = hidden_size
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        h0, c0 = h if h else (None, None)
        if h0 is None: #!这里不能h=None时就不参加计算,否则有Parameter无梯度
            h0 = init.zeros(X.shape[0], self.hidden_size, 
                           device=X.device, 
                           dtype=X.dtype, 
                           requires_grad=True)
        out = X @ self.W_ih + h0 @ self.W_hh
        if self.bias:
            out += self.bias_ih.reshape((1, 4*self.hidden_size)).broadcast_to(out.shape) + self.bias_hh.reshape((1, 4*self.hidden_size)).broadcast_to(out.shape)
        out_list = ops.split(out, 1)
        i = self.sig(ops.stack(tuple([out_list[i] for i in range(0, self.hidden_size)]), 1))
        f = self.sig(ops.stack(tuple([out_list[i] for i in range(self.hidden_size, 2*self.hidden_size)]), 1))
        g = self.tanh(ops.stack(tuple([out_list[i] for i in range(2*self.hidden_size, 3*self.hidden_size)]), 1))
        o = self.sig(ops.stack(tuple([out_list[i] for i in range(3*self.hidden_size, 4*self.hidden_size)]), 1))
        if c0 is None:
            c0 = init.zeros(X.shape[0], self.hidden_size, 
                           device=X.device, 
                           dtype=X.dtype, 
                           requires_grad=True)
        c = i * g + f * c0
        h = o * self.tanh(c)
        return h, c
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype
        assert num_layers > 0
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)]
        for _ in range(num_layers-1):
            self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        output = []
        h0, c0 = h if h else (None, None)
        X_seq = ops.split(X, 0)
        if h:
            h_seq = ops.split(h0, 0)
            c_seq = ops.split(c0, 0)
        else:
            h_seq = [None] * self.num_layers
            c_seq = [None] * self.num_layers

        for x in X_seq:
            hidden, cell_out = [], []
            for j, cell in enumerate(self.lstm_cells):
                x, c = cell(x, (h_seq[j], c_seq[j]))
                hidden.append(x)
                cell_out.append(c)
            output.append(x)
            h_seq = hidden
            c_seq = cell_out
        return ops.stack(output, 0), (ops.stack(hidden, 0), ops.stack(cell_out, 0))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            init.randn(num_embeddings, embedding_dim, 
                       device=device, 
                       requires_grad=True)
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        onehot_x = init.one_hot(self.num_embeddings, x,
                                device=x.device,
                                dtype=x.dtype,
                                requires_grad=True)
        seq_len, bs, _ = onehot_x.shape
        return (onehot_x.reshape((seq_len*bs, self.num_embeddings)) @ self.weight).reshape(
            (seq_len, bs, self.embedding_dim)
        )
        ### END YOUR SOLUTION