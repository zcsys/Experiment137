import torch

class MultiHeadAttention:
    def __init__(self, d_model, num_heads, num_monads):
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_monads = num_monads

    def setup_weights(self, weights_start, weights):
        d2 = self.d_model ** 2
        self.W_q = weights[
            :, weights_start:weights_start + d2
        ].view(self.num_monads, self.d_model, self.d_model)
        self.W_k = weights[
            :, weights_start + d2:weights_start + 2 * d2
        ].view(self.num_monads, self.d_model, self.d_model)
        self.W_v = weights[
            :, weights_start + 2 * d2:weights_start + 3 * d2
        ].view(self.num_monads, self.d_model, self.d_model)
        self.W_o = weights[
            :, weights_start + 3 * d2:weights_start + 4 * d2
        ].view(self.num_monads, self.d_model, self.d_model)

    def forward(self, inputs):
        Q = torch.bmm(
            inputs.view(self.num_monads, 1, self.d_model), self.W_q
        ).view(self.num_monads, self.num_heads, self.d_k)
        K = torch.bmm(
            inputs.view(self.num_monads, 1, self.d_model), self.W_k
        ).view(self.num_monads, self.num_heads, self.d_k)
        V = torch.bmm(
            inputs.view(self.num_monads, 1, self.d_model), self.W_v
        ).view(self.num_monads, self.num_heads, self.d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention = torch.softmax(scores, dim = -1)
        attended = torch.matmul(attention, V).view(self.num_monads,
                                                   self.d_model)

        return torch.bmm(
            attended.view(self.num_monads, 1, self.d_model), self.W_o
        ).view(self.num_monads, self.d_model)

class TransformerLayer:
    def __init__(self, d_model, num_heads, d_ff, num_monads):
        self.attention = MultiHeadAttention(d_model, num_heads, num_monads)
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_monads = num_monads
        self.d2 = d_model ** 2
        self.increment = self.d_model * self.d_ff

    def setup_weights(self, weights_start, weights):
        # Multi-head attention
        attention_weights_size = 4 * (self.d2 + self.d_model)
        self.attention.setup_weights(weights_start, weights)
        current_pos = weights_start + attention_weights_size

        # Feed Forward weights
        self.ff1 = weights[
            :, current_pos:current_pos + self.increment
        ].view(self.num_monads, self.d_model, self.d_ff)
        current_pos += self.increment

        self.ff2 = weights[
            :, current_pos:current_pos + self.increment
        ].view(self.num_monads, self.d_ff, self.d_model)

    def forward(self, inputs):
        attended = self.attention.forward(inputs)

        residual1 = inputs + attended

        ff_hidden = torch.nn.functional.gelu(
            torch.bmm(
                residual1.view(self.num_monads, 1, self.d_model),
                self.ff1
            )
        ).view(self.num_monads, self.d_ff)

        ff_out = torch.bmm(
            ff_hidden.view(self.num_monads, 1, self.d_ff),
            self.ff2
        ).view(self.num_monads, self.d_model)

        return residual1 + ff_out

    def weights_per_layer(self):
        return 4 * self.d2 + 2 * self.increment

class Transformer:
    def __init__(self, d_model, num_heads, num_layers, d_ff, input_dim,
                 output_dim, num_monads):
        self.layers = [TransformerLayer(d_model, num_heads, d_ff, num_monads)
                      for _ in range(num_layers)]
        self.d_model = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_monads = num_monads

    def setup_weights(self, weights):
        # Number of weights per monad: d_model * (input_dim + output_dim) +
        #                              num_layers * weights_per_layer()
        # 2*(4*16*16+2*16*16*4)+16*(16+9)
        increment = self.input_dim * self.d_model
        layer_weights = self.layers[0].weights_per_layer()

        # Input projection
        self.input_proj = weights[
            :, :increment
        ].view(self.num_monads, self.input_dim, self.d_model)
        current_pos = increment

        # Transformer layers
        for layer in self.layers:
            layer.setup_weights(current_pos, weights)
            current_pos += layer_weights

        # Output projection
        self.output_proj = weights[
            :, current_pos:current_pos + self.d_model * self.output_dim
        ].view(self.num_monads, self.d_model, self.output_dim)

    def forward(self, inputs):
        # Input projection
        x = torch.bmm(
            inputs.view(self.num_monads, 1, self.input_dim),
            self.input_proj
        ).view(self.num_monads, self.d_model)

        # Transformer layers
        for layer in self.layers:
            x = layer.forward(x)

        # Output projection
        return torch.bmm(
            x.view(self.num_monads, 1, self.d_model),
            self.output_proj
        ).view(self.num_monads, self.output_dim)