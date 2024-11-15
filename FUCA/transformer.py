import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def attention(self, Q, K, V, mask = None):
        # Q, K, V shapes: (num_agents, num_heads, d_k)
        scores = torch.matmul(Q.unsqueeze(-2), K.unsqueeze(-1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        return torch.matmul(attention_weights, V.unsqueeze(-1)).squeeze(-1), attention_weights

    def forward(self, Q, K, V, mask=None):
        # Input shape: (num_agents, d_model)
        num_agents = Q.size(0)

        # Linear transformations and split into heads
        Q = self.W_q(Q).view(num_agents, self.num_heads, self.d_k)
        K = self.W_k(K).view(num_agents, self.num_heads, self.d_k)
        V = self.W_v(V).view(num_agents, self.num_heads, self.d_k)

        # Apply attention
        x, attention_weights = self.attention(Q, K, V, mask)

        # Reshape and apply final linear transformation
        x = x.contiguous().view(num_agents, self.d_model)
        return self.W_o(x)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # x shape: (num_agents, d_model)
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

class PopulationTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention and residual connection
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))

        # Feed forward and residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class PopulationTransformer(nn.Module):
    def __init__(self,
                 input_dim,           # Dimension of agent's sensory input
                 output_dim,          # Dimension of agent's actions
                 d_model=64,          # Embedding dimension
                 num_heads=4,         # Number of attention heads
                 num_layers=3,        # Number of transformer blocks
                 d_ff=256,           # Feed-forward dimension
                 dropout=0.1):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            PopulationTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, x, mask=None):
        """
        Forward pass for the entire population

        Args:
            x: Input tensor of shape (num_agents, input_dim)
            mask: Optional attention mask for agent interactions

        Returns:
            Output tensor of shape (num_agents, output_dim)
        """
        # Project input to transformer dimension
        x = self.input_projection(x)
        x = self.dropout(x)

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        # Project to output dimension
        return self.output_projection(x)

# Example usage:
def simulate_step(population_transformer, sensory_inputs):
    """
    Simulate a single step for the population

    Args:
        population_transformer: The transformer model
        sensory_inputs: Tensor of shape (num_agents, input_dim)

    Returns:
        Actions for all agents
    """
    with torch.no_grad():  # No gradient computation during simulation
        actions = population_transformer(sensory_inputs)
    return actions

# Example initialization and usage:
"""
# Initialize transformer for a population
transformer = PopulationTransformer(
    input_dim=10,    # Size of each agent's sensory input
    output_dim=2,    # Size of each agent's action space (e.g., 2D movement)
    d_model=64,      # Internal representation size
    num_heads=4      # Number of attention heads
)

# Simulate one step with 100 agents
num_agents = 100
input_dim = 10      # Size of sensory input

# Create sensory inputs for all agents
sensory_inputs = torch.randn(num_agents, input_dim)

# Get actions for all agents
actions = simulate_step(transformer, sensory_inputs)
# actions shape: (num_agents, output_dim)
"""
