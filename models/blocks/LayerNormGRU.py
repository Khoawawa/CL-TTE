import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNormGRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__()

        self.ln_i2h = torch.nn.LayerNorm(2*hidden_size, elementwise_affine=False)
        self.ln_h2h = torch.nn.LayerNorm(2*hidden_size, elementwise_affine=False)
        self.ln_cell_1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln_cell_2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.i2h = torch.nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.h_hat_W = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.h_hat_U = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x, h):

        h = h.view(h.size(0), -1)
        x = x.view(x.size(0), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)

        # Layer norm
        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)

        preact = i2h + h2h

        # activations
        gates = preact[:, :].sigmoid()
        z_t = gates[:, :self.hidden_size]
        r_t = gates[:, -self.hidden_size:]

        # h_hat
        h_hat_first_half = self.h_hat_W(x)
        h_hat_last_half = self.h_hat_U(h)

        # layer norm
        h_hat_first_half = self.ln_cell_1( h_hat_first_half )
        h_hat_last_half = self.ln_cell_2( h_hat_last_half )

        h_hat = torch.tanh(  h_hat_first_half + torch.mul(r_t,   h_hat_last_half ) )

        h_t = torch.mul( 1-z_t , h ) + torch.mul( z_t, h_hat)

        # Reshape for compatibility

        h_t = h_t.view( h_t.size(0), -1)
        return h_t

class LayerNormGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, bias=True):
        super(LayerNormGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.hidden0 = nn.ModuleList([
            LayerNormGRUCell(
                input_size=(input_dim if layer == 0 else hidden_dim), 
                hidden_size=hidden_dim, 
                bias=bias
            ) for layer in range(num_layers)
        ])

    def forward(self, input: torch.Tensor, seq_lens=None, hx=None):
        device = input.device
        seq_len, batch_size, _ = input.size()
        
        # --- Handle Initial Hidden State ---
        if hx is None:
            # Default to zeros if no h0 is provided
            h = input.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        else:
            # hx should be (num_layers, batch_size, hidden_dim)
            h = hx

        seq_lens = seq_lens.to(device).long() if seq_lens is not None else None
        
        # Storage for all time steps (for 'y' output)
        ht_all_steps = []

        # Create mask for variable length sequences
        seq_len_mask = input.new_ones(batch_size, seq_len, self.hidden_dim)
        if seq_lens is not None:
            for i, l in enumerate(seq_lens):
                if l < seq_len:
                    seq_len_mask[i, l:, :] = 0
        seq_len_mask = seq_len_mask.transpose(0, 1) # (seq_len, batch, hidden)

        # Loop through time
        for t, x in enumerate(input):
            current_layer_hiddens = []
            layer_input = x
            
            for l, layer in enumerate(self.hidden0):
                # Process cell
                h_next = layer(layer_input, h[l])
                
                # Apply mask: if current step t >= seq_len, keep the old hidden state 
                # or zero it out based on your masking preference. 
                # Here we follow your logic: mask current output.
                h_masked = h_next * seq_len_mask[t]
                
                current_layer_hiddens.append(h_masked)
                # Input for the next layer is the output of this layer
                layer_input = h_masked
            
            # Update h for the next time step
            h = torch.stack(current_layer_hiddens) 
            ht_all_steps.append(h)

        # y: The output of the last layer for all time steps
        # shape: (seq_len, batch_size, hidden_dim)
        y = torch.stack([step_h[-1] for step_h in ht_all_steps])
        
        # hy: The hidden state at the last valid time step for each layer
        # Use gather to pick the hidden state at index (lens - 1)
        indices = (seq_lens - 1).view(1, batch_size, 1).expand(self.num_layers, -1, self.hidden_dim)
        all_hiddens_tensor = torch.stack(ht_all_steps) # (seq_len, num_layers, batch, hidden)
        
        # Permute to (num_layers, seq_len, batch, hidden) to gather over time
        all_hiddens_tensor = all_hiddens_tensor.permute(1, 0, 2, 3)
        hy = all_hiddens_tensor.gather(dim=1, index=indices.unsqueeze(1)).squeeze(1)

        return y, hy
        
class LayerNormBiGRUCell(torch.nn.Module):
    def __init__(self, hidden_size, bias=True):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.h2h = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size, bias=bias),
            nn.LayerNorm(2 * hidden_size, elementwise_affine=False)
        )
        
        self.ln_cell_2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.h_hat_U = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, pre_i2h, pre_h_hat, h):

        h2h = self.h2h(h)
        preact = pre_i2h + h2h

        # activations
        gates = preact.sigmoid()
        z_t = gates[:, :self.hidden_size]
        r_t = gates[:, -self.hidden_size:]

        # h_hat
        h_hat_last_half = self.ln_cell_2(self.h_hat_U(h))
        h_hat = torch.tanh(pre_h_hat + torch.mul(r_t, h_hat_last_half))
        
        h_t = torch.mul(1 - z_t, h) + torch.mul(z_t, h_hat)
        
        return h_t # [batch_size, hidden_size]


class LayerNormBiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers = 2, bias=True):
        super().__init__()

        self.input_dim = input_dim
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers
        
        self.forward_cells = nn.ModuleList()
        self.backward_cells = nn.ModuleList()
        
        self.i2h_fwd = nn.ModuleList()
        self.h_hat_W_fwd = nn.ModuleList()
        self.ln_i2h_fwd = nn.ModuleList()
        self.ln_cell_1_fwd = nn.ModuleList()
        
        self.i2h_bwd = nn.ModuleList()
        self.h_hat_W_bwd = nn.ModuleList()
        self.ln_i2h_bwd = nn.ModuleList()
        self.ln_cell_1_bwd = nn.ModuleList()
        
        for layer in range(num_layers):
            # Input dimension is `input_dim` for layer 0, and `2 * hidden_dim` for subsequent layers
            layer_in_dim = input_dim if layer == 0 else 2 * hidden_dim

            # Setup Forward Components
            self.forward_cells.append(LayerNormBiGRUCell(hidden_dim, bias=bias))
            self.i2h_fwd.append(nn.Linear(layer_in_dim, 2 * hidden_dim, bias=bias))
            self.h_hat_W_fwd.append(nn.Linear(layer_in_dim, hidden_dim, bias=bias))
            self.ln_i2h_fwd.append(nn.LayerNorm(2 * hidden_dim, elementwise_affine=False))
            self.ln_cell_1_fwd.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))

            # Setup Backward Components
            self.backward_cells.append(LayerNormBiGRUCell(hidden_dim, bias=bias))
            self.i2h_bwd.append(nn.Linear(layer_in_dim, 2 * hidden_dim, bias=bias))
            self.h_hat_W_bwd.append(nn.Linear(layer_in_dim, hidden_dim, bias=bias))
            self.ln_i2h_bwd.append(nn.LayerNorm(2 * hidden_dim, elementwise_affine=False))
            self.ln_cell_1_bwd.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))


    def forward(self, input: torch.Tensor, seq_lens=None):
        
        L, B, _ = input.size()
        device = input.device
        
        if seq_lens is None:
            seq_lens = torch.full((B,), L, dtype=torch.long, device=device)
        else:
            seq_lens = seq_lens.to(device).long()
        
        mask = (torch.arange(L, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)).transpose(0, 1)  # L,B
        mask = mask.unsqueeze(-1).float()
        
        layer_input = input
        all_hy_fwd = []
        all_hy_bwd = []
        
        batch_indices = torch.arange(B, device=device)
        
        for l in range(self.num_layers):
            seq_i2h_fwd = self.ln_i2h_fwd[l](self.i2h_fwd[l](layer_input))
            seq_h_hat_fwd = self.ln_cell_1_fwd[l](self.h_hat_W_fwd[l](layer_input))

            seq_i2h_bwd = self.ln_i2h_bwd[l](self.i2h_bwd[l](layer_input))
            seq_h_hat_bwd = self.ln_cell_1_bwd[l](self.h_hat_W_bwd[l](layer_input))
            
            h_fwd = torch.zeros(B, self.hidden_dim, device=device)
            h_bwd = torch.zeros(B, self.hidden_dim, device=device)
            
            fwd_outputs = []
            bwd_outputs = [None] * L
            
            for t in range(L):
                m_t = mask[t]
                h_new  = self.forward_cells[l](seq_i2h_fwd[t], seq_h_hat_fwd[t], h_fwd)
                h_fwd = h_new * m_t + h_fwd * (1 - m_t) # Apply padding mask
                fwd_outputs.append(h_fwd)
            
            for t in reversed(range(L)):
                m_t = mask[t]
                h_new = self.backward_cells[l](seq_i2h_bwd[t], seq_h_hat_bwd[t], h_bwd)
                h_bwd = h_new * m_t + h_bwd * (1 - m_t) # Keeps state at 0 until first valid token is hit from right
                bwd_outputs[t] = h_bwd
            fwd_outputs = torch.stack(fwd_outputs)
            bwd_outputs = torch.stack(bwd_outputs)
            layer_input = torch.cat([fwd_outputs, bwd_outputs], dim=-1)
            
            last_fwd = fwd_outputs[seq_lens - 1, batch_indices, :]
            all_hy_fwd.append(last_fwd)

            last_bwd = bwd_outputs[0, batch_indices, :] # Backward always ends at index 0
            all_hy_bwd.append(last_bwd)
            
        y = layer_input 

        # Format final hidden states `hy` to match nn.GRU output: 
        # (2 * num_layers, batch_size, hidden_dim) -> [fwd_L0, bwd_L0, fwd_L1, bwd_L1, ...]
        hy = []
        for l in range(self.num_layers):
            hy.append(all_hy_fwd[l])
            hy.append(all_hy_bwd[l])
        hy = torch.stack(hy)

        return y, hy

'''
test the module
'''
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable
def is_equal(a, b, epsilon=1e-5):
    return torch.all(torch.lt(torch.abs(torch.add(a, -b)), epsilon)).item() == 1
def test_GRU():
    pass
def test_layernorm_LSTMCell():
    batch_size = 4
    hidden_size = 2
    num_input_features = 3
    # create two objects
    rnn = LayerNormGRUCell(num_input_features, hidden_size, bias=True)
    rnn_old = torch.nn.GRUCell(num_input_features, hidden_size, bias=True)
    # initialize two objects with same weights & biases
    for param in rnn_old.named_parameters():
        rnn.register_parameter(param[0], param[1])
    # initialize the hidden state
    states = torch.tensor(torch.zeros(batch_size, hidden_size,1))
    # create the input data
    input_tensor = torch.FloatTensor(np.random.rand(batch_size, num_input_features))

    # normal operation for use LSTM to decode the data
    rnn_old_h, rnn_old_c = rnn_old(input_tensor, states)
    # use the new LSTM to decode the data
    rnn_h, rnn_c = rnn(input_tensor, states)

    # check whether the two objects' outputs are the same
    print("whether the two objects' h_1 are the same: ", is_equal(rnn_old_h, rnn_h))
    print("whether the two objects' c_1 are the same: ", is_equal(rnn_old_c, rnn_c))

    # check whether the gradient backward can be done
    x = torch.ones(hidden_size)
    f = torch.matmul(rnn_h, x)
    f.backward(torch.ones(batch_size))
    print("the backward operation can be run normally")

def test_layernorm_LSTM(use_biLSTM=True):
    batch_size = 4
    max_length = 3
    hidden_size = 2
    n_layer = 5
    num_input_features = 3
    n_direction = 2 if use_biLSTM else 1
    # create two objects
    rnn = LayerNormLSTM(num_input_features, hidden_size, n_layer, bias=True, bidirectional=use_biLSTM, use_layer_norm=False)
    rnn_old = torch.nn.LSTM(num_input_features, hidden_size, n_layer, bias=True, bidirectional=use_biLSTM)
    # initialize two objects with same weights
    rnn.copy_parameters(rnn_old)
    # initialize the hidden state
    states = (torch.zeros(n_layer*n_direction, batch_size, hidden_size), torch.zeros(n_layer*n_direction, batch_size, hidden_size))
    # create the sequence data with padding
    input_tensor = torch.zeros(batch_size, max_length, num_input_features)
    input_tensor[0] = torch.FloatTensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    input_tensor[1] = torch.FloatTensor([[4, 5, 6], [5, 7, 8], [0, 0, 0]])
    input_tensor[2] = torch.FloatTensor([[6, 4, 3], [8, 1, 9], [0, 0, 0]])
    input_tensor[3] = torch.FloatTensor([[7, 3, 5], [0, 0, 0], [0, 0, 0]])
    seq_lengths = [3, 2, 2, 1]
    # transform the sequence data into new shape [max_length, batch_size, num_input_features]
    batch_in = Variable(input_tensor)
    batch_in = batch_in.permute(1, 0, 2)
    # normal operation for use LSTM to decode the sequence
    pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths)
    rnn_old_out, rnn_old_states = rnn_old(pack, states)
    rnn_old_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_old_out)
    # use the new LSTM to decode the sequence
    rnn_out, rnn_states = rnn(batch_in, states, seq_lengths)

    # check whether the two objects' outputs are the same
    print("whether the two objects' outputs are the same: ", is_equal(rnn_old_out, rnn_out))
    print("whether the two objects' h_n are the same: ", is_equal(rnn_old_states[0], rnn_states[0]))
    print("whether the two objects' c_n are the same: ", is_equal(rnn_old_states[1], rnn_states[1]))

    # check whether the gradient backward can be done
    x = torch.ones(hidden_size * n_direction)
    f = torch.matmul(rnn_out, x)
    f.backward(torch.ones(max_length, batch_size))
    print("the backward operation can be run normally")


if __name__ == "__main__":
    print("start checking the layernorm-LSTMCell......")
    test_layernorm_LSTMCell()
    print()
    # print("start checking the layernorm-LSTM......")
    # test_layernorm_LSTM(use_biLSTM=False)
    # print()
    # print("start checking the bi-layernorm-LSTM......")
    # test_layernorm_LSTM(use_biLSTM=True)
