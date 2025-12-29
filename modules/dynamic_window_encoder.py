import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from modules.dynamics.variational_inference import Decoder, kl_distance
import numpy as np

class DynamicTimeWindow(nn.Module):
    def __init__(self, args, obs_dim, act_dim):
        super(DynamicTimeWindow, self).__init__()
        self.args = args
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_window = 15
        # Center index in the input chunk.
        # We expect input chunk to be [t-14, ..., t, ..., t+8]
        # Length = 15 + 8 = 23.
        # t is at index 14.
        self.center_idx = 14 

        # GRU for implicit history encoding
        # Input: obs + act
        self.gru_input_dim = obs_dim + act_dim
        self.gru_hidden_dim = 32
        self.history_gru = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, batch_first=True)

        # MLP for window length prediction
        # Input: 3 manual features + GRU hidden state
        self.mlp_input_dim = 3 + self.gru_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 14) # Output 14 classes (lengths 2 to 15)
        )
        
        # Layer normalization for features
        self.layer_norm = nn.LayerNorm(self.mlp_input_dim)

    def forward(self, obs_chunk, act_chunk, test_mode=False, gru_hidden=None):
        """
        obs_chunk: (B, T_chunk, obs_dim)
        act_chunk: (B, T_chunk, act_dim)
        T_chunk should be enough to cover t-14 to t.
        Assumes t is at self.center_idx.
        """
        B, T, _ = obs_chunk.shape
        
        # 1. Extract History Window [t-14, t] for Feature Calculation
        # Indices: 0 to 14 (inclusive)
        history_obs = obs_chunk[:, :self.center_idx+1, :] # (B, 15, obs_dim)
        history_act = act_chunk[:, :self.center_idx+1, :] # (B, 15, act_dim)
        
        # --- Manual Features (Explicit) ---
        # a. Entropy H(o_t)
        obs_t = history_obs[:, -1, :] # (B, obs_dim)
        p_obs = F.softmax(obs_t, dim=-1)
        entropy = -th.sum(p_obs * th.log(p_obs + 1e-8), dim=-1, keepdim=True) # (B, 1)

        # b. Short-term rate of change (using last 4 steps: t, t-1, t-2, t-3)
        diffs = []
        for k in range(3):
            curr = history_obs[:, -1 - k, :]
            prev = history_obs[:, -1 - k - 1, :]
            diff = th.norm(curr - prev, p=2, dim=-1)
            diffs.append(diff)
        rate_of_change = th.stack(diffs, dim=1).mean(dim=1, keepdim=True) # (B, 1)

        # c. Correlation Cor(a_{t-1}, o_t)
        act_prev = history_act[:, -2, :] # (B, act_dim)
        
        curr_obs_dim = obs_t.shape[-1]
        curr_act_dim = act_prev.shape[-1]
        max_dim = max(curr_obs_dim, curr_act_dim)
        
        obs_pad = F.pad(obs_t, (0, max_dim - curr_obs_dim))
        act_pad = F.pad(act_prev, (0, max_dim - curr_act_dim))
        
        obs_centered = obs_pad - obs_pad.mean(dim=1, keepdim=True)
        act_centered = act_pad - act_pad.mean(dim=1, keepdim=True)
        
        denom = th.norm(obs_centered, p=2, dim=1, keepdim=True) * th.norm(act_centered, p=2, dim=1, keepdim=True) + 1e-8
        correlation = (obs_centered * act_centered).sum(dim=1, keepdim=True) / denom

        # --- GRU Encoding (Implicit) ---
        # We run GRU over the history window [t-14, t]
        # If gru_hidden is provided (inference), we might just run one step?
        # But here we are given a chunk.
        # If test_mode=True and gru_hidden is not None, we assume we are stepping through time.
        # But the input is a chunk.
        # To support both training (random chunks) and inference (sequential),
        # we will run the GRU over the provided history chunk.
        # For inference, the caller should provide the history chunk or we maintain state.
        # Given the current architecture, re-running over the short history window (15 steps) is cheap and robust.
        
        gru_input = th.cat([history_obs, history_act], dim=-1) # (B, 15, D)
        _, h_n = self.history_gru(gru_input) # h_n: (1, B, H)
        gru_feature = h_n.squeeze(0) # (B, H)

        # 2. Combine and Normalize
        features = th.cat([entropy, rate_of_change, correlation, gru_feature], dim=1) # (B, 3+H)
        features_norm = self.layer_norm(features)

        # 3. MLP Window Prediction
        logits = self.mlp(features_norm) # (B, 14)
        probs = F.softmax(logits, dim=-1)
        
        if test_mode:
            window_idx = th.argmax(probs, dim=1)
        else:
            dist = Categorical(probs)
            window_idx = dist.sample()
            
        window_len = window_idx + 2 # (B,)

        # 4. Create Mask and Padded Window for BiGRU
        # The BiGRU expects a window centered at t.
        # We need to extract [t - (W-1)//2, t + W//2] from the input chunk.
        # Input chunk center is self.center_idx (14).
        
        mask = th.zeros((B, self.max_window), device=obs_chunk.device)
        indices = th.arange(self.max_window, device=obs_chunk.device).unsqueeze(0).expand(B, -1) # (B, 15)
        
        # BiGRU Window Center in the output padded window is 7.
        # We map from input chunk to output padded window.
        
        start_offset = (window_len - 1) // 2 # How far back from t
        end_offset = window_len // 2 # How far forward from t
        
        # We want to extract: chunk[center_idx - start_offset : center_idx + end_offset + 1]
        # And place it into padded_window centered at 7.
        
        # Since we can't easily do vectorized slicing with variable lengths into a fixed tensor in one go without loop,
        # we will use a loop similar to BiGRUEncoder.
        
        padded_window = th.zeros((B, self.max_window, self.obs_dim + self.act_dim), device=obs_chunk.device)
        
        for i in range(B):
            s_off = start_offset[i].item()
            e_off = end_offset[i].item()
            
            # Source indices
            src_start = self.center_idx - s_off
            src_end = self.center_idx + e_off + 1
            
            # Target indices (centered at 7)
            tgt_start = 7 - s_off
            tgt_end = 7 + e_off + 1
            
            # Extract and place
            # Note: src_end might exceed chunk length if we are at the end of episode and padding wasn't enough?
            # But we assume chunk is padded.
            
            chunk_slice = th.cat([obs_chunk[i, src_start:src_end], act_chunk[i, src_start:src_end]], dim=-1)
            padded_window[i, tgt_start:tgt_end] = chunk_slice
            mask[i, tgt_start:tgt_end] = 1.0
            
        return window_len, padded_window, mask

class BiGRUEncoder(nn.Module):
    def __init__(self, args, input_dim, hidden_dim):
        super(BiGRUEncoder, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        
        # Forward GRU
        self.gru_f = nn.GRU(input_dim, hidden_dim, batch_first=True)
        # Backward GRU
        self.gru_b = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # MLP for fusion
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim) # Output z_i dimension (embedding shape)
        )

    def forward(self, padded_window, window_len, mask_backward_prob=0.0):
        """
        padded_window: (B, 15, input_dim)
        window_len: (B,)
        mask_backward_prob: float, probability to mask out backward window (Curriculum Learning)
        """
        B, T, D = padded_window.shape
        center_idx = 7
        
        len_f = (window_len - 1) // 2 + 1
        len_b = window_len // 2 + 1
        
        batch_f = []
        batch_b = []
        
        # Determine which samples in batch to mask backward
        # If mask_backward_prob > 0, we randomly select samples to mask.
        # Or we can mask the whole batch? Usually per-sample is better variance.
        # Let's do per-sample.
        
        use_backward = (th.rand(B, device=padded_window.device) > mask_backward_prob)
        
        for i in range(B):
            # Forward
            l_f = len_f[i].item()
            seq_f = padded_window[i, center_idx - l_f + 1 : center_idx + 1, :]
            batch_f.append(seq_f)
            
            # Backward
            if use_backward[i]:
                l_b = len_b[i].item()
                seq_b = padded_window[i, center_idx : center_idx + l_b, :]
                seq_b = th.flip(seq_b, [0])
            else:
                # If masked, we provide a dummy sequence of length 1 (zeros)
                # Or just zeros of length 1.
                # The GRU needs at least length 1.
                seq_b = th.zeros((1, D), device=padded_window.device)
                len_b[i] = 1 # Update length to 1
                
            batch_b.append(seq_b)
            
        # Pad sequences
        padded_f = nn.utils.rnn.pad_sequence(batch_f, batch_first=True) 
        padded_b = nn.utils.rnn.pad_sequence(batch_b, batch_first=True) 
        
        # Pack
        packed_f = nn.utils.rnn.pack_padded_sequence(padded_f, len_f.cpu(), batch_first=True, enforce_sorted=False)
        packed_b = nn.utils.rnn.pack_padded_sequence(padded_b, len_b.cpu(), batch_first=True, enforce_sorted=False)
        
        # Run GRU
        _, h_f = self.gru_f(packed_f) # h_f: (1, B, H)
        _, h_b = self.gru_b(packed_b) # h_b: (1, B, H)
        
        h_f = h_f.squeeze(0)
        h_b = h_b.squeeze(0)
        
        # If backward was masked, h_b is result of zero input.
        # We might want to zero it out explicitly if we want "no information".
        # But GRU(0) is not necessarily 0.
        # Let's explicitly zero out h_b for masked samples to be cleaner.
        if mask_backward_prob > 0:
             h_b = h_b * use_backward.unsqueeze(1).float()

        # Fusion
        combined = th.cat([h_f, h_b], dim=1)
        z = self.mlp(combined)
        
        return z

class SMPE2_Encoder(nn.Module):
    def __init__(self, args, obs_dim, act_dim, embedding_shape):
        super(SMPE2_Encoder, self).__init__()
        self.args = args
        self.dynamic_window = DynamicTimeWindow(args, obs_dim, act_dim)
        
        input_dim = obs_dim + act_dim
        self.bi_gru = BiGRUEncoder(args, input_dim, embedding_shape)
        
        # VAE heads
        self.mu = nn.Linear(embedding_shape, embedding_shape)
        self.logvar = nn.Linear(embedding_shape, embedding_shape)
        
        self.N = th.distributions.Normal(0, 1)
        if args.use_cuda:
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()
        
        self.kl = 0

    def forward(self, obs_chunk, act_chunk, test_mode=False, mask_backward_prob=0.0):
        # 1. Dynamic Window
        window_len, padded_window, mask = self.dynamic_window(obs_chunk, act_chunk, test_mode)
        
        # 2. BiGRU Encoding
        z_encoded = self.bi_gru(padded_window, window_len, mask_backward_prob)
        
        # 3. VAE
        mu = self.mu(z_encoded)
        sigma = th.exp(0.5 * self.logvar(z_encoded))
        
        if test_mode:
            z = mu
        else:
            z = mu + sigma * self.N.sample(mu.shape)
            
        self.kl = kl_distance(mu, sigma, th.zeros_like(mu), th.ones_like(sigma))
        return z, mu, sigma

class SMPE2_VAE(nn.Module):
    def __init__(self, obs_dim, act_dim, embedding_shape, output_dim, args):
        super(SMPE2_VAE, self).__init__()
        self.encoder = SMPE2_Encoder(args, obs_dim, act_dim, embedding_shape)
        self.decoder = Decoder(embedding_shape, output_dim, args)

    def forward(self, obs_chunk, act_chunk, test_mode=False, mask_backward_prob=0.0):
        z, mu, sigma = self.encoder(obs_chunk, act_chunk, test_mode, mask_backward_prob)
        return self.decoder(z), z, mu, sigma
