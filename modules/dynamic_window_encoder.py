import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from modules.dynamics.variational_inference import Decoder, kl_distance

class DynamicTimeWindow(nn.Module):
    def __init__(self, args, obs_dim, act_dim):
        super(DynamicTimeWindow, self).__init__()
        self.args = args
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_window = 15
        self.center_idx = 7  # 0..14, center is 7. (15 steps)

        # MLP for window length prediction
        # Input: 3 features
        self.mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 14) # Output 14 classes (lengths 2 to 15)
        )
        
        # Layer normalization for features
        self.layer_norm = nn.LayerNorm(3)

    def forward(self, obs_chunk, act_chunk, test_mode=False):
        """
        obs_chunk: (B, 15, obs_dim)
        act_chunk: (B, 15, act_dim)
        Assumes chunks are centered at t (index 7).
        """
        B, T, _ = obs_chunk.shape
        
        # Debug prints
        # print(f"DEBUG: obs_dim={self.obs_dim}, act_dim={self.act_dim}")
        # print(f"DEBUG: obs_chunk.shape={obs_chunk.shape}, act_chunk.shape={act_chunk.shape}")

        # 1. Feature Calculation
        # a. Entropy H(o_t)
        # Normalize obs to prob distribution (softmax) for entropy calc?
        # Or just use values if they are suitable. Assuming softmax for safety as per "Entropy" name.
        obs_t = obs_chunk[:, self.center_idx, :] # (B, obs_dim)
        p_obs = F.softmax(obs_t, dim=-1)
        entropy = -th.sum(p_obs * th.log(p_obs + 1e-8), dim=-1, keepdim=True) # (B, 1)

        # b. Short-term rate of change
        # K=3. t, t-1, t-2, t-3.
        # Indices: 7, 6, 5, 4.
        # diffs: ||o_t - o_{t-1}||, ||o_{t-1} - o_{t-2}||, ||o_{t-2} - o_{t-3}||
        diffs = []
        for k in range(3):
            curr = obs_chunk[:, self.center_idx - k, :]
            prev = obs_chunk[:, self.center_idx - k - 1, :]
            diff = th.norm(curr - prev, p=2, dim=-1)
            diffs.append(diff)
        rate_of_change = th.stack(diffs, dim=1).mean(dim=1, keepdim=True) # (B, 1)

        # c. Correlation Cor(a_{t-1}, o_t)
        # a_{t-1} is at index 6 (if act aligned such that act[t] is at t).
        # Wait, usually act[t] causes obs[t+1].
        # In EpisodeBatch, actions[:, t] is action at t.
        # So action at t-1 is at index t-1.
        # If center is 7 (time t), then t-1 is 6.
        act_prev = act_chunk[:, self.center_idx - 1, :] # (B, act_dim)
        
        # Pearson correlation between two vectors
        # (x - mean(x)) . (y - mean(y)) / (std(x)*std(y))
        # We compute this per sample in batch.

        # obs_dim and act_dim might differ.
        # The formula Cor(a, o) implies some correlation measure.
        # If dims differ, maybe we project them? Or just flatten?
        # The image shows Cor(a_{t-1}, o_t).
        # If dimensions differ, standard Pearson is not directly applicable element-wise.
        # Maybe it means Canonical Correlation? Or just Cosine Similarity?
        # "Correlation" often implies Cosine Similarity of centered vectors.
        # If sizes differ, we can't do element-wise dot product unless we pad or truncate.
        # I'll assume we use the min dimension or they are same.
        # Or maybe we just use Cosine Similarity.
        # Let's use Cosine Similarity for now, as it's robust.
        # But we need to handle dimension mismatch.
        # I'll pad the smaller one with zeros.
        curr_obs_dim = obs_t.shape[-1]
        curr_act_dim = act_prev.shape[-1]
        max_dim = max(curr_obs_dim, curr_act_dim)
        
        obs_pad = F.pad(obs_t, (0, max_dim - curr_obs_dim))
        act_pad = F.pad(act_prev, (0, max_dim - curr_act_dim))
        
        obs_centered = obs_pad - obs_pad.mean(dim=1, keepdim=True)
        act_centered = act_pad - act_pad.mean(dim=1, keepdim=True)
        
        denom = th.norm(obs_centered, p=2, dim=1, keepdim=True) * th.norm(act_centered, p=2, dim=1, keepdim=True) + 1e-8
        correlation = (obs_centered * act_centered).sum(dim=1, keepdim=True) / denom

        # 2. Normalize features
        features = th.cat([entropy, rate_of_change, correlation], dim=1) # (B, 3)
        features_norm = self.layer_norm(features)

        # 3. MLP Window Prediction
        logits = self.mlp(features_norm) # (B, 14)
        probs = F.softmax(logits, dim=-1)
        
        # Output: window_len (2 to 15)
        # Classes 0..13 correspond to lengths 2..15
        if test_mode:
            window_idx = th.argmax(probs, dim=1)
        else:
            dist = Categorical(probs)
            window_idx = dist.sample()
            
        window_len = window_idx + 2 # (B,)

        # 4. Create Mask and Padded Window
        # window_len is tensor of shape (B,)
        # We need to mask obs_chunk and act_chunk
        # Center is 7.
        # Range: [7 - (W-1)//2, 7 + W//2 + 1)
        
        mask = th.zeros((B, self.max_window), device=obs_chunk.device)
        
        # Vectorized mask creation is tricky with variable lengths.
        # We can use a loop or broadcasting.
        indices = th.arange(self.max_window, device=obs_chunk.device).unsqueeze(0).expand(B, -1) # (B, 15)
        
        start = self.center_idx - (window_len - 1) // 2
        end = self.center_idx + window_len // 2 + 1
        
        # start and end are (B,)
        start = start.unsqueeze(1)
        end = end.unsqueeze(1)
        
        mask = (indices >= start) & (indices < end)
        mask = mask.float()
        
        # Concatenate obs and act
        window_obs_act = th.cat([obs_chunk, act_chunk], dim=-1) # (B, 15, obs+act)
        padded_window = window_obs_act * mask.unsqueeze(-1)
        
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

    def forward(self, padded_window, window_len):
        """
        padded_window: (B, 15, input_dim)
        window_len: (B,)
        """
        B, T, D = padded_window.shape
        center_idx = 7
        
        # Prepare sequences for GRU
        # Forward: X(t-k)...X(t)
        # Backward: X(t+k)...X(t)
        
        # We need to extract these sequences.
        # Forward length: (W-1)//2 + 1
        # Backward length: W//2 + 1
        
        len_f = (window_len - 1) // 2 + 1
        len_b = window_len // 2 + 1
        
        # Max lengths
        max_len_f = (15 - 1) // 2 + 1 # 8
        max_len_b = 15 // 2 + 1 # 8
        
        # Extract and pad forward sequences
        # Forward seq ends at center_idx. Starts at center_idx - len_f + 1
        # We want input order: t-k, ..., t.
        # So we take slice [center_idx - len_f + 1 : center_idx + 1]
        
        # Extract and pad backward sequences
        # Backward seq starts at center_idx. Ends at center_idx + len_b
        # We want input order: t+k, ..., t.
        # So we take slice [center_idx : center_idx + len_b] and REVERSE it?
        # Image says: X(t+2), X(t+1), X(t). Yes, reverse time.
        
        # Since we can't easily slice with variable indices in a tensor, 
        # we might have to loop or use advanced indexing.
        # Given batch size is small (e.g. 32), loop is fine.
        
        batch_f = []
        batch_b = []
        
        for i in range(B):
            # Forward
            l_f = len_f[i].item()
            # Slice: from (7 - l_f + 1) to (7 + 1)
            seq_f = padded_window[i, center_idx - l_f + 1 : center_idx + 1, :]
            batch_f.append(seq_f)
            
            # Backward
            l_b = len_b[i].item()
            # Slice: from 7 to (7 + l_b)
            seq_b = padded_window[i, center_idx : center_idx + l_b, :]
            # Reverse
            seq_b = th.flip(seq_b, [0])
            batch_b.append(seq_b)
            
        # Pad sequences
        padded_f = nn.utils.rnn.pad_sequence(batch_f, batch_first=True) # (B, max_len_f, D)
        padded_b = nn.utils.rnn.pad_sequence(batch_b, batch_first=True) # (B, max_len_b, D)
        
        # Pack
        packed_f = nn.utils.rnn.pack_padded_sequence(padded_f, len_f.cpu(), batch_first=True, enforce_sorted=False)
        packed_b = nn.utils.rnn.pack_padded_sequence(padded_b, len_b.cpu(), batch_first=True, enforce_sorted=False)
        
        # Run GRU
        _, h_f = self.gru_f(packed_f) # h_f: (1, B, H)
        _, h_b = self.gru_b(packed_b) # h_b: (1, B, H)
        
        h_f = h_f.squeeze(0)
        h_b = h_b.squeeze(0)
        
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

    def forward(self, obs_chunk, act_chunk, test_mode=False):
        # 1. Dynamic Window
        window_len, padded_window, mask = self.dynamic_window(obs_chunk, act_chunk, test_mode)
        
        # 2. BiGRU Encoding
        z_encoded = self.bi_gru(padded_window, window_len)
        
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

    def forward(self, obs_chunk, act_chunk, test_mode=False):
        z, mu, sigma = self.encoder(obs_chunk, act_chunk, test_mode)
        return self.decoder(z), z, mu, sigma
