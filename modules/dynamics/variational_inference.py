import torch as th
import torch.nn as nn
import torch.nn.functional as F


class VariationalEncoder(nn.Module):
    def __init__(self, input_shape, embedding_shape, args):
        super(VariationalEncoder, self).__init__()
        self.input_shape = input_shape
        self.args = args
        self.embedding_shape = embedding_shape

        self.fc1 = nn.Linear(self.input_shape, self.embedding_shape)
        self.fc2 = nn.Linear(self.embedding_shape, self.embedding_shape)
        self.mu = nn.Linear(self.embedding_shape, self.embedding_shape)
        self.logvar = nn.Linear(self.embedding_shape, self.embedding_shape)

        self.N = th.distributions.Normal(0, 1)
        if args.use_cuda:
            self.N.loc = self.N.loc.cuda()          # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        else:
            self.N.loc = self.N.loc          # hack to get sampling on the GPU
            self.N.scale = self.N.scale
        self.kl = 0
        return

    def forward(self, x, test_mode=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu =  self.mu(x)
        l_var = self.logvar(x)
        l_var = th.clamp(l_var, min=-20, max=10) # Clamp logvar for stability
        sigma = th.exp(0.5 * l_var)
        if test_mode:
            z = mu
        else:
            z = mu + sigma * self.N.sample(mu.shape)  # reparameterisation trick
        self.kl = kl_distance(mu, sigma, th.zeros_like(mu), th.ones_like(sigma))
        return z, mu, sigma


class Decoder(nn.Module):
    def __init__(self, embedding_shape, output_shape, args, n_agents_others=None):
        super(Decoder, self).__init__()
        self.args = args
        self.embedding_shape = embedding_shape
        self.output_shape = output_shape
        if n_agents_others is None:
            n_agents_others = self.args.n_agents - 1

        self.fc1 = nn.Linear(self.embedding_shape, self.embedding_shape)
        self.fc2 = nn.Linear(self.embedding_shape, self.embedding_shape)
        self.fc3 = nn.Linear(self.embedding_shape, self.output_shape)

        if args.use_actions:
            self.fc4 = nn.Linear(self.embedding_shape, self.args.n_actions * n_agents_others)
        return

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

    def forward_actions(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        out = self.fc4(x)
        return out



class VAE(nn.Module):
    def __init__(self, input_shape, embedding_shape, output_dim, args, n_agents_others=None):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(input_shape, embedding_shape, args)
        self.decoder = Decoder(embedding_shape, output_dim, args, n_agents_others=n_agents_others)
        return

    def forward(self, x, test_mode=False):
        z, mu, sigma = self.encoder(x, test_mode)
        return self.decoder(z), z, mu, sigma


class VariationalEncoder_RNN(nn.Module):
    def __init__(self, input_shape, embedding_shape, args):
        super(VariationalEncoder_RNN, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.embedding_shape = embedding_shape

        self.rnn = nn.GRUCell(self.input_shape, self.embedding_shape)
        self.h = nn.Linear(self.embedding_shape, self.embedding_shape)
        self.mu = nn.Linear(self.embedding_shape, self.embedding_shape)
        self.logvar = nn.Linear(self.embedding_shape, self.embedding_shape)

        self.N = th.distributions.Normal(0, 1)
        if args.use_cuda:
            self.N.loc = self.N.loc.cuda()          # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        else:
            self.N.loc = self.N.loc          # hack to get sampling on the GPU
            self.N.scale = self.N.scale
        self.kl = 0
        return

    def forward(self, x, hidden_state, test_mode=False):
        hidden_state = self.rnn(x, hidden_state)
        h = F.relu(self.h(hidden_state))

        mu = self.mu(h)
        l_var = self.logvar(h)
        l_var = th.clamp(l_var, min=-20, max=10)
        sigma = th.exp(0.5 * l_var)
        if test_mode:
            z = mu
        else:
            z = mu + sigma * self.N.sample(mu.shape)  # reparameterisation trick
        self.kl = kl_distance(mu, sigma, th.zeros_like(mu), th.ones_like(sigma))
        return z, hidden_state, mu, sigma


class Variational_Encoder_Decoder_RNN(nn.Module):
    def __init__(self, input_shape, embedding_shape, output_shape, args):
        super(Variational_Encoder_Decoder_RNN, self).__init__()
        self.encoder = VariationalEncoder_RNN(input_shape, embedding_shape, args)
        self.decoder = Decoder(embedding_shape, output_shape, args)
        return

    def forward(self, x, hidden_state):
        z, hidden_state, mu, sigma = self.encoder(x, hidden_state)
        return self.decoder(z), hidden_state, z, mu, sigma


class Variational_Encoder_Decoder(nn.Module):
    def __init__(self, input_shape, embedding_shape, output_shape, args):
        super(Variational_Encoder_Decoder, self).__init__()
        self.encoder = VariationalEncoder(input_shape, embedding_shape, args)
        self.decoder = Decoder(embedding_shape, output_shape, args)
        return

    def forward(self, x):
        z, mu, sigma = self.encoder(x)
        return self.decoder(z), z, mu, sigma


def kl_distance(mu1, sigma1, mu2, sigma2):
    # Fully Factorized Gaussians
    numerator = (mu1 - mu2)**2 + (sigma1)**2
    denominator = 2 * (sigma2)**2 + 1e-8
    return th.sum(numerator / denominator + th.log(sigma2) - th.log(sigma1) - 1/2)


class Aux(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(Aux, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.fc1 = nn.Linear(self.input_shape, self.output_shape)
        return

    def forward(self, inputs):
        out = self.fc1(inputs)
        return out
    

class Filter(nn.Module):
    def __init__(self, input_shape, embedding_shape, args):
        super(Filter, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.embedding_shape = embedding_shape
        if self.args.use_gumbel:

            self.fc1 = nn.Linear(self.input_shape, 2*self.embedding_shape)

        else:
            self.fc1 = nn.Linear(self.input_shape, self.embedding_shape)
        self.fc2 = nn.Linear(self.embedding_shape, self.embedding_shape)
        return

    def forward(self, x):
        if self.args.use_gumbel:
          
            x = F.gumbel_softmax(self.fc1(x).view(-1, self.embedding_shape, 2), hard=True)
            x = th.argmax(x, dim=-1)
        else:
            if self.args.use_2layer_filter:
                x = F.relu(self.fc1(x))
                x = F.sigmoid(self.fc2(x))
            else:
                x = F.sigmoid(self.fc1(x))
            if self.args.use_clip_weights: 
                x = th.clamp(x, min=self.args.clip_min, max=self.args.clip_max)
        return x
