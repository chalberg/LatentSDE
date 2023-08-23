import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsde
from utils import _stable_division

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define neural SDE
class NeuralSDE(nn.Module):
    def __init__(self, state_size, hidden_size, context_size, bm_size, batch_size):
        super(NeuralSDE, self).__init__()

        self.state_size = state_size
        self.bm_size = bm_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.context_size = context_size

        # TO DO: update to include context
        self.post_net = nn.Sequential(nn.Linear(state_size + 1 + context_size, hidden_size), # [state_size, t, context_size] --> [state_size]
                                       nn.Softmax(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.Softmax(),
                                       nn.Linear(hidden_size, state_size))
        
        self.sigma_net = nn.Sequential(nn.Linear(state_size + 1, hidden_size), # [state_size, t] --> [state_size * bm_size]
                                           nn.Softmax(),
                                           nn.Linear(hidden_size, hidden_size),
                                           nn.Softplus(),
                                           nn.Linear(hidden_size, state_size * bm_size)) 
        
        self.prior_net = nn.Sequential(nn.Linear(state_size+1, hidden_size), # [state_size, t] --> [state_size]
                                       nn.Softmax(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.Softmax(),
                                       nn.Linear(hidden_size, state_size))
        
        # Initialization trick from Glow.
        self.post_net[-1].weight.data.fill_(0.)
        self.post_net[-1].bias.data.fill_(0.)
        self.sigma_net[-1].weight.data.fill_(0.)
        self.sigma_net[-1].bias.data.fill_(0.)
        self.prior_net[-1].weight.data.fill_(0.)
        self.prior_net[-1].bias.data.fill_(0.)
        
    # appx posterior drift
    def mu_post(self, t, x):
        t = t.unsqueeze(0)
        x = x.squeeze(0)
        tx = torch.stack([t, x], dim=-1)
        return self.post_net(tx)
    
    # prior drift
    def mu_prior(self, t, x):
        t = t.unsqueeze(0)
        x = x.squeeze(0)
        tx = torch.stack([t, x], dim=-1)
        return self.prior_net(tx)

    # diffusion
    def sigma(self, t, x):
        t = t.unsqueeze(0)
        x = x.squeeze(0)
        tx = torch.stack([t, x], dim=-1)
        return self.sigma_net(tx).view(self.batch_size, self.state_size, self.bm_size)
    
    # augmented state drift (X, u)
    def mu_aug(self, t, x):
        x = x[:, 0:1] 
        f, g, h = self.mu_post(t, x), self.sigma(t, x), self.mu_prior(t, x)
        u = _stable_division(f - h, g)
        kl = .5 * (u ** 2).sum(dim=1, keepdim=True) # 1/2 ||u||_2^2 = KL(q||p)
        return torch.cat([f, kl], dim=1)
    
    # augmented state diffusion (X, u)
    def sigma_aug(self, t, x):
        x = x[:, 0:1]
        g = self.g(t, x)
        g_logqp = torch.zeros_like(x)
        return torch.cat([g, g_logqp], dim=1)
    
    def forward(self, ts, x):
        # Initial condition
        y0 = self.encoder(x[:3])  # shape (t_size)
        #Y = torch.zeros(len(ts), device=device) # shape (t_size)

        # generate Brownian motion sample
        bm = torchsde.BrownianInterval(t0=ts[3].item(),
                                       t1=ts[-1].item(),
                                       size=(self.state_size, self.bm_size),
                                       device=device)
            
        # numerically solve SDE
        y0 = y0.unsqueeze(0)
        sol = torchsde.sdeint_adjoint(sde=self,
                                       y0=y0,
                                       ts=ts[3:],
                                       bm=bm,
                                       method="reversible_heun",
                                       names={'drift': 'f_aug', 'diffusion': 'g_aug'})
        sol = sol.squeeze().view(-1)
        return sol # [ys, kl]


class RNNEncoder(nn.Module):
    def __init__(self, rnn_input_dim, hidden_dim, encoder_network=None):
        super(RNNEncoder, self).__init__()
        self.encoder_network = encoder_network
        self.rnn_cell = nn.GRUCell(rnn_input_dim, hidden_dim)

    def forward(self, h, x_current, y_prev, t_current, t_prev):
        t_current = torch.ones(x_current.shape[0], 1).to(t_current) * t_current
        t_prev = torch.ones_like(t_current) * t_prev

        if self.encoder_network is None:
            t_diff = t_current - t_prev
            input = torch.cat([x_current, y_prev, t_current, t_prev, t_diff], 1)
        else:
            input = self.encoder_network(x_current, y_prev, t_current, t_prev)
        return self.rnn_cell(input, h)