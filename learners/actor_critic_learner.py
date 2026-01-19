import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from modules.critics.centralV import CentralVCritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import Adam
from modules.critics import REGISTRY as critic_resigtry, CentralQCritic, CentralVCritic
from model_learners import REGISTRY as mle_REGISTRY
from components.standarize_stream import RunningMeanStd

class ActorCriticVLearner:
    def __init__(self, mac, scheme, logger, args, mle_learner=None):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.critic_w = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic_w = copy.deepcopy(self.critic_w)

        self.critic_params_w = list(self.critic_w.parameters())
        self.critic_optimiser_w = Adam(params=self.critic_params_w, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        # Dynamics model learner
        self.mle_learner = mle_learner

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            if self.args.use_intrinsic:
                self.rew_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
            else:
                self.rew_ms = RunningMeanStd(shape=(1,), device=device)
        return

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, new_rewards=None):

        # Get the relevant quantities
        if new_rewards is None:
            rewards = batch["reward"][:, :-1]       # shape: (batch_size, maxseqlen-1, num_agents)
        else:
            rewards = new_rewards[:, :-1]

        # rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]        # shape: (batch_size, maxseqlen, num_agents, 1)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()  # shape: (batch_size, maxseqlen, 1)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])        

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            # Use a slightly larger epsilon (1e-4) or a stabilized version to prevent 
            # excessive amplification of intrinsic rewards when environmental rewards are 0.
            rewards = (rewards - self.rew_ms.mean) / (th.sqrt(self.rew_ms.var) + 1e-4)       # shape: (batch_size, maxseqlen-1, 1)
        
        # No experiences to train on in this minibatch
        if mask.sum() == 0:
            self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
            self.logger.console_logger.error("Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env))
            return

        # Mask for critic and actor
        mask = mask.repeat(1, 1, self.n_agents)     # share episode termination mask for all agents in the same env
        critic_mask = mask.clone()      # shape: (batch_size, maxseqlen, num_agents)

        # Initialize the hidden states for the first timestep
        self.mac.init_hidden(batch.batch_size)

        # Compute forward passes for each agent for all parallel envs
        mac_out = []
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        pi = th.stack(mac_out, dim=1)      # shape: (batch_size, maxseqlen, num_agents, num_actions)

        advantages, critic_train_stats = self.train_critic_sequential(self.critic, self.target_critic, batch, rewards,
                                                                      critic_mask, t_env)
        # advantages:   shape = (batch_size, maxseqlen-1, num_agents)

        # Detach advantages from the computational graph
        advantages = advantages.detach()        
        
        # Calculate policy grad with mask
        actions = actions[:, :-1]
        pi[mask == 0] = 1.0     
        pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)       # filter on π(u_i|τ_i, θ_i)
        log_pi_taken = th.log(pi_taken + 1e-10)        # shape: (batch_size, maxseqlen, num_agents)

        # Add entropy to loss
        entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)      # shape: (batch_size, maxseqlen, num_agents)
        
        # Policy Gradient Loss
        pg_loss = -((advantages * log_pi_taken + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        # Target Update
        self.critic_training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "v_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key])/ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env
        return pi_taken

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask, t_env):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        
        #print(batch["obs"].shape) #10,26,3,18
        #print(batch["state"].unsqueeze(2).repeat(1, 1, self.n_agents, 1).shape) #10,26,3,54
        if self.args.use_w and self.args.use_w_critic and t_env % self.args.update_filter_critic == 0:
            weights = []
            for agent_id in range(self.n_agents):
                obs = batch["obs"][:,:,agent_id]
                mu_obs = self.mac.vae_controller.obs_ms.mean
                std_obs = th.sqrt(self.mac.vae_controller.obs_ms.var) + 1e-8
                obs = (obs - mu_obs) / std_obs
                weight = self.mac.vae_controller.filters[agent_id].forward(obs)
                
                if getattr(self.args, "env", "gymma") == "sc2":
                    # For SC2, weight already has state_dim 
                    # and the state is not a simple concatenation of obs
                    weights.append(weight.unsqueeze(2))
                else:
                    cut = obs.shape[-1]
                    padding_mat = th.ones_like(obs)
                    weight = th.cat((weight[:, :, 0:agent_id*cut], padding_mat, weight[:, :, (agent_id*cut):]), dim=-1)
                    weights.append(weight.unsqueeze(2))
            weights = th.cat(weights, dim=2)

        inputs = []
        inputs_w = []
        if self.args.use_w and self.args.use_w_critic and t_env % self.args.update_filter_critic == 0:
            input_batch_w = weights * batch["state"].unsqueeze(2).repeat(1, 1, self.n_agents, 1)
            inputs_w.append(input_batch_w)  
            inputs_w.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
            inputs_w = th.cat(inputs_w, dim=-1)

        input_batch = batch["state"].unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        inputs.append(input_batch)  
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat(inputs, dim=-1)

        with th.no_grad():
            target_vals = target_critic(inputs)
            target_vals = target_vals.squeeze(3)

        if self.args.use_w and self.args.use_w_critic and t_env % self.args.update_filter_critic == 0:
            with th.no_grad():
                target_vals_w = self.target_critic_w(inputs_w)
                target_vals_w = target_vals_w.squeeze(3)

        if self.args.standardise_returns:
            target_vals = target_vals * (th.sqrt(self.ret_ms.var) + 1e-8) + self.ret_ms.mean
            target_vals_w = target_vals_w * (th.sqrt(self.ret_ms.var) + 1e-8) + self.ret_ms.mean

        # Compute n-step target returns
        target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)
        if self.args.use_w and self.args.use_w_critic and t_env % self.args.update_filter_critic == 0:
            target_returns_w = self.nstep_returns(rewards, mask, target_vals_w, self.args.q_nstep)

        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / (th.sqrt(self.ret_ms.var) + 1e-8)
            if self.args.use_w and self.args.use_w_critic and t_env % self.args.update_filter_critic == 0:
                target_returns_w = (target_returns_w - self.ret_ms.mean) / (th.sqrt(self.ret_ms.var) + 1e-8)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "v_taken_mean": [],
        }

        # Forward pass
        v = critic(inputs)[:, :-1].squeeze(3)
        td_error = (target_returns.detach() - v)
        masked_td_error = td_error * mask

        if self.args.use_w and self.args.use_w_critic and t_env % self.args.update_filter_critic == 0:
            v_w = self.critic_w(inputs_w)[:, :-1].squeeze(3)
            td_error_w = (target_returns_w.detach() - v_w)
            masked_td_error_w = td_error_w * mask

        # Critic Loss: TD-Error
        loss = (masked_td_error ** 2).sum() / mask.sum()
        if self.args.use_w and self.args.use_w_critic and t_env % self.args.update_filter_critic == 0:
            loss_w = (masked_td_error_w ** 2).sum() / mask.sum()

        if self.args.use_w and self.args.use_w_critic and t_env % self.args.update_filter_critic == 0:
            self.critic_optimiser_w.zero_grad()
            for agent_id in range(self.n_agents):
                self.mac.vae_controller.filter_optimizers[agent_id].zero_grad()

        # Optimize critic
        self.critic_optimiser.zero_grad()
        loss.backward()
        if self.args.use_w and self.args.use_w_critic and t_env % self.args.update_filter_critic == 0:
            loss_w.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        if self.args.use_w and self.args.use_w_critic and t_env % self.args.update_filter_critic == 0:
            self.critic_optimiser_w.step()
            for agent_id in range(self.n_agents):
                self.mac.vae_controller.filter_optimizers[agent_id].lr = self.args.lr_w_critic
                self.mac.vae_controller.filter_optimizers[agent_id].step()
                self.mac.vae_controller.filter_optimizers[agent_id].lr = self.args.lr_filter
        
        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        running_log["v_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((target_returns * mask).sum().item() / mask_elems)
        return masked_td_error, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])    # shape: (batch_size, maxseqlen-1, num_agents)
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])    # shape: (batch_size, num_agents)
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** step * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** step * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t+1]
                else:
                    nstep_return_t += self.args.gamma ** step * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        return

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        return

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        return
    
    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()
        self.critic_w.cuda()
        self.target_critic_w.cuda()
        return
    
    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        return

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
        return