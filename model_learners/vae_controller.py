from modules.dynamics import REGISTRY as mle_model_REGISTRY
from modules.dynamics import VAE, kl_distance, Aux, Filter
from modules.dynamic_window_encoder import SMPE2_VAE
from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from components.simhash import HashCount

from torch.optim import Adam, RMSprop
from torch.distributions import MultivariateNormal
import torch as th
import torch.nn as nn
import numpy as np
from operator import itemgetter 
import gc
import time
import os


class VAEController:
    def __init__(self, scheme, args):
        self.n_agents = args.n_agents
        self.scheme = scheme
        self.args = args
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.last_save_t = 0

        # Input shapes
        self.state_dim = args.state_dim
        self.n_actions = self.args.n_actions
        self.obs_dim = scheme["obs"]["vshape"]
        self.agent_input_shape = self.obs_dim   
        self.full_input_shape =  self.state_dim
        self.state_embedding_shape = self.args.latent_dim           # mean and sigma dimension
        if self.args.use_actions:
            self.actions_dim = self.n_actions
            self.actions_criterion = nn.CrossEntropyLoss(reduction='sum')
        else:
            self.actions_dim = 0
        if self.args.use_rewards:
            self.rewards_dim = 1
        else:
            self.rewards_dim = 0
       
        # Replay buffer (dataset)
        self.dataset_is_full = False
        self.dataset_size = 50
        self.dataset_count = 0
        self.dataset = [ 0 for _ in range(self.dataset_size) ]
        self.obs_ms = RunningMeanStd(shape=(self.obs_dim,), device=self.args.device)
        self.state_ms = RunningMeanStd(shape=(self.state_dim,), device=self.args.device)
        self.rew_ms = RunningMeanStd(shape=(1,), device=self.args.device)

        # Build Hashers
        self.build_hashers()

        # Build agent models
        self.build_agent_models()

        if self.args.use_aux:
            # Build Auxiliary models
            self.build_agent_auxiliary_models()

        if self.args.use_w:
            self.build_filters()
            self.build_filters_targets()
        
        return
        

    def build_hashers(self):
        self.hash_obs = [HashCount(self.obs_dim) for _ in range(self.n_agents)]
        self.hash_z = [HashCount(self.state_embedding_shape) for _ in range(self.n_agents)]
        return


    def build_filters(self):
        self.filters = nn.ModuleList([Filter(self.obs_dim + self.actions_dim + self.rewards_dim, self.state_dim - self.obs_dim, self.args) \
                                           .to(self.args.device)
                                           for _ in range(self.n_agents)])
        self.filter_params = [ list(model.parameters()) for model in self.filters ]
        self.filter_optimizers = [ RMSprop(params=param, lr=self.args.lr_filter) for param in self.filter_params ]
        return
    

    def build_filters_targets(self):
        self.target_filters = nn.ModuleList([Filter(self.obs_dim + self.actions_dim + self.rewards_dim, self.state_dim - self.obs_dim, self.args) \
                                           .to(self.args.device)
                                           for _ in range(self.n_agents)])
        self.update_filters_targets()


    def update_filters_targets(self):
        for agent_id in range(self.n_agents):
            self.target_filters[agent_id].load_state_dict(self.filters[agent_id].state_dict())


    def build_agent_models(self):
        # Force use_dynamic_window for this task
        self.args.use_dynamic_window = True
        
        if self.args.use_actions:
            # Agents' models (with partial observability)
            if getattr(self.args, "use_dynamic_window", False):
                self.agent_models = nn.ModuleList([SMPE2_VAE(self.obs_dim, self.n_actions, self.state_embedding_shape, self.state_dim - self.obs_dim, self.args) \
                                                .to(self.args.device)
                                                for _ in range(self.n_agents)])
            else:
                self.agent_models = nn.ModuleList([VAE(self.obs_dim, self.state_embedding_shape, self.state_dim - self.obs_dim, self.args) \
                                                .to(self.args.device)
                                                for _ in range(self.n_agents)])
            self.agent_params = [ list(model.parameters()) for model in self.agent_models ]
            self.agent_optimizers = [ RMSprop(params=param, lr=self.args.lr_agent_model) for param in self.agent_params ]
        
        else:
            # Agents' models (with partial observability)
            if getattr(self.args, "use_dynamic_window", False):
                self.agent_models = nn.ModuleList([SMPE2_VAE(self.obs_dim, self.n_actions, self.state_embedding_shape, self.state_dim - self.obs_dim, self.args) \
                                                .to(self.args.device)
                                                for _ in range(self.n_agents)])
            else:
                self.agent_models = nn.ModuleList([VAE(self.obs_dim, self.state_embedding_shape, self.state_dim - self.obs_dim, self.args) \
                                                .to(self.args.device)
                                                for _ in range(self.n_agents)])
            self.agent_params = [ list(model.parameters()) for model in self.agent_models ]
            self.agent_optimizers = [ RMSprop(params=param, lr=self.args.lr_agent_model) for param in self.agent_params ]
        return


    def build_agent_auxiliary_models(self):
        # Auxiliary Agent Models
        # input_shape = self.state_embedding_shape + self.n_actions * self.n_agents
        # output_shape = self.state_embedding_shape
        
        input_shape = 2 * self.state_embedding_shape
        output_shape = self.n_actions
        self.aux_models = nn.ModuleList([Aux(input_shape, output_shape, self.args) \
                                            .to(self.args.device)
                                            for _ in range(self.n_agents)])
        self.aux_agent_params = [ list(model.parameters()) for model in self.aux_models ]
        self.aux_optimizers = [ RMSprop(params=param, lr=self.args.lr_agent_model) for param in self.aux_agent_params ]
        self.aux_criterion = nn.CrossEntropyLoss(reduction='sum')
        return


    def init_hidden(self, testing=False):
        if not testing:
            self.hidden_states = th.zeros(self.n_agents, self.args.batch_size, self.state_embedding_shape).to(self.args.device)        
        else:
            self.hidden_states = th.zeros(self.n_agents, 1, self.state_embedding_shape).to(self.args.device) 
        return


    def update_stats(self, batch):
        obs = batch["obs"]
        self.obs_ms.update(obs)
        states = batch["state"]
        self.state_ms.update(states)
        self.rew_ms.update(batch["reward"])
        return


    def addBatch(self, batch: EpisodeBatch):
        # Delete unnecessary information for low memory requirements
        #new_batch = {"state": batch["state"], "obs": batch["obs"], "max_seq_length": batch.max_seq_length}
        new_batch = batch
        # Add batch sample to dataset
        self.dataset[self.dataset_count % self.dataset_size] = new_batch
        self.dataset_count += 1
        if self.dataset_count >= self.dataset_size:
            self.dataset_is_full = True
        if self.dataset_count == self.dataset_size:
            self.dataset_count = 0
        if np.random.rand() > 0.8: gc.collect() 
        return


    def sampleBatches(self, batch_size):
        if self.dataset_is_full:
            idx = list(np.random.randint(0, self.dataset_size, batch_size))
        else:
            idx = list(np.random.randint(0, self.dataset_count, batch_size))
        return itemgetter(*idx)(self.dataset)


    def forward(self, inputs, agent_id, test_mode=False, ep_batch=None, t=None, t_env=None):
        # Agent model's forward -> z
        if getattr(self.args, "use_dynamic_window", False):
            if ep_batch is not None and t is not None:
                obs_chunk, act_chunk = self._extract_window(ep_batch, t, agent_id)
                
                # Normalize
                mu_obs = self.obs_ms.mean
                std_obs = th.sqrt(self.obs_ms.var) + 1e-8
                obs_chunk = (obs_chunk - mu_obs) / std_obs

                # Curriculum masking for backward window
                if t_env is None:
                    mask_prob = 0.0
                else:
                    max_t = self.args.t_max
                    curr_t = t_env
                    mask_prob = min(1.0, max(0.0, curr_t / (0.8 * max_t)))

                _, z, _, _ = self.agent_models[agent_id].forward(
                    obs_chunk,
                    act_chunk,
                    test_mode,
                    mask_backward_prob=mask_prob,
                )
            else:
                # Fallback or error
                # For now, try to proceed if inputs is enough (unlikely) or raise error
                # Assuming BasicMAC will be updated to pass ep_batch
                raise ValueError("ep_batch and t must be provided for DynamicTimeWindow")
        else:
            _, z, _, _ = self.agent_models[agent_id].forward(inputs, test_mode)
            
        if self.args.use_detach: z = z.detach()
        return z        

    def _extract_window(self, ep_batch, t, agent_id):
        # ep_batch["obs"]: (B, T_max, N, D)
        # ep_batch["actions_onehot"]: (B, T_max, N, D)
        
        # We need [t-14, ..., t, ..., t+8]
        # Total length 23.
        # t is at index 14.
        
        start = t - 14
        end = t + 9 # Exclusive, so t+8 is included.
        
        obs = ep_batch["obs"][:, :, agent_id, :]
        act = ep_batch["actions_onehot"][:, :, agent_id, :]
        
        B, T_max, _ = obs.shape
        
        valid_start = max(0, start)
        valid_end = min(T_max, end)
        
        slice_obs = obs[:, valid_start:valid_end, :]
        slice_act = act[:, valid_start:valid_end, :]
        
        pad_left = max(0, -start)
        pad_right = max(0, end - T_max)
        
        slice_obs = th.nn.functional.pad(slice_obs, (0, 0, pad_left, pad_right))
        slice_act = th.nn.functional.pad(slice_act, (0, 0, pad_left, pad_right))
        
        return slice_obs, slice_act

    def train_agent_vaes(self, t_env):

        batch_size = self.args.agent_vae_batch_size
        epochs = self.args.agent_epochs

        first_agents_loss = []
        last_agents_loss = []
        first_agents_kl = []
        last_agents_kl = []
        first_agents_aux = []
        last_agents_aux = []

        for agent_id in range(self.n_agents):
            
            for epoch in range(epochs):
            
                big_batches = self.sampleBatches(batch_size)

                for big_batch in big_batches:
                    loss = 0
                    kl_loss__ = 0 
                    aux_loss = 0

                    if self.args.use_w and t_env % self.args.period_filter_update == 0:
                        self.filter_optimizers[agent_id].zero_grad()
                    # Get trajectory data
                    # obs = big_batch["obs"][:, :-1, agent_id, :]
                    # obs = obs.reshape(-1, self.obs_dim)
                    # states = big_batch["state"][:, :-1, :]
                    # states = states.reshape(-1, self.state_dim)
                    
                    if getattr(self.args, "use_dynamic_window", False):
                        # Only sample transitions where both (t) and (t+1) exist and are non-terminal.
                        filled_t = big_batch["filled"][:, :-1, 0].float()
                        filled_tp1 = big_batch["filled"][:, 1:, 0].float()
                        terminated_t = big_batch["terminated"][:, :-1, 0].float()
                        valid = filled_t * filled_tp1 * (1.0 - terminated_t)

                        valid_indices = th.nonzero(valid)
                        if len(valid_indices) > 32:
                            idx = np.random.choice(len(valid_indices), 32, replace=False)
                            sampled_indices = valid_indices[idx]
                        else:
                            sampled_indices = valid_indices
                        
                        obs_chunks = []
                        act_chunks = []
                        next_obs_chunks = []
                        next_act_chunks = []
                        states_list = []
                        rewards_list = []
                        actions_list = []
                        actions_onehot_others_list = []
                        
                        for b, t in sampled_indices:
                            obs_c, act_c = self._extract_window(big_batch, t, agent_id)
                            obs_chunks.append(obs_c[b])
                            act_chunks.append(act_c[b])
                            
                            obs_nc, act_nc = self._extract_window(big_batch, t+1, agent_id)
                            next_obs_chunks.append(obs_nc[b])
                            next_act_chunks.append(act_nc[b])
                            
                            states_list.append(big_batch["state"][b, t])
                            rewards_list.append(big_batch["reward"][b, t])
                            actions_list.append(big_batch["actions"][b, t])
                            actions_onehot_others_list.append(big_batch["actions_onehot"][b, t].reshape(-1))
                            
                        obs = th.stack(obs_chunks)
                        actions_onehot = th.stack(act_chunks)
                        next_obs = th.stack(next_obs_chunks)
                        next_actions_onehot = th.stack(next_act_chunks)
                        
                        states = th.stack(states_list)
                        rewards = th.stack(rewards_list)
                        actions = th.stack(actions_list)
                        actions_onehot_others = th.stack(actions_onehot_others_list)
                        
                        mu_obs = self.obs_ms.mean
                        std_obs = th.sqrt(self.obs_ms.var) + 1e-8
                        obs = (obs - mu_obs) / std_obs
                        next_obs = (next_obs - mu_obs) / std_obs
                        
                        mu_state = self.state_ms.mean
                        std_state = th.sqrt(self.state_ms.var) + 1e-8
                        states = (states - mu_state) / std_state
                        rewards = (rewards - self.rew_ms.mean) / (th.sqrt(self.rew_ms.var) + 1e-8)
                        
                        obs_center = obs[:, 14, :]
                        act_center = actions_onehot[:, 14, :]
                        inputs_w = obs_center
                        if self.args.use_actions:
                            inputs_w = th.cat((inputs_w, act_center), dim=-1)
                        if self.args.use_rewards:
                            inputs_w = th.cat((inputs_w, rewards), dim=-1)
                            
                        if self.args.use_w:
                            w_i = self.filters[agent_id].forward(inputs_w)
                            w_i_targets = self.target_filters[agent_id].forward(inputs_w)
                            w_i_targets = w_i_targets.detach()
                        
                        # Calculate mask_backward_prob based on t_env
                        # Curriculum: 0.0 at start, 1.0 at max_t_env (or earlier)
                        # Let's say we want to reach 1.0 at 80% of training.
                        max_t = self.args.t_max
                        curr_t = t_env
                        mask_prob = min(1.0, max(0.0, curr_t / (0.8 * max_t)))
                        
                        z_others, _, _ = self.agent_models[agent_id].encoder.forward(obs, actions_onehot, mask_backward_prob=mask_prob)
                        
                    else:
                        terminated = big_batch["terminated"][:, :].float()
                        mask_ = big_batch["filled"][:, :].float()  # shape: (batch_size, maxseqlen, 1)
                        mask_ = mask_ * (1 - terminated)            
                        mask = mask_[:, :-1, :]
                        mask_next = mask_[:, 1:, :]
                        mask = mask.reshape(-1, 1)
                        mask_next = mask_next.reshape(-1, 1)

                        actions_onehot = big_batch["actions_onehot"][:, :-1, agent_id, :]
                        actions_onehot = actions_onehot.reshape(-1,  self.n_actions)
                        actions_onehot_others = big_batch["actions_onehot"][:, :-1, :, :]
                        actions_onehot_others = actions_onehot_others.reshape(-1, self.n_actions * self.n_agents)
                        actions = big_batch["actions"][:, :-1, :]
                        actions = actions.reshape(-1, self.n_agents)
                        obs = big_batch["obs"][:, :-1, agent_id, :]
                        obs = obs.reshape(-1, self.obs_dim)
                        next_obs = big_batch["obs"][:, 1:, agent_id, :]
                        next_obs = next_obs.reshape(-1, self.obs_dim)
                        states = big_batch["state"][:, :-1, :]
                        rewards = big_batch["reward"][:, :-1, :]
                        rewards = rewards.reshape(-1, 1)
                        
                        states = states.reshape(-1, self.state_dim)
                        next_states = big_batch["state"][:, 1:, :]
                        next_states = next_states.reshape(-1, self.state_dim)
                        assert len(obs) == len(states)

                        idx = np.random.randint(low=0, high=len(states), size=32)
                        obs = obs[idx]
                        states = states[idx]
                        next_obs = next_obs[idx]
                        actions = actions[idx]
                        actions_onehot = actions_onehot[idx]
                        actions_onehot_others = actions_onehot_others[idx]
                        rewards = rewards[idx] 
                        if self.args.use_aux:
                            self.aux_optimizers[agent_id].zero_grad()

                        # Normalize obs and states and rewards
                        mu_obs = self.obs_ms.mean
                        std_obs = th.sqrt(self.obs_ms.var) + 1e-8
                        obs = (obs - mu_obs) / std_obs
                        mu_state = self.state_ms.mean
                        std_state = th.sqrt(self.state_ms.var) + 1e-8
                        states = (states - mu_state) / std_state
                        next_obs = (next_obs - mu_obs) / std_obs
                        rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)      

                        inputs_w = obs
                        if self.args.use_actions:
                            inputs_w = th.cat((inputs_w, actions_onehot), dim=-1)
                        if self.args.use_rewards:
                            inputs_w = th.cat((inputs_w, rewards), dim=-1)

                        # Encoding
                        if self.args.use_w:
                            w_i = self.filters[agent_id].forward(inputs_w)
                            w_i_targets = self.target_filters[agent_id].forward(inputs_w)
                            w_i_targets = w_i_targets.detach()
                        
                        # Forward pass
                        z_others, _, _ = self.agent_models[agent_id].encoder.forward(obs)
                    
                    # Filtering
                    z_others_filtered = z_others
                    predicted_states = self.agent_models[agent_id].decoder.forward(z_others_filtered)

                    last_dim = states.shape[1]
                    cut = int(last_dim/self.n_agents)
                    states = th.cat((states[:, 0:agent_id*cut], states[:, (agent_id*cut+cut):]), dim=-1)
                    actions_onehot_others = th.cat((actions_onehot_others[:, 0:agent_id*self.n_actions], 
                                                    actions_onehot_others[:, (agent_id*self.n_actions+self.n_actions):]), dim=-1)
                    #assert states.shape == predicted_states.shape

                    if self.args.use_aux:
                        # Aux Loss
                        if getattr(self.args, "use_dynamic_window", False):
                             z_others_next, _, _ = self.agent_models[agent_id].encoder.forward(next_obs, next_actions_onehot)
                        else:
                             z_others_next, _, _ = self.agent_models[agent_id].encoder.forward(next_obs)
                        
                        aux_inputs = th.cat((z_others, z_others_next), dim=-1)
                        predicted_actions_logits = self.aux_models[agent_id].forward(aux_inputs)
                        aux_loss = self.args.lambda_aux_loss * self.aux_criterion(predicted_actions_logits, actions[:, agent_id])

                    # ELBO loss
                    if self.args.use_w:
                        reconstrunction_loss = (((w_i*predicted_states - w_i_targets*states))**2).sum()
                    else:
                        reconstrunction_loss = (((predicted_states - states))**2).sum()
                    reconstrunction_loss = self.args.lambda_rec * reconstrunction_loss

                    reg_loss = 0
                    if self.args.use_w:
                        l2_lambda = self.args.l2_lambda
                        l2_reg = th.tensor(0., device=self.args.device)
                        for param in self.filters[agent_id].parameters():
                            l2_reg += th.norm(param)
                        reg_loss = l2_lambda * l2_reg

                    if self.args.use_actions:
                        predicted_actions = self.agent_models[agent_id].decoder.forward_actions(z_others_filtered)
                        actions_loss = self.args.actions_loss_lambda * ((predicted_actions - actions_onehot_others)**2).sum()
                        loss = loss + actions_loss

                    kl_loss = self.args.lambda_kl_loss_obs * self.agent_models[agent_id].encoder.kl
                    kl_loss__ = kl_loss__ + kl_loss
                    loss = loss + (reconstrunction_loss + kl_loss + reg_loss)

                    self.agent_optimizers[agent_id].zero_grad()
                    loss.backward()

                    self.agent_optimizers[agent_id].step()
                    if self.args.use_aux:
                        self.aux_optimizers[agent_id].step()

                    if self.args.use_w and t_env % self.args.period_filter_update == 0 and self.args.w_upd:
                        self.filter_optimizers[agent_id].step()    

                    loss = loss.detach()
                    kl_loss__ = kl_loss__.detach()

                if epoch == 0:
                    first_epoch_loss = loss.item()
                    first_agents_loss.append(first_epoch_loss)
                    first_epoch_kl = kl_loss__.item()
                    first_agents_kl.append(first_epoch_kl)
                    if self.args.use_aux:
                            first_agents_aux.append(aux_loss.item())
                
                if epoch == epochs - 1:
                    last_epoch_loss = loss.item()
                    last_agents_loss.append(last_epoch_loss)
                    last_epoch_kl = kl_loss__.item()
                    last_agents_kl.append(kl_loss__)
                    if self.args.use_aux:
                            last_agents_aux.append(aux_loss.item())        

        if t_env - self.last_save_t >= self.args.save_period:
            self.save_models(t_env=t_env)
            self.save_filters(t_env=t_env)
            self.load_models(t_env=t_env)
            self.load_filters(t_env=t_env)
            self.last_save_t = t_env
        return
        

    def add_intrinsic_rewards(self, batch: EpisodeBatch):
        time_dim = batch["obs"].shape[1]
        new_rewards = th.zeros(self.args.batch_size, time_dim, self.n_agents, device=self.args.device)

        for agent_id in range(self.n_agents):

            obs = batch["obs"][:, :, agent_id, :]
            # Normalize obs and states
            mu_obs = self.obs_ms.mean
            std_obs = th.sqrt(self.obs_ms.var) + 1e-8
            obs = (obs - mu_obs) / std_obs

            if getattr(self.args, "use_dynamic_window", False):
                z_list = []
                for t in range(time_dim):
                    z_t = self.forward(None, agent_id, ep_batch=batch, t=t)
                    z_list.append(z_t)
                z_others = th.stack(z_list, dim=1)
            else:
                z_others = self.forward(obs, agent_id)

            z_others = z_others.detach()
            z_others = z_others.view(-1, self.state_embedding_shape)
            self.hash_z[agent_id].inc_hash(z_others)
            z_rewards = self.hash_z[agent_id].predict(z_others)
            z_rewards = th.tensor(z_rewards, device=self.args.device)
            z_rewards = z_rewards.view(self.args.batch_size, time_dim)

            obs = obs.view(-1, self.obs_dim)
            self.hash_obs[agent_id].inc_hash(obs)
            obs_rewards = self.hash_obs[agent_id].predict(obs)    
            obs_rewards = th.tensor(obs_rewards, device=self.args.device)
            obs_rewards = obs_rewards.view(self.args.batch_size, time_dim)            

            intr_rews_agent = self.args.z_rew_coeff * z_rewards + self.args.obs_rew_coeff * obs_rewards
            new_rewards[:, :, agent_id] = self.args.true_rew_coeff * batch["reward"][:, :].squeeze(-1) + intr_rews_agent

        new_rewards = new_rewards.detach().to(self.args.device)
        return new_rewards  # batch_size x (maxseqlen-1) x num_agents x 1


    def parameters(self):
        pass


    def cuda(self):
        pass


    def save_models(self, t_env, path=None):
        if path is None:
            os.makedirs("saves", exist_ok=True)
            path = "saves/ed_" + str(t_env) + ".pth"
        elif os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        th.save(self.agent_models.state_dict(), path)
        return


    def load_models(self, t_env, path=None):
        if path is None:
            path = "saves/ed_" + str(t_env) + ".pth"
        self.agent_models.load_state_dict(th.load(path))
        return

    def save_filters(self, t_env, path=None):
        if path is None:
            os.makedirs("saves", exist_ok=True)
            path = "saves/filters_" + str(t_env) + ".pth"
        elif os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        th.save(self.filters.state_dict(), path)
        return

    def load_filters(self, t_env, path=None):
        if path is None:
            path = "saves/filters_" + str(t_env) + ".pth"
        self.filters.load_state_dict(th.load(path))
        return

