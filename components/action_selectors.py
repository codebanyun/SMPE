import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)
        return

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()
        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector

class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        return

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector

class SoftPoliciesSelector():

    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)
        return

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # agent_inputs is expected to be action probabilities (already softmaxed).
        # Always mask out unavailable actions.
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            # Categorical(probs=...) requires at least one positive prob per row.
            probs_sum = masked_policies.sum(dim=2, keepdim=True)
            all_zero = probs_sum.squeeze(2) == 0
            if all_zero.any():
                # Fallback to uniform over available actions.
                uniform = avail_actions.float()
                uniform_sum = uniform.sum(dim=2, keepdim=True)
                uniform = th.where(uniform_sum > 0, uniform / uniform_sum, th.full_like(uniform, 1.0 / uniform.size(2)))
                masked_policies[all_zero] = uniform[all_zero]

            m = Categorical(probs=masked_policies)
            picked_actions = m.sample().long()

        return picked_actions           # shape: (batch_size, num_agents)


REGISTRY["soft_policies"] = SoftPoliciesSelector