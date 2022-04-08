import numpy as np
import torch as T
from rich import print
from src.A2C.ActorCritic import ActorNetwork, CriticNetwork
from src.A2C.PPO import PPOMemory
from torch._C import dtype


class Agent:
    def __init__(self, n_actions, gamma = 0.99, alpha = 0.0003, gae_lambda = 0.95, policy_clip = 0.2, batch_size=64, N=2048, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, terminals):
        self.memory.storeMemory(state, action, probs, vals, reward, terminals)

    def save_models(self):
        print('[bold]Saving models ... [/bold]', locals())
        self.actor.saveCheckpoint()
        self.critic.saveCheckpoint()
        print('Models saved!')

    def load_models(self):
        print('[bold]Loading models ... [/bold]', locals())
        self.actor.loadCheckpoint()
        self.critic.loadCheckpoint()
        print('Models loaded!')

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs()):
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, terminals_arr, batches = self.memory.getBatches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0

                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k]+self.gamma*values[k+1]*(1-int(terminals_arr[k]))-values[k])

                    discount *= self.gamma*self.gae_lambda
                
                advantage[t] = a_t
            
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp()/old_probs.exp()
                weighted_probs = advantage[batch]*prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 
                    1-self.policy_clip, 
                    1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch]+values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimiser.zero_grad()
                self.critic.optimiser.zero_grad()
                total_loss.backward()
                self.actor.optimiser.step()
                self.critic.optimiser.step()
        self.memory.clearMemory()