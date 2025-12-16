import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
from collections import deque

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class ReplayBuffer:
    """Experience replay buffer for off-policy learning"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Stochastic policy network for SAC"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.sigmoid(x_t)  # Bounded to [0,1]
        
        # Log probability with change of variables correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(action * (1 - action) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """Q-function network for SAC"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 network
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2 network
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        # Q1
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        # Q2
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1


class SACAgent:
    """Soft Actor-Critic agent for thermostat control"""
    def __init__(
        self,
        state_dim=5,
        action_dim=1,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_entropy_tuning=True,
        buffer_capacity=100000
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy parameters to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Automatic entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            self.target_entropy = -action_dim  # Heuristic: -dim(A)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Metrics
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.alpha_history = []
        
    def select_action(self, state, evaluate=False):
        """Select action from policy"""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor(state)
                action = torch.sigmoid(mean)
        else:
            action, _ = self.actor.sample(state)
        
        return action.cpu().data.numpy()[0]
    
    def update_parameters(self, batch_size=256, updates=1):
        """Update actor and critic networks"""
        if len(self.replay_buffer) < batch_size:
            return None, None, None
        
        total_actor_loss = 0
        total_critic_loss = 0
        
        for _ in range(updates):
            # Sample batch
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
                self.replay_buffer.sample(batch_size)
            
            state_batch = state_batch.to(self.device)
            action_batch = action_batch.to(self.device)
            reward_batch = reward_batch.to(self.device)
            next_state_batch = next_state_batch.to(self.device)
            done_batch = done_batch.to(self.device)
            
            # ======= Update Critic =======
            with torch.no_grad():
                next_action, next_log_prob = self.actor.sample(next_state_batch)
                q1_next, q2_next = self.critic_target(next_state_batch, next_action)
                q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
                q_target = reward_batch + (1 - done_batch) * self.gamma * q_next
            
            q1, q2 = self.critic(state_batch, action_batch)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # ======= Update Actor =======
            new_action, log_prob = self.actor.sample(state_batch)
            q1_new, q2_new = self.critic(state_batch, new_action)
            q_new = torch.min(q1_new, q2_new)
            
            actor_loss = (self.alpha * log_prob - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # ======= Update Alpha (optional) =======
            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = self.log_alpha.exp()
            
            # ======= Soft update target network =======
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
        
        # Store metrics
        avg_actor_loss = total_actor_loss / updates
        avg_critic_loss = total_critic_loss / updates
        self.actor_loss_history.append(avg_actor_loss)
        self.critic_loss_history.append(avg_critic_loss)
        
        if self.auto_entropy_tuning:
            self.alpha_history.append(self.alpha.item())
        
        return avg_actor_loss, avg_critic_loss, self.alpha.item() if self.auto_entropy_tuning else self.alpha
    
    def save(self, path):
        """Save agent state"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None,
        }, path)
        print(f"SAC agent saved to {path}")
    
    def load(self, path):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if self.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = self.log_alpha.exp()
        
        print(f"SAC agent loaded from {path}")
