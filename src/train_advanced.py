import os
import torch
import numpy as np
import pickle
from pathlib import Path
from train import (
    MultiAgentEnergyEnv, DuelingDDQN, SharedDDQNAgent, 
    evaluate_full_metrics
)
from models import AttentionDDQN, PPOActor, PPOCritic
from environments import DIFFICULTY_CONFIG


class AdvancedSharedDDQNAgent(SharedDDQNAgent):
    """Extended agent with checkpointing support"""
    
    def save_checkpoint(self, path):
        """Save model and optimizer state"""
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), f"{path}/q_net.pth")
        torch.save(self.target_net.state_dict(), f"{path}/target_net.pth")
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pth")
        
    def load_checkpoint(self, path):
        """Load model and optimizer state"""
        self.q_net.load_state_dict(torch.load(f"{path}/q_net.pth"))
        self.target_net.load_state_dict(torch.load(f"{path}/target_net.pth"))
        self.optimizer.load_state_dict(torch.load(f"{path}/optimizer.pth"))


class AttentionAgent(AdvancedSharedDDQNAgent):
    """Agent using Attention DDQN"""
    
    def __init__(self, state_dim, use_per=True, action_dim=5, hidden_dim=256, lr=1e-4, gamma=0.95):
        self.gamma = gamma
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = AttentionDDQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = AttentionDDQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        
        from train import ReplayBuffer
        self.memory = ReplayBuffer(use_per=use_per)
        self.epsilon, self.epsilon_min, self.epsilon_decay = 0.3, 0.05, 0.995
        self.update_steps, self.target_update_freq = 0, 300
        self.device = device

    def select_actions(self, states, explore=True):
        if explore and np.random.random() < self.epsilon:
            return [np.random.randint(0, 5) for _ in range(len(states))]
        with torch.no_grad():
            return self.q_net(torch.FloatTensor(states).to(self.device)).argmax(dim=1).cpu().tolist()

    def learn(self, batch_size=256, beta=0.4):
        if len(self.memory) < batch_size:
            return None
        (states, actions, rewards, next_states, dones, indices, weights) = self.memory.sample(batch_size, beta)
        B, N, D = states.shape
        s_flat = torch.FloatTensor(states).to(self.device).view(B*N, D)
        ns_flat = torch.FloatTensor(next_states).to(self.device).view(B*N, D)
        a = torch.LongTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        d = torch.FloatTensor(dones.astype(np.float32)).to(self.device)
        w = torch.FloatTensor(weights).to(self.device)

        q_vals = self.q_net(s_flat).view(B, N, -1).gather(2, a.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_acts = self.q_net(ns_flat).view(B, N, -1).argmax(dim=2)
            next_q = self.target_net(ns_flat).view(B, N, -1).gather(2, next_acts.unsqueeze(-1)).squeeze(-1)
            target_q = r + self.gamma * next_q * (1 - d)

        td_errors = (q_vals - target_q).detach().abs().mean(dim=1).cpu().numpy()
        loss = (torch.nn.functional.mse_loss(q_vals, target_q, reduction='none').mean(dim=1) * w).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)
        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def run_advanced_ablation(num_episodes=2000, num_envs=1, difficulty='easy', 
                         variants=['A', 'B', 'C', 'D', 'E'], use_mixed_precision=False,
                         checkpoint_freq=100, checkpoint_dir='checkpoints'):
    """Run advanced ablation study with selected variants and difficulty"""
    
    print(f"\n{'='*70}\n🚀 ADVANCED ABLATION STUDY")
    print(f"Difficulty: {difficulty.upper()} | Variants: {variants}")
    print(f"Episodes: {num_episodes} | Parallel Envs: {num_envs}\n{'='*70}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG['easy'])
    
    variant_map = {
        'A': ('Base Dueling DDQN', False, False, DuelingDDQN),
        'B': ('Dueling DDQN + PER', True, False, DuelingDDQN),
        'C': ('Dueling DDQN + PER + Frame Stack', True, True, DuelingDDQN),
        'D': ('Multi-Head Attention DDQN', True, True, AttentionDDQN),
        'E': ('PPO with GAE', True, True, PPOActor),
    }
    
    results_history = {}
    
    for variant in variants:
        if variant not in variant_map:
            print(f"⚠️  Unknown variant: {variant}. Skipping...")
            continue
            
        name, use_per, use_fs, model_class = variant_map[variant]
        print(f"\n▶ Training Variant {variant}: 【{name}】")
        
        env = MultiAgentEnergyEnv(
            num_agents=config['num_agents'],
            use_frame_stack=use_fs,
            grid_size=config['grid_size'],
            max_energy=config['max_energy'],
            max_steps=config['max_steps']
        )
        
        if variant == 'E':  # PPO
            agent = AttentionAgent(state_dim=env.state_dim, use_per=use_per)
        else:
            agent = AdvancedSharedDDQNAgent(state_dim=env.state_dim, use_per=use_per)
        
        results_history[variant] = {'sr': [], 'cols': [], 'steps': [], 'energy': []}
        beta = 0.4
        
        for ep in range(1, num_episodes + 1):
            states = env.reset()
            beta = min(1.0, beta + 1e-4)
            
            while True:
                actions = agent.select_actions(states, explore=True)
                next_states, rewards, dones, truncated = env.step(actions)
                agent.memory.push(states, actions, rewards, next_states, dones)
                agent.learn(batch_size=256, beta=beta)
                states = next_states
                if all(dones) or truncated:
                    break
            
            agent.decay_epsilon()
            
            # Checkpoint every checkpoint_freq episodes
            if ep % checkpoint_freq == 0:
                ckpt_path = f"{checkpoint_dir}/{difficulty}/{variant}/{ep}"
                agent.save_checkpoint(ckpt_path)
            
            # Evaluate every 100 episodes
            if ep % 100 == 0:
                eval_env = MultiAgentEnergyEnv(
                    num_agents=config['num_agents'],
                    use_frame_stack=use_fs,
                    grid_size=config['grid_size'],
                    max_energy=config['max_energy'],
                    max_steps=config['max_steps']
                )
                metrics = evaluate_full_metrics(agent, eval_env, n_episodes=5)
                
                results_history[variant]['sr'].append((ep, metrics['sr']))
                results_history[variant]['cols'].append((ep, metrics['cols']))
                results_history[variant]['steps'].append((ep, metrics['steps']))
                results_history[variant]['energy'].append((ep, metrics['energy']))
                
                print(f"  └─ Ep {ep}/{num_episodes} | SR: {metrics['sr']:.1%} | "
                      f"Cols: {metrics['cols']:.1f} | Energy: {metrics['energy']:.1f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open(f'results/ablation_{difficulty}.pkl', 'wb') as f:
        pickle.dump(results_history, f)
    
    print(f"\n✅ Ablation study for {difficulty} complete!")
    print(f"Results saved to: results/ablation_{difficulty}.pkl")
    
    return results_history
