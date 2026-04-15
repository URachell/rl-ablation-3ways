import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════
#  VARIANT D: Multi-Head Attention DDQN
# ═══════════════════════════════════════════════════════════
class AttentionDDQN(nn.Module):
    """Multi-Head Attention DDQN for agent coordination"""
    def __init__(self, state_dim, action_dim=5, hidden_dim=256, num_heads=4):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 128), 
            nn.ReLU()
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=128, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        f = self.feature(x)
        if len(f.shape) == 2:
            f = f.unsqueeze(1)
        attn_out, _ = self.attention(f, f, f)
        attn_out = attn_out.squeeze(1) if attn_out.shape[1] == 1 else attn_out[:, 0]
        v = self.value_stream(attn_out)
        a = self.advantage_stream(attn_out)
        return v + a - a.mean(dim=-1, keepdim=True)


# ═══════════════════════════════════════════════════════════
#  VARIANT E: PPO (Policy Gradient) Network
# ═══════════════════════════════════════════════════════════
class PPOActor(nn.Module):
    """PPO Actor Network"""
    def __init__(self, state_dim, action_dim=5, hidden_dim=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        f = self.feature(x)
        log_probs = F.log_softmax(self.policy_head(f), dim=-1)
        value = self.value_head(f)
        return log_probs, value

    def get_action(self, state):
        log_probs, value = self.forward(state)
        probs = torch.exp(log_probs)
        action = torch.multinomial(probs, 1).squeeze()
        return action.item(), log_probs.gather(1, action.unsqueeze(1)).squeeze()


class PPOCritic(nn.Module):
    """PPO Critic Network"""
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU()
        )
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        f = self.feature(x)
        return self.value_head(f)
