"""
PPO(clip) + GAE 직접 구현 — 이산 액션용

외부 RL 라이브러리 없이 PyTorch로 actor-critic·GAE·clip loss 구현
DQN과 동일한 액션공간으로 공정 비교
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """정책(actor)·가치(critic) 헤드를 공유 신경망으로 묶은 구조"""
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)

    def act(self, obs):
        # 확률적 샘플링 ― 학습 중 탐험 (평가 시엔 argmax 사용)
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), value

    def evaluate(self, obs, act):
        # 수집한 행동을 현재 정책으로 재평가 ― ratio(정책 변화율) 계산용
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(act), dist.entropy(), value


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """GAE-lambda: 어드밴티지(평균 대비 행동 가치) 추정 ― 분산 낮춰 안정화"""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * next_nonterminal - values[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


def train_ppo(env, *, total_steps=200_000, rollout_len=2048, epochs=10,
              minibatch=256, gamma=0.99, lam=0.95, clip=0.2, lr=3e-4,
              ent_coef=0.0, vf_coef=0.5, hidden=128, seed=0, device="cpu"):
    """환경에서 PPO 학습 진행 ― (model, logs) 반환"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    model = ActorCritic(obs_dim, n_actions, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    logs = {"episode_return": [], "policy_loss": [], "value_loss": []}

    obs, _ = env.reset(seed=seed)
    ep_ret = 0.0
    step = 0
    while step < total_steps:
        # ---- rollout 수집 ---------------------------------------------- #
        b_obs = np.zeros((rollout_len, obs_dim), dtype=np.float32)
        b_act = np.zeros(rollout_len, dtype=np.int64)
        b_logp = np.zeros(rollout_len, dtype=np.float32)
        b_rew = np.zeros(rollout_len, dtype=np.float32)
        b_val = np.zeros(rollout_len, dtype=np.float32)
        b_done = np.zeros(rollout_len, dtype=np.float32)

        for i in range(rollout_len):
            ot = torch.as_tensor(obs, device=device).unsqueeze(0)
            with torch.no_grad():
                a, logp, v = model.act(ot)
            next_obs, reward, terminated, truncated, _ = env.step(int(a.item()))
            done = terminated or truncated
            b_obs[i], b_act[i], b_logp[i] = obs, int(a.item()), float(logp.item())
            b_rew[i], b_val[i], b_done[i] = reward, float(v.item()), float(done)
            obs = next_obs
            ep_ret += reward
            step += 1
            if done:
                logs["episode_return"].append(ep_ret)
                ep_ret = 0.0
                obs, _ = env.reset()

        with torch.no_grad():
            last_v = model.forward(torch.as_tensor(obs, device=device).unsqueeze(0))[1].item()
        adv, ret = compute_gae(b_rew, b_val, b_done, last_v, gamma, lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # 어드밴티지 정규화: 스케일 차이로 업데이트가 튀지 않게

        t_obs = torch.as_tensor(b_obs, device=device)
        t_act = torch.as_tensor(b_act, device=device)
        t_logp = torch.as_tensor(b_logp, device=device)
        t_adv = torch.as_tensor(adv, device=device)
        t_ret = torch.as_tensor(ret, device=device)

        # ---- PPO 업데이트 ---------------------------------------------- #
        idx = np.arange(rollout_len)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, rollout_len, minibatch):
                mb = idx[s:s + minibatch]
                new_logp, entropy, value = model.evaluate(t_obs[mb], t_act[mb])
                ratio = torch.exp(new_logp - t_logp[mb])
                a_mb = t_adv[mb]
                surr1 = ratio * a_mb
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * a_mb  # clipped surrogate: 정책이 너무 급변하지 않게 clip
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((value - t_ret[mb]) ** 2).mean()
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
            logs["policy_loss"].append(float(policy_loss.item()))
            logs["value_loss"].append(float(value_loss.item()))
    return model, logs


@torch.no_grad()
def ppo_greedy_action(model, obs, device="cpu"):
    """평가용 결정론적 행동(argmax)"""
    logits, _ = model.forward(torch.as_tensor(obs, device=torch.device(device)).unsqueeze(0))
    return int(logits.argmax(dim=1).item())
