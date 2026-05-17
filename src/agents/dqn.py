"""
DQN 에이전트 직접 구현 (Double / Dueling 옵션 포함)

외부 RL 라이브러리 없이 PyTorch로 신경망·경험 저장소·학습 루프를 작성
"""
from __future__ import annotations

from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """상태 → 각 행동의 Q값(기대 누적 보상) 출력 신경망"""
    def __init__(self, obs_dim, n_actions, hidden=128, dueling=False):
        super().__init__()
        self.dueling = dueling
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        if dueling:
            self.value = nn.Linear(hidden, 1)
            self.adv = nn.Linear(hidden, n_actions)
        else:
            self.head = nn.Linear(hidden, n_actions)

    def forward(self, x):
        h = self.body(x)
        if self.dueling:
            v = self.value(h)
            a = self.adv(h)
            return v + (a - a.mean(dim=1, keepdim=True))  # Dueling: 상태 가치(V) + 행동 우위(A), 평균 차감으로 안정화
        return self.head(h)


class ReplayBuffer:
    """과거 경험(상태·행동·보상)을 저장해두고 학습 시 랜덤으로 꺼내 사용"""
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def push(self, o, a, r, no, d):
        i = self.idx
        self.obs[i], self.act[i], self.rew[i] = o, a, r
        self.next_obs[i], self.done[i] = no, d
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size, device):
        hi = len(self)
        idx = np.random.randint(0, hi, size=batch_size)
        to = lambda x: torch.as_tensor(x[idx], device=device)
        return to(self.obs), to(self.act), to(self.rew), to(self.next_obs), to(self.done)


class DQNAgent:
    """온라인 네트워크(행동 선택)와 타깃 네트워크(학습 목표)를 분리해 학습 안정화"""
    def __init__(self, obs_dim, n_actions, *, lr=1e-3, gamma=0.99, hidden=128, double=True, dueling=True, buffer_size=100_000, batch_size=128, target_sync=500, device="cpu"):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.double = double
        self.q = QNetwork(obs_dim, n_actions, hidden, dueling).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, hidden, dueling).to(self.device)
        self.target.load_state_dict(self.q.state_dict())  # 처음엔 두 신경망을 같은 상태로 맞춰 시작
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_dim)
        self.learn_steps = 0

    @torch.no_grad()
    def act(self, obs, epsilon=0.0):
        # ε-greedy: epsilon 확률 탐험(무작위), 나머지는 Q값 최대 행동 선택(활용)
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        o = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        return int(self.q(o).argmax(dim=1).item())

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size, self.device)
        q = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:
                next_a = self.q(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target(next_obs).gather(1, next_a).squeeze(1)
            else:
                next_q = self.target(next_obs).max(dim=1).values
            target = rew + self.gamma * (1 - done) * next_q  # 벨만 방정식: 현재 보상 + 다음 상태 Q값 (종료 시 0)
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)  # 기울기 클리핑: 업데이트가 너무 크게 튀지 않게
        self.opt.step()
        self.learn_steps += 1
        if self.learn_steps % self.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())  # 주기적으로만 갱신해 학습 목표를 안정적으로 유지
        return float(loss.item())


def train_dqn(env, *, total_steps=200_000, eps_start=1.0, eps_end=0.05, eps_decay_steps=50_000, learn_every=1, warmup=1000, seed=0, device="cpu", **agent_kw):
    """환경과 에이전트를 연결해 학습 진행 ― (에이전트, 로그) 반환"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n,
                     device=device, **agent_kw)
    logs = {"episode_return": [], "loss": []}

    obs, _ = env.reset(seed=seed)
    ep_ret, step = 0.0, 0
    while step < total_steps:
        frac = min(1.0, step / eps_decay_steps)
        epsilon = eps_start + frac * (eps_end - eps_start)  # epsilon 선형 감소: 초반 탐험 → 후반 활용
        action = agent.act(obs, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.buffer.push(obs, action, reward, next_obs, float(done))
        obs = next_obs
        ep_ret += reward
        step += 1
        if step > warmup and step % learn_every == 0:  # 버퍼에 충분히 쌓인 뒤 학습 시작
            loss = agent.update()
            if loss is not None:
                logs["loss"].append(loss)
        if done:
            logs["episode_return"].append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()
    return agent, logs
