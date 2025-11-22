import os
import sys

import random
import logging
import numpy as np
import pickle

import torch
import torch.nn.functional as F

from replay_buffer import ReplayBuffer
import atari
from core import MLPActorCritic


def ge_traj(replay, env, name):
    with open('{}.pkl'.format(name), 'rb') as f:
        action_seqs = pickle.load(f)

    target_seq = None
    states = []
    for item in action_seqs[-1:]:
        list_of_actions, acc_rew, timestamp = item

        timestamp = 'ge-{}'.format(timestamp)
        score = 0
        s_t = env.reset()
        done = False

        #for a_t in list_of_actions:
        target_seq = list_of_actions
        for a_t in target_seq:
            replay.add(s_t, a_t)
            states.append(s_t)

            s_t, reward, done = env.step(a_t)
            score += reward
            if done:
                break

        #assert done
        #assert score == acc_rew, "{} / {}".format(score, acc_rew)
        print(
            't: {: >20}, score (truncated): {: >8}, traj. len: {: >8}, acc_rew (real): {}'
            .format(timestamp, score, len(target_seq), acc_rew))
        import time
        time.sleep(5)
    return target_seq, np.asarray(states)


def setup_logging(save_dir, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fh = logging.FileHandler(os.path.join(save_dir, '{}'.format(logger_name)))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def sac(
    name,
    actor_critic,
    lr,
    batch_size,
):

    env = atari.Atari(name)
    act_dim = env.action_space.n
    print('act_dim: {}'.format(act_dim))

    ac = actor_critic(4, act_dim).cuda()
    # Experience buffer
    replay_buffer = ReplayBuffer(int(1e6))
    target_seq, ge_states = ge_traj(replay_buffer, env, name)
    print(ge_states.shape)
    #exit(0)
    #_, test_states = ge_traj(replay_buffer, env, name)
    #assert (ge_states == test_states).all()
    target_seq = np.asarray(target_seq)
    opt = torch.optim.Adam(ac.parameters(), lr)

    def compute_pi_loss(data, h_0):
        logit, h_0 = ac(data['obs'].cuda(), h_0)
        act = data['act'].cuda()
        acc = ((logit.argmax(1) == act).sum() / act.shape[0]).item()
        if acc < 1.0 and random.uniform(0, 1) < 1e-2:
            print("Accuracy: {:.4f}".format(acc))
        #if random.uniform(0, 1) < 1e-3: print(logit.argmax(1))
        return F.cross_entropy(logit, act, reduction='sum') / 64, h_0

    def update(x, h_0):
        opt.zero_grad()
        loss, h_0 = compute_pi_loss(x, h_0)
        h_0 = h_0.detach()
        loss.backward()
        opt.step()
        return h_0

    @torch.no_grad()
    def get_action(o, deterministic, h_0):
        o = torch.from_numpy(np.asarray(o)).unsqueeze(0).cuda()
        return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic,
                      h_0)

    def eval(env):
        ep_ret, ep_len = 0., 0
        o = env.reset()
        done = False
        h_0 = None
        act_seq = []
        states = []
        while not done:
            act, h_0 = get_action(o, True, h_0)
            states.append(o)
            act_seq.append(act)
            o, r, done = env.step(act)
            ep_ret += r
            ep_len += 1
            if done:
                break
        logger.info('ep_ret: {}, ep_len: {}'.format(ep_ret, ep_len))
        return np.asarray(act_seq), np.asarray(states)

    logger = setup_logging('output', '{}.txt'.format('imitation'))
    h_0 = None
    while True:
        x = replay_buffer.sample(batch_size)
        if x.pop('start'):
            h_0 = None
        h_0 = update(x, h_0)
        if random.uniform(0, 1) < 1e-3:
            act_seq, states = eval(env)
            if False:
                print("#" * 20)
                print(target_seq)
                print(act_seq)
                print((states == ge_states).all())
                print(ac(torch.from_numpy(states).cuda(), None)[0].argmax(-1))


if __name__ == '__main__':
    sac(
        "MontezumaRevenge",
        MLPActorCritic,
        1e-4,
        #64,
        1024,
    )
