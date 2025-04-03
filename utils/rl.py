import numpy as np
import torch
import torch.nn as nn

from envs.car import CarEnv
from envs.pvtol import PvtolEnv
from envs.quadrotor import QuadRotorEnv
from envs.neurallander import NeuralLanderEnv
from envs.segway import SegwayEnv


def estimate_advantages(
    rewards, terminals, values, gamma=0.99, gae=0.95, device=torch.device("cpu")
):
    rewards, terminals, values = (
        rewards.to(torch.device("cpu")),
        terminals.to(torch.device("cpu")),
        values.to(torch.device("cpu")),
    )
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * (1 - terminals[i]) - values[i]
        advantages[i] = deltas[i] + gamma * gae * prev_advantage * (1 - terminals[i])

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    # advantages = (advantages - advantages.mean()) / advantages.std()
    advantages, returns = advantages.to(device), returns.to(device)
    return advantages, returns


def call_env(args):
    task = args.task

    if task == "car":
        env = CarEnv(sigma=args.sigma)
    elif task == "pvtol":
        env = PvtolEnv(sigma=args.sigma)
    elif task == "quadrotor":
        env = QuadRotorEnv(sigma=args.sigma)
    elif task == "neurallander":
        env = NeuralLanderEnv(sigma=args.sigma)
    elif task == "segway":
        env = SegwayEnv(sigma=args.sigma)
    else:
        raise NotImplementedError(f"{task} is not implemented.")

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.episode_len = env.episode_len

    return env


def get_policy(env, args):
    algo_name = args.algo_name
    nupdates = args.timesteps / (args.minibatch_size * args.num_minibatch)

    # this was not discussed in paper nut implemented by c3m author
    effective_indices = env.effective_indices

    if algo_name in ("lqr", "lqr-approx"):
        from policy.lqr import LQR, LQR_Approximation
        from policy.layers.dynamic_networks import DynamicLearner

        if algo_name == "lqr":
            policy = LQR(
                x_dim=env.num_dim_x,
                effective_indices=effective_indices,
                action_dim=args.action_dim,
                f_func=env.f_func,
                B_func=env.B_func,
                Bbot_func=env.Bbot_func,
                num_minibatch=args.num_minibatch,
                minibatch_size=args.minibatch_size,
                nupdates=nupdates,
            )
        else:
            Dynamic_func = DynamicLearner(
                x_dim=env.num_dim_x,
                action_dim=args.action_dim,
                hidden_dim=args.DynamicLearner_dim,
            )
            policy = LQR_Approximation(
                x_dim=env.num_dim_x,
                effective_indices=effective_indices,
                action_dim=args.action_dim,
                Dynamic_func=Dynamic_func,
                Dynamic_lr=args.Dynamic_lr,
                f_func=env.f_func,
                B_func=env.B_func,
                Bbot_func=env.Bbot_func,
                num_minibatch=args.num_minibatch,
                minibatch_size=args.minibatch_size,
                nupdates=nupdates,
                dt=env.dt,
            )

    elif algo_name in ("ppo", "ppo-manual"):
        from policy.ppo import PPO
        from policy.layers.ppo_networks import Manual_PPO_Actor, PPO_Actor, PPO_Critic

        if algo_name == "ppo":
            actor = PPO_Actor(
                args.state_dim,
                hidden_dim=args.actor_dim,
                a_dim=args.action_dim,
            )
        else:
            actor = Manual_PPO_Actor(
                x_dim=env.num_dim_x,
                effective_indices=effective_indices,
                action_dim=args.action_dim,
                task=args.task,
            )

        critic = PPO_Critic(args.state_dim, hidden_dim=args.critic_dim)

        policy = PPO(
            x_dim=env.num_dim_x,
            effective_indices=effective_indices,
            actor=actor,
            critic=critic,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            eps_clip=args.eps_clip,
            entropy_scaler=args.entropy_scaler,
            target_kl=args.target_kl,
            gamma=args.gamma,
            gae=args.gae,
            K=args.K_epochs,
            nupdates=nupdates,
            device=args.device,
        )

    elif algo_name in ("c3m", "c3m-approx"):
        from policy.c3m import C3M, C3M_Approximation
        from policy.layers.c3m_networks import C3M_W, C3M_U
        from policy.layers.dynamic_networks import DynamicLearner

        W_func = C3M_W(
            x_dim=env.num_dim_x,
            state_dim=args.state_dim,
            effective_indices=effective_indices,
            action_dim=args.action_dim,
            w_lb=args.w_lb,
            task=args.task,
            device=args.device,
        )
        u_func = C3M_U(
            x_dim=env.num_dim_x,
            state_dim=args.state_dim,
            effective_indices=effective_indices,
            action_dim=args.action_dim,
            task=args.task,
        )
        if algo_name == "c3m":
            policy = C3M(
                x_dim=env.num_dim_x,
                effective_indices=effective_indices,
                action_dim=args.action_dim,
                W_func=W_func,
                u_func=u_func,
                f_func=env.f_func,
                B_func=env.B_func,
                Bbot_func=env.Bbot_func,
                W_lr=args.W_lr,
                u_lr=args.u_lr,
                lbd=args.lbd,
                eps=args.eps,
                w_ub=args.w_ub,
                nupdates=nupdates,
                device=args.device,
            )
        else:
            Dynamic_func = DynamicLearner(
                x_dim=env.num_dim_x,
                action_dim=args.action_dim,
                hidden_dim=args.DynamicLearner_dim,
                drop_out=0.2,
            )
            policy = C3M_Approximation(
                x_dim=env.num_dim_x,
                effective_indices=effective_indices,
                action_dim=args.action_dim,
                W_func=W_func,
                u_func=u_func,
                Dynamic_func=Dynamic_func,
                f_func=env.f_func,
                B_func=env.B_func,
                Bbot_func=env.Bbot_func,
                W_lr=args.W_lr,
                u_lr=args.u_lr,
                Dynamic_lr=args.Dynamic_lr,
                lbd=args.lbd,
                eps=args.eps,
                w_ub=args.w_ub,
                nupdates=nupdates,
                dt=env.dt,
                device=args.device,
            )

    elif algo_name in ("mrl", "mrl-manual"):
        from policy.mrl import MRL
        from policy.layers.c3m_networks import C3M_W
        from policy.layers.ppo_networks import Manual_PPO_Actor, PPO_Actor, PPO_Critic

        # this was not discussed in paper nut implemented by c3m author
        W_func = C3M_W(
            x_dim=env.num_dim_x,
            state_dim=args.state_dim,
            effective_indices=effective_indices,
            action_dim=args.action_dim,
            w_lb=args.w_lb,
            task=args.task,
            device=args.device,
        )

        if algo_name == "mrl":
            actor = PPO_Actor(
                args.state_dim,
                hidden_dim=args.actor_dim,
                a_dim=args.action_dim,
            )
        else:
            actor = Manual_PPO_Actor(
                x_dim=env.num_dim_x,
                effective_indices=effective_indices,
                action_dim=args.action_dim,
                task=args.task,
            )

        critic = PPO_Critic(args.state_dim, hidden_dim=args.critic_dim)

        policy = MRL(
            x_dim=env.num_dim_x,
            effective_indices=effective_indices,
            W_func=W_func,
            f_func=env.f_func,
            B_func=env.B_func,
            Bbot_func=env.Bbot_func,
            actor=actor,
            critic=critic,
            W_lr=args.W_lr,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            w_ub=args.w_ub,
            lbd=args.lbd,
            eps=args.eps,
            eps_clip=args.eps_clip,
            entropy_scaler=args.entropy_scaler,
            target_kl=args.target_kl,
            gamma=args.gamma,
            gae=args.gae,
            K=args.K_epochs,
            nupdates=nupdates,
            dt=env.dt,
            device=args.device,
        )
    elif algo_name in ("mrl-approx", "mrl-approx-manual"):
        from policy.mrl import MRL_Approximation
        from policy.layers.c3m_networks import C3M_W
        from policy.layers.ppo_networks import Manual_PPO_Actor, PPO_Actor, PPO_Critic
        from policy.layers.dynamic_networks import DynamicLearner

        # this was not discussed in paper nut implemented by c3m author
        effective_indices = env.effective_indices

        W_func = C3M_W(
            x_dim=env.num_dim_x,
            state_dim=args.state_dim,
            effective_indices=effective_indices,
            action_dim=args.action_dim,
            w_lb=args.w_lb,
            task=args.task,
            device=args.device,
        )

        Dynamic_func = DynamicLearner(
            x_dim=env.num_dim_x,
            action_dim=args.action_dim,
            hidden_dim=args.DynamicLearner_dim,
            activation=nn.LeakyReLU(),
            drop_out=0.2,
        )

        if algo_name == "mrl-approx":
            actor = PPO_Actor(
                args.state_dim,
                hidden_dim=args.actor_dim,
                a_dim=args.action_dim,
            )
        else:
            actor = Manual_PPO_Actor(
                x_dim=env.num_dim_x,
                effective_indices=effective_indices,
                action_dim=args.action_dim,
                task=args.task,
            )

        critic = PPO_Critic(args.state_dim, hidden_dim=args.critic_dim)

        policy = MRL_Approximation(
            x_dim=env.num_dim_x,
            effective_indices=effective_indices,
            W_func=W_func,
            Dynamic_func=Dynamic_func,
            f_func=env.f_func,
            B_func=env.B_func,
            Bbot_func=env.Bbot_func,
            actor=actor,
            critic=critic,
            W_lr=args.W_lr,
            Dynamic_lr=args.Dynamic_lr,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            w_ub=args.w_ub,
            lbd=args.lbd,
            eps=args.eps,
            eps_clip=args.eps_clip,
            entropy_scaler=args.entropy_scaler,
            target_kl=args.target_kl,
            gamma=args.gamma,
            gae=args.gae,
            K=args.K_epochs,
            nupdates=nupdates,
            dt=env.dt,
            device=args.device,
        )
    elif algo_name in ("cmrl"):  # , "cmrl-approx"):
        from policy.cmrl import CMRL
        from policy.layers.c3m_networks import C3M_W, C3M_U
        from policy.layers.ppo_networks import PPO_Actor, PPO_Critic

        # from policy.layers.dynamic_networks import DynamicLearner

        # this was not discussed in paper nut implemented by c3m author
        effective_indices = env.effective_indices

        W_func = C3M_W(
            x_dim=env.num_dim_x,
            state_dim=args.state_dim,
            effective_indices=effective_indices,
            action_dim=args.action_dim,
            w_lb=args.w_lb,
            task=args.task,
            device=args.device,
        )

        u_func = C3M_U(
            x_dim=env.num_dim_x,
            state_dim=args.state_dim,
            effective_indices=effective_indices,
            action_dim=args.action_dim,
            task=args.task,
        )

        # Dynamic_func = DynamicLearner(
        #     x_dim=env.num_dim_x,
        #     action_dim=args.action_dim,
        #     hidden_dim=args.DynamicLearner_dim,
        #     activation=nn.LeakyReLU(),
        #     drop_out=0.2,
        # )

        actor = PPO_Actor(
            args.state_dim,
            hidden_dim=args.actor_dim,
            a_dim=args.action_dim,
        )

        critic = PPO_Critic(args.state_dim, hidden_dim=args.critic_dim)

        policy = CMRL(
            x_dim=env.num_dim_x,
            effective_indices=effective_indices,
            W_func=W_func,
            u_func=u_func,
            f_func=env.f_func,
            B_func=env.B_func,
            Bbot_func=env.Bbot_func,
            actor=actor,
            critic=critic,
            W_lr=args.W_lr,
            u_lr=args.u_lr,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            w_ub=args.w_ub,
            lbd=args.lbd,
            eps=args.eps,
            eps_clip=args.eps_clip,
            entropy_scaler=args.entropy_scaler,
            target_kl=args.target_kl,
            gamma=args.gamma,
            gae=args.gae,
            K=args.K_epochs,
            nupdates=nupdates,
            dt=env.dt,
            device=args.device,
        )

    elif algo_name == "mrl-ensemble":
        from policy.mrl_ensemble import MRL_Ensemble
        from policy.layers.c3m_networks import C3M_W
        from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
        from policy.layers.dynamic_networks import DynamicLearner

        # this was not discussed in paper nut implemented by c3m author
        effective_indices = env.effective_indices

        W_func = C3M_W(
            x_dim=env.num_dim_x,
            state_dim=args.state_dim,
            effective_indices=effective_indices,
            action_dim=args.action_dim,
            w_lb=args.w_lb,
            task=args.task,
            device=args.device,
        )

        Dynamic_func = DynamicLearner(
            x_dim=env.num_dim_x,
            action_dim=args.action_dim,
            hidden_dim=args.DynamicLearner_dim,
            activation=nn.LeakyReLU(),
            drop_out=0.2,
        )

        actor = PPO_Actor(
            args.state_dim,
            hidden_dim=args.actor_dim,
            a_dim=args.action_dim,
        )

        critic = PPO_Critic(args.state_dim, hidden_dim=args.critic_dim)

        policy = MRL_Ensemble(
            x_dim=env.num_dim_x,
            effective_indices=effective_indices,
            W_func=W_func,
            Dynamic_func=Dynamic_func,
            f_func=env.f_func,
            B_func=env.B_func,
            Bbot_func=env.Bbot_func,
            actor=actor,
            critic=critic,
            W_lr=args.W_lr,
            Dynamic_lr=args.Dynamic_lr,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            w_ub=args.w_ub,
            lbd=args.lbd,
            eps=args.eps,
            eps_clip=args.eps_clip,
            entropy_scaler=args.entropy_scaler,
            target_kl=args.target_kl,
            gamma=args.gamma,
            gae=args.gae,
            K=args.K_epochs,
            nupdates=nupdates,
            dt=env.dt,
            device=args.device,
        )

    return policy
