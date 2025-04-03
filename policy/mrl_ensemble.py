import time
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch import matmul, inverse, transpose
from torch.linalg import matrix_norm
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from typing import Callable
from copy import deepcopy

# from utils.torch import get_flat_grad_from, get_flat_params_from, set_flat_params_to
from utils.rl import estimate_advantages

# from actor.layers.building_blocks import MLP
from policy.base import Base


# from models.layers.ppo_networks import PPO_Policy, PPO_Critic


class MRL_Ensemble(Base):
    def __init__(
        self,
        x_dim: int,
        effective_indices: list,
        Dynamic_func: nn.Module,
        W_func: nn.Module,
        f_func: Callable,
        B_func: Callable,
        Bbot_func: Callable,
        actor: nn.Module,
        critic: nn.Module,
        Dynamic_lr: float = 3e-4,
        W_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        w_ub: float = 1e-2,
        lbd: float = 1e-2,
        eps: float = 1e-2,
        eps_clip: float = 0.2,
        entropy_scaler: float = 1e-3,
        l2_reg: float = 1e-8,
        target_kl: float = 0.03,
        gamma: float = 0.99,
        gae: float = 0.9,
        K: int = 5,
        nupdates: int = 0,
        dt: float = 0.03,
        device: str = "cpu",
    ):
        super(MRL_Ensemble, self).__init__()

        # constants
        self.name = "MRL_Ensemble"
        self.device = device

        self.x_dim = x_dim
        self.action_dim = actor.action_dim
        self.effective_x_dim = len(effective_indices)
        self.effective_indices = effective_indices

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size
        self._entropy_scaler = entropy_scaler
        self.eps = eps
        self.gamma = gamma
        self.gae = gae
        self.K = K
        self.l2_reg = l2_reg
        self.target_kl = target_kl
        self.lbd = lbd
        self.eps_clip = eps_clip
        self.w_ub = w_ub
        self.dt = dt

        self.f_func = f_func
        self.B_func = B_func
        if Bbot_func is None:
            self.Bbot_func = self.B_null
        else:
            self.Bbot_func = Bbot_func

        self._forward_steps = 0
        self.nupdates = nupdates
        self.num_outer_update = 0
        self.num_inner_update = 0

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        # trainable networks
        self.Dynamic_func = Dynamic_func
        self.modes = ["increase", "none", "decrease"]
        self.W_funcs = {
            "increase": deepcopy(W_func).to(self.device),
            "none": deepcopy(W_func).to(self.device),
            "decrease": deepcopy(W_func).to(self.device),
        }

        self.Dynamic_optimizer = torch.optim.Adam(
            self.Dynamic_func.parameters(), lr=Dynamic_lr
        )

        self.W_optimizer = torch.optim.Adam(
            params=[p for W_func in self.W_funcs.values() for p in W_func.parameters()],
            lr=W_lr,
        )

        self.actor = actor
        self.critic = critic

        self.cloned_actor = deepcopy(self.actor)
        self.cloned_critic = deepcopy(self.critic)

        self.ppo_optimizer = torch.optim.Adam(
            [
                {"params": self.cloned_actor.parameters(), "lr": actor_lr},
                {"params": self.cloned_critic.parameters(), "lr": critic_lr},
            ]
        )

        self.tau = 0.5  # for soft update of target networks

        self.W_lr_scheduler = LambdaLR(self.W_optimizer, lr_lambda=self.W_lr_fn)
        self.D_lr_scheduler = LambdaLR(self.Dynamic_optimizer, lr_lambda=self.D_lr_fn)

        #
        self.to(self.device)

    def W_lr_fn(self, step):
        return 1.0 - float(step) / float(self.nupdates)

    def D_lr_fn(self, step):
        return 0.999**step

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        self._forward_steps += 1
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        x, xref, uref, x_trim, xref_trim = self.trim_state(state)
        a, metaData = self.cloned_actor(
            x, xref, uref, x_trim, xref_trim, deterministic=deterministic
        )

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
        }

    def contraction_loss(
        self,
        states: torch.Tensor,
        mode: str,
        detach: bool,
    ):
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        x = x.requires_grad_()

        W = self.W_funcs[mode](x, xref, uref, x_trim, xref_trim)
        M = inverse(W)

        f = self.f_func(x).to(self.device)  # n, x_dim
        B = self.B_func(x).to(self.device)  # n, x_dim, action

        DfDx = self.Jacobian(f, x)  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x)  # n, x_dim, x_dim, b_dim
        Bbot = self.Bbot_func(x).to(self.device)  # n, x_dim, state - action dim

        u, _ = self.cloned_actor(x, xref, uref, x_trim, xref_trim, deterministic=True)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

        u = u.detach()
        K = K.detach()

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.action_dim)
            ]
        )

        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
        dot_M = self.weighted_gradients(M, dot_x, x, detach)
        ABK = A + matmul(B, K)
        ABK = self.perturb_eigenvalues(ABK, mode=mode)
        if detach:
            MABK = matmul(M.detach(), ABK)
            sym_MABK = MABK + transpose(MABK, 1, 2)
            C_u = dot_M + sym_MABK + 2 * self.lbd * M.detach()
        else:
            MABK = matmul(M, ABK)
            sym_MABK = MABK + transpose(MABK, 1, 2)
            C_u = dot_M + sym_MABK + 2 * self.lbd * M

        # C1
        DfW = self.weighted_gradients(W, f, x, detach)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = DfDxW + transpose(DfDxW, 1, 2)

        # this has to be a negative definite matrix
        C1_inner = -DfW + sym_DfDxW + 2 * self.lbd * W
        C1 = matmul(matmul(transpose(Bbot, 1, 2), C1_inner), Bbot)

        C2_inners = []
        C2s = []
        for j in range(self.action_dim):
            DbW = self.weighted_gradients(W, B[:, :, j], x, detach)
            DbDxW = matmul(DBDx[:, :, :, j], W)
            sym_DbDxW = DbDxW + transpose(DbDxW, 1, 2)
            C2_inner = DbW - sym_DbDxW
            C2 = matmul(matmul(transpose(Bbot, 1, 2), C2_inner), Bbot)
            C2_inners.append(C2_inner)
            C2s.append(C2)

        #### COMPUTE LOSS ####
        pd_loss = self.loss_pos_matrix_random_sampling(
            -C_u - self.eps * torch.eye(C_u.shape[-1]).to(self.device)
        )
        c1_loss = self.loss_pos_matrix_random_sampling(
            -C1 - self.eps * torch.eye(C1.shape[-1]).to(self.device)
        )
        # c2_loss = sum([C2.sum().mean() for C2 in C2s])
        c2_loss = sum([(matrix_norm(C2) ** 2).mean() for C2 in C2s])
        overshoot_loss = self.loss_pos_matrix_random_sampling(
            self.w_ub * torch.eye(W.shape[-1]).to(self.device) - W
        )

        loss = pd_loss + c1_loss + c2_loss + overshoot_loss

        ### for loggings
        with torch.no_grad():
            dot_M_pos_eig, dot_M_neg_eig = self.get_matrix_eig(dot_M)
            sym_MABK_pos_eig, sym_MABK_neg_eig = self.get_matrix_eig(sym_MABK)
            M_pos_eig, M_neg_eig = self.get_matrix_eig(M)

            C_pos_eig, C_neg_eig = self.get_matrix_eig(C_u)
            C1_pos_eig, C1_neg_eig = self.get_matrix_eig(C1)

        return (
            loss,
            {
                "pd_loss": pd_loss.item(),
                "C1_loss": c1_loss.item(),
                "C2_loss": c2_loss.item(),
                "overshoot_loss": overshoot_loss.item(),
                "C_pos_eig": C_pos_eig.item(),
                "C_neg_eig": C_neg_eig.item(),
                "C1_pos_eig": C1_pos_eig.item(),
                "C1_neg_eig": C1_neg_eig.item(),
                "dot_M_pos_eig": dot_M_pos_eig.item(),
                "dot_M_neg_eig": dot_M_neg_eig.item(),
                "sym_MABK_pos_eig": sym_MABK_pos_eig.item(),
                "sym_MABK_neg_eig": sym_MABK_neg_eig.item(),
                "M_pos_eig": M_pos_eig.item(),
                "M_neg_eig": M_neg_eig.item(),
            },
        )

    def get_matrix_eig(self, A: torch.Tensor):
        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(A)  # (batch, dim), real symmetric
            pos_eigvals = torch.relu(eigvals)
            neg_eigvals = torch.relu(-eigvals)
        return pos_eigvals.mean(dim=1).mean(), neg_eigvals.mean(dim=1).mean()

    def learn(self, batch):
        if self.num_inner_update <= int(0.1 * self.nupdates):
            loss_dict, update_time = self.learn_Dynamics(batch)
            loss_dict = {}
            timesteps = 0
            update_time = 0
            # timesteps = batch["rewards"].shape[0]
            self.num_inner_update += 1
        else:
            detach = (
                True if self.num_outer_update <= int(0.1 * self.nupdates) else False
            )

            loss_dict, timesteps, update_time = self.learn_ppo(batch)

            if self.num_inner_update % 10 == 0:
                D_loss_dict, D_update_time = self.learn_Dynamics(batch)
                W_loss_dict, W_update_time = self.learn_W(batch, detach)

                self.update_params()

                loss_dict.update(W_loss_dict)
                loss_dict.update(D_loss_dict)
                update_time += W_update_time
                update_time += D_update_time

                self.num_outer_update += 1
                self.W_lr_scheduler.step()
                self.D_lr_scheduler.step()
            else:
                loss_dict = {}
                timesteps = 0
                update_time = 0

            self.num_inner_update += 1

        return loss_dict, timesteps, update_time

    def learn_Dynamics(self, batch: dict):
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        actions = to_tensor(batch["actions"])

        x, _, _, _, _ = self.trim_state(states)
        f_approx, B_approx, Bbot_approx = self.Dynamic_func(x)

        f = self.f_func(x).to(self.device)  # n, x_dim
        B = self.B_func(x).to(self.device)  # n, x_dim, action

        dot_x_true = f + matmul(B, actions.unsqueeze(-1)).squeeze(-1)
        dot_x_approx = f_approx + matmul(B_approx, actions.unsqueeze(-1)).squeeze(-1)

        fB_loss = F.mse_loss(dot_x_true, dot_x_approx)

        ortho_loss = torch.mean(
            (matrix_norm(matmul(Bbot_approx.transpose(1, 2), B_approx.detach())))
        )

        loss = fB_loss + ortho_loss  # + adversarial_loss  # + cont_loss
        with torch.no_grad():
            f_error = F.l1_loss(f, f_approx)
            B_error = F.l1_loss(B, B_approx)

        self.Dynamic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Dynamic_func.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.Dynamic_func],
            ["Dynamic_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        self.Dynamic_optimizer.step()

        norm_dict = self.compute_weight_norm(
            [self.Dynamic_func],
            ["Dynamic_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict = {
            f"{self.name}/Dynamic_loss/loss": loss.item(),
            f"{self.name}/Dynamic_loss/fB_loss": fB_loss.item(),
            f"{self.name}/Dynamic_loss/ortho_loss": ortho_loss.item(),
            # f"{self.name}/Dynamic_loss/adversarial_loss": adversarial_loss.item(),
            f"{self.name}/Dynamic_analytics/f_error": f_error.item(),
            f"{self.name}/Dynamic_analytics/B_error": B_error.item(),
            f"{self.name}/learning_rate/D_lr": self.Dynamic_optimizer.param_groups[0][
                "lr"
            ],
        }
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states, x
        self.eval()

        update_time = time.time() - t0
        return loss_dict, update_time

    def learn_W(self, batch: dict, detach: bool):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])

        # List to track actor loss over minibatches
        losses = []
        infos = []
        grad_dicts = []
        norm_dicts = []
        for mode in self.modes:
            loss, info = self.contraction_loss(states, mode, detach)

            self.W_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.W_funcs[mode].parameters(), max_norm=10.0
            )
            grad_dict = self.compute_gradient_norm(
                [self.W_funcs[mode]],
                ["W_func"],
                dir=f"{self.name}",
                device=self.device,
            )
            self.W_optimizer.step()
            norm_dict = self.compute_weight_norm(
                [self.W_funcs[mode]],
                ["W_func"],
                dir=f"{self.name}",
                device=self.device,
            )

            losses.append(loss)
            infos.append(info)
            grad_dicts.append(grad_dict)
            norm_dicts.append(norm_dict)

        loss = torch.mean(torch.stack(losses))
        infos = self.average_dict_values(infos)
        grad_dict = self.average_dict_values(grad_dicts)
        norm_dict = self.average_dict_values(norm_dicts)

        # Logging
        loss_dict = {
            f"{self.name}/W_loss/loss": loss.item(),
            f"{self.name}/W_loss/pd_loss": infos["pd_loss"],
            f"{self.name}/W_loss/C1_loss": infos["C1_loss"],
            f"{self.name}/W_loss/C2_loss": infos["C2_loss"],
            f"{self.name}/W_loss/overshoot_loss": infos["overshoot_loss"],
            f"{self.name}/C_analytics/C_pos_eig": infos["C_pos_eig"],
            f"{self.name}/C_analytics/C_neg_eig": infos["C_neg_eig"],
            f"{self.name}/C_analytics/C1_pos_eig": infos["C1_pos_eig"],
            f"{self.name}/C_analytics/C1_neg_eig": infos["C1_neg_eig"],
            f"{self.name}/C_analytics/dot_M_pos_eig": infos["dot_M_pos_eig"],
            f"{self.name}/C_analytics/dot_M_neg_eig": infos["dot_M_neg_eig"],
            f"{self.name}/C_analytics/sym_MABK_pos_eig": infos["sym_MABK_pos_eig"],
            f"{self.name}/C_analytics/sym_MABK_neg_eig": infos["sym_MABK_neg_eig"],
            f"{self.name}/C_analytics/M_pos_eig": infos["M_pos_eig"],
            f"{self.name}/C_analytics/M_neg_eig": infos["M_neg_eig"],
        }

        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states
        self.eval()

        update_time = time.time() - t0
        return loss_dict, update_time

    def update_params(self):
        # soft update
        for param_main, param_clone in zip(
            self.actor.parameters(), self.cloned_actor.parameters()
        ):
            param_main.data.copy_(
                (1.0 - self.tau) * param_main.data + self.tau * param_clone.data
            )
        for param_main, param_clone in zip(
            self.critic.parameters(), self.cloned_critic.parameters()
        ):
            param_main.data.copy_(
                (1.0 - self.tau) * param_main.data + self.tau * param_clone.data
            )

        self.cloned_actor = deepcopy(self.actor)
        self.cloned_critic = deepcopy(self.critic)

        self.ppo_optimizer = torch.optim.Adam(
            [
                {"params": self.cloned_actor.parameters(), "lr": self.actor_lr},
                {"params": self.cloned_critic.parameters(), "lr": self.critic_lr},
            ]
        )

    def learn_ppo(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        actions = to_tensor(batch["actions"])
        original_rewards = to_tensor(batch["rewards"])
        rewards = self.get_rewards(states)
        terminals = to_tensor(batch["terminals"])
        old_logprobs = to_tensor(batch["logprobs"])

        # Compute advantages and returns
        with torch.no_grad():
            values = self.cloned_critic(states)
            advantages, returns = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self.gamma,
                gae=self.gae,
                device=self.device,
            )

        # Mini-batch training
        batch_size = states.size(0)

        # List to track actor loss over minibatches
        losses = []
        actor_losses = []
        value_losses = []
        entropy_losses = []

        clip_fractions = []
        target_kl = []
        grad_dicts = []

        for k in range(self.K):
            for n in range(self.num_minibatch):
                indices = torch.randperm(batch_size)[: self.minibatch_size]
                mb_states, mb_actions = states[indices], actions[indices]
                mb_old_logprobs, mb_returns = old_logprobs[indices], returns[indices]

                # advantages
                mb_advantages = advantages[indices]
                mb_advantages = (
                    mb_advantages - mb_advantages.mean()
                ) / mb_advantages.std()

                # 1. Critic Update (with optional regularization)
                mb_values = self.cloned_critic(mb_states)
                value_loss = self.mse_loss(mb_values, mb_returns)
                l2_reg = (
                    sum(param.pow(2).sum() for param in self.cloned_critic.parameters())
                    * self.l2_reg
                )
                value_loss += l2_reg

                # Track value loss for logging
                value_losses.append(value_loss.item())

                # 2. actor Update
                x, xref, uref, x_trim, xref_trim = self.trim_state(mb_states)
                _, metaData = self.cloned_actor(x, xref, uref, x_trim, xref_trim)
                logprobs = self.cloned_actor.log_prob(metaData["dist"], mb_actions)
                entropy = self.cloned_actor.entropy(metaData["dist"])
                ratios = torch.exp(logprobs - mb_old_logprobs)

                surr1 = ratios * mb_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * mb_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = self._entropy_scaler * entropy.mean()

                # Track actor loss for logging
                actor_losses.append(actor_loss.item())
                entropy_losses.append(entropy_loss.item())

                # Total loss
                loss = actor_loss + 0.5 * value_loss - entropy_loss
                losses.append(loss.item())

                # Compute clip fraction (for logging)
                clip_fraction = torch.mean(
                    (torch.abs(ratios - 1) > self.eps_clip).float()
                ).item()
                clip_fractions.append(clip_fraction)

                # Check if KL divergence exceeds target KL for early stopping
                kl_div = torch.mean(mb_old_logprobs - logprobs)
                target_kl.append(kl_div.item())
                if kl_div.item() > self.target_kl:
                    break

                # Update critic parameters
                self.ppo_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                grad_dict = self.compute_gradient_norm(
                    [self.cloned_actor, self.cloned_critic],
                    ["actor", "critic"],
                    dir=f"{self.name}",
                    device=self.device,
                )
                grad_dicts.append(grad_dict)
                self.ppo_optimizer.step()

            if kl_div.item() > self.target_kl:
                break

        # Logging
        loss_dict = {
            f"{self.name}/loss/loss": np.mean(losses),
            f"{self.name}/loss/actor_loss": np.mean(actor_losses),
            f"{self.name}/loss/value_loss": np.mean(value_losses),
            f"{self.name}/loss/entropy_loss": np.mean(entropy_losses),
            f"{self.name}/analytics/clip_fraction": np.mean(clip_fractions),
            f"{self.name}/analytics/klDivergence": target_kl[-1],
            f"{self.name}/analytics/K-epoch": k + 1,
            f"{self.name}/analytics/avg_rewards": torch.mean(original_rewards).item(),
            f"{self.name}/analytics/corrected_avg_rewards": torch.mean(rewards).item(),
        }
        grad_dict = self.average_dict_values(grad_dicts)
        norm_dict = self.compute_weight_norm(
            [self.cloned_actor, self.cloned_critic],
            ["actor", "critic"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states, actions, rewards, terminals, old_logprobs
        self.eval()

        timesteps = self.num_minibatch * self.minibatch_size
        update_time = time.time() - t0
        return loss_dict, timesteps, update_time

    def perturb_eigenvalues(self, A: torch.Tensor, scale=0.2, mode="increase"):
        if mode == "none":
            return A

        # Eigen-decomposition
        eigvals, eigvecs = torch.linalg.eig(A)

        # eigvals may be complex, so handle absolute value properly
        perturb = scale * eigvals.abs()

        if mode == "increase":
            new_eigvals = eigvals + perturb
        elif mode == "decrease":
            new_eigvals = eigvals - perturb
        else:
            raise ValueError("mode must be 'increase' or 'decrease'")

        # Reconstruct perturbed matrix: A' = V Λ' V⁻¹
        V = eigvecs
        D_new = torch.diag_embed(new_eigvals)
        V_inv = torch.linalg.inv(V)
        A_perturbed = V @ D_new @ V_inv

        return A_perturbed.real  # You can drop imaginary if matrix is real

    def loss_pos_matrix_random_sampling(self, A: torch.Tensor):
        # A: n x d x d
        # z: K x d
        n, A_dim, _ = A.shape

        z = torch.randn((n, A_dim)).to(self.device)
        z = z / z.norm(dim=-1, keepdim=True)
        z = z.unsqueeze(-1)
        zT = transpose(z, 1, 2)

        # K x d @ d x d = n x K x d
        zTAz = matmul(matmul(zT, A), z)

        negative_index = zTAz.detach().cpu().numpy() < 0
        if negative_index.sum() > 0:
            negative_zTAz = zTAz[negative_index]
            return -1.0 * (negative_zTAz.mean())
        else:
            return torch.tensor(0.0).to(self.device).requires_grad_()

    def trim_state(self, state: torch.Tensor):
        # state trimming
        x = state[:, : self.x_dim]
        xref = state[:, self.x_dim : -self.action_dim]
        uref = state[:, -self.action_dim :]

        x_trim = x[:, self.effective_indices]
        xref_trim = xref[:, self.effective_indices]

        return x, xref, uref, x_trim, xref_trim

    def get_rewards(self, states):
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        error = (x - xref).unsqueeze(-1)
        errorT = transpose(error, 1, 2)

        x = x.requires_grad_()

        f_approx, B_approx, _ = self.Dynamic_func(x)
        DfDx = self.Jacobian(f_approx, x).detach()  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B_approx, x).detach()  # n, x_dim, x_dim, b_dim

        f_approx = f_approx.detach()
        B_approx = B_approx.detach()

        u, _ = self.cloned_actor(x, xref, uref, x_trim, xref_trim, deterministic=True)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

        u = u.detach()
        K = K.detach()

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.action_dim)
            ]
        )

        ABK = A + matmul(B_approx, K)

        rewards_list = []
        for mode in self.modes:
            with torch.no_grad():
                W = self.W_funcs[mode](x, xref, uref, x_trim, xref_trim)
                M = torch.inverse(W)

                rewards = (1 / (errorT @ M @ error + 1)).squeeze(-1)

            perturbed_ABK = self.perturb_eigenvalues(ABK, mode=mode)
            MABK = matmul(M, perturbed_ABK)
            sym_MABK = MABK + transpose(MABK, 1, 2)

            C_u_only = -sym_MABK - self.eps * torch.eye(sym_MABK.shape[-1]).to(
                self.device
            )

            aux_rewards = torch.linalg.eigvalsh(C_u_only).mean(dim=1).unsqueeze(-1)

            pos_indices = aux_rewards > 0
            neg_indices = aux_rewards <= 0

            aux_rewards[pos_indices] = torch.tanh(aux_rewards[pos_indices] / 30)
            aux_rewards[neg_indices] = -1.0

            alpha = 0.5
            rewards = alpha * rewards + (1 - alpha) * aux_rewards

            rewards_list.append(rewards)

        rewards = torch.cat(rewards_list, dim=-1).mean(dim=-1, keepdim=True)

        return rewards

    def get_true_metrics(self, states: torch.Tensor):
        #### COMPUTE THE REAL DYNAMICS TO MEASURE ERRORS ####
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        x = x.requires_grad_()

        with torch.no_grad():
            W = self.W_func(x, xref, uref, x_trim, xref_trim)
            M = inverse(W)

        f = self.f_func(x).to(self.device)  # n, x_dim
        B = self.B_func(x).to(self.device)  # n, x_dim, action

        DfDx = self.Jacobian(f, x).detach()  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x).detach()  # n, x_dim, x_dim, b_dim
        Bbot = self.Bbot_func(x).detach().to(self.device)

        u, _ = self.cloned_actor(x, xref, uref, x_trim, xref_trim, deterministic=True)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

        u = u.detach()
        K = K.detach()

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.action_dim)
            ]
        )

        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
        dot_M = self.weighted_gradients(M, dot_x, x, True)

        ABK = A + matmul(B, K)
        MABK = matmul(M.detach(), ABK)
        sym_MABK = MABK + transpose(MABK, 1, 2)

        return {
            "dot_x_true": dot_x.detach(),
            "dot_M_true": dot_M.detach(),
            "ABK_true": ABK.detach(),
            "sym_MABK_true": sym_MABK.detach(),
            "Bbot_true": Bbot.detach(),
            "f_true": f.detach(),
            "B_true": B.detach(),
        }

    def B_null(self, x: torch.Tensor):
        n = x.shape[0]
        Bbot = torch.cat(
            (
                torch.eye(self.x_dim - self.action_dim, self.x_dim - self.action_dim),
                torch.zeros(self.action_dim, self.x_dim - self.action_dim),
            ),
            dim=0,
        )
        Bbot.unsqueeze(0).to(self.device)
        return Bbot.repeat(n, 1, 1)

    def Jacobian(self, f: torch.Tensor, x: torch.Tensor):
        # NOTE that this function assume that data are independent of each other
        f = f + 0.0 * x.sum()  # to avoid the case that f is independent of x

        n = x.shape[0]
        f_dim = f.shape[-1]
        x_dim = x.shape[-1]

        J = torch.zeros(n, f_dim, x_dim).to(self.device)
        for i in range(f_dim):
            J[:, i, :] = grad(f[:, i].sum(), x, create_graph=True)[0]  # [0]
        return J

    def Jacobian_Matrix(self, M: torch.Tensor, x: torch.Tensor):
        # NOTE that this function assume that data are independent of each other
        M = M + 0.0 * x.sum()  # to avoid the case that f is independent of x

        n = x.shape[0]
        matrix_dim = M.shape[-1]
        x_dim = x.shape[-1]

        J = torch.zeros(n, matrix_dim, matrix_dim, x_dim).to(self.device)
        for i in range(matrix_dim):
            for j in range(matrix_dim):
                J[:, i, j, :] = grad(M[:, i, j].sum(), x, create_graph=True)[0]

        return J

    def B_Jacobian(self, B: torch.Tensor, x: torch.Tensor):
        n = x.shape[0]
        x_dim = x.shape[-1]

        DBDx = torch.zeros(n, x_dim, x_dim, self.action_dim).to(self.device)
        for i in range(self.action_dim):
            DBDx[:, :, :, i] = self.Jacobian(B[:, :, i].unsqueeze(-1), x)
        return DBDx

    def weighted_gradients(
        self, W: torch.Tensor, v: torch.Tensor, x: torch.Tensor, detach: bool
    ):
        # v, x: bs x n x 1
        # DWDx: bs x n x n x n
        assert v.size() == x.size()
        bs = x.shape[0]
        if detach:
            return (self.Jacobian_Matrix(W, x).detach() * v.view(bs, 1, 1, -1)).sum(
                dim=3
            )
        else:
            return (self.Jacobian_Matrix(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)

    def compute_B_perp_batch(self, B, B_perp_dim, method="svd", threshold=1e-1):
        """
        Compute the nullspace basis B_perp for each sample (or a single matrix) from B,
        using either SVD or QR decomposition, and return a tensor of shape
        (batch, x_dim, B_perp_dim) or (x_dim, B_perp_dim) for single input.

        Parameters:
            B: Tensor of shape (batch, x_dim, x_dim) or (x_dim, x_dim).
            B_perp_dim: Desired number of nullspace vectors (output columns).
            method: "svd" or "qr".
            threshold: Threshold below which singular values or R diagonals are considered zero.

        Returns:
            B_perp_tensor: Tensor of shape (batch, x_dim, B_perp_dim) or (x_dim, B_perp_dim).
        """
        # Handle single matrix input by unsqueezing to batch size 1
        batch_size, x_dim, _ = B.shape
        B_perp_list = []

        for i in range(batch_size):
            B_i = B[i]

            if method.lower() == "svd":
                U, S, _ = torch.linalg.svd(B_i)
                null_indices = (S < threshold).nonzero(as_tuple=True)[0]
                B_perp = (
                    U[:, null_indices]
                    if null_indices.numel() > 0
                    else torch.empty(x_dim, 0, device=B.device, dtype=B.dtype)
                )

            elif method.lower() == "qr":
                Q, R = torch.linalg.qr(B_i, mode="complete")
                diag_R = torch.abs(torch.diag(R))
                null_indices = (diag_R < threshold).nonzero(as_tuple=True)[0]
                B_perp = (
                    Q[:, null_indices]
                    if null_indices.numel() > 0
                    else torch.empty(x_dim, 0, device=B.device, dtype=B.dtype)
                )
            else:
                raise ValueError("Method must be either 'svd' or 'qr'.")

            # Pad or truncate to fixed B_perp_dim
            padded = torch.zeros(x_dim, B_perp_dim, device=B.device, dtype=B.dtype)
            m = B_perp.shape[1]
            if m > 0:
                padded[:, : min(m, B_perp_dim)] = B_perp[:, :B_perp_dim]
            B_perp_list.append(padded)

        B_perp_tensor = torch.stack(B_perp_list, dim=0)  # (batch, x_dim, B_perp_dim)

        return B_perp_tensor

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values and counts for each key
        sum_dict = {}
        count_dict = {}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                if key not in sum_dict:
                    sum_dict[key] = 0
                    count_dict[key] = 0
                sum_dict[key] += value
                count_dict[key] += 1

        # Calculate the average for each key
        avg_dict = {key: sum_val / count_dict[key] for key, sum_val in sum_dict.items()}

        return avg_dict
