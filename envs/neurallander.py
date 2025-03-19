import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# NEURAL-LANDER PARAMETERS
rho = 1.225
drone_height = 0.09
g = 9.81
mass = 1.47

X_MIN = np.array([-5.0, -5.0, 0.0, -1.0, -1.0, -1.0]).reshape(-1, 1)
X_MAX = np.array([5.0, 5.0, 2.0, 1.0, 1.0, 1.0]).reshape(-1, 1)

lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim, -lim, -lim, -lim]).reshape(-1, 1)
XE_MAX = np.array([lim, lim, lim, lim, lim, lim]).reshape(-1, 1)

UREF_MIN = np.array([-1.0, -1.0, -3.0]).reshape(-1, 1)
UREF_MAX = np.array([1.0, 1.0, 9.0]).reshape(-1, 1)

# for sampling ref
X_INIT_MIN = np.array([-3.0, -3.0, 0.5, 1.0, 0.0, 0.0])
X_INIT_MAX = np.array([3.0, 3.0, 1.0, 1.0, 0.0, 0.0])

XE_INIT_MIN = np.array([-1, -1, -0.4, -1.0, -1.0, 0.0])
XE_INIT_MAX = np.array([1, 1.0, 1.0, 1.0, 1.0, 0.0])


state_weights = np.array([1, 1, 1, 0.1, 0.1, 0.1])

STATE_MIN = np.concatenate((X_MIN.flatten(), X_MIN.flatten(), UREF_MIN.flatten()))
STATE_MAX = np.concatenate((X_MAX.flatten(), X_MAX.flatten(), UREF_MAX.flatten()))


# NEURAL-LANDER FUNCTIONS
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(12, 25)
        self.fc2 = nn.Linear(25, 30)
        self.fc3 = nn.Linear(30, 15)
        self.fc4 = nn.Linear(15, 3)

    def forward(self, x):
        if not x.is_cuda:
            self.cpu()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


def read_weight(filename):
    model_weight = torch.load(filename, map_location=torch.device("cpu"))
    model = Network().double()
    model.load_state_dict(model_weight)
    model = model.float()
    # .cuda()
    return model


Fa_model = read_weight("data/Fa_net_12_3_full_Lip16.pth")


def Fa_func(z, vx, vy, vz):
    if next(Fa_model.parameters()).device != z.device:
        Fa_model.to(z.device)
    # use prediction from NN as ground truth
    state = torch.zeros((12,))
    state[0] = z + drone_height
    state[1] = vx  # velocity
    state[2] = vy  # velocity
    state[3] = vz  # velocity
    state[7] = 1.0
    state[8:12] = 6508.0 / 8000

    with torch.no_grad():
        Fa = Fa_model(state) * torch.tensor([30.0, 15.0, 10.0])

    return Fa


def Fa_func_np(x):
    z = torch.tensor(x[2]).float().view(1, -1)
    vx = torch.tensor(x[3]).float().view(1, -1)
    vy = torch.tensor(x[4]).float().view(1, -1)
    vz = torch.tensor(x[5]).float().view(1, -1)
    Fa = Fa_func(z, vx, vy, vz).numpy()
    return Fa


class NeuralLanderEnv(gym.Env):
    def __init__(self, sigma: float = 0.0):
        super(NeuralLanderEnv, self).__init__()
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """
        self.num_dim_x = 6
        self.num_dim_control = 3
        self.pos_dimension = 3

        self.tracking_scaler = 1.0
        self.control_scaler = 1e-1

        self.time_bound = 3.0
        self.dt = 0.03
        self.episode_len = int(self.time_bound / self.dt)
        self.t = np.arange(0, self.time_bound, self.dt)

        self.state_weights = state_weights
        self.sigma = sigma
        self.d_up = 3 * sigma

        self.observation_space = spaces.Box(
            low=STATE_MIN.flatten(), high=STATE_MAX.flatten(), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=UREF_MIN.flatten(), high=UREF_MAX.flatten(), dtype=np.float64
        )

    def f_func(self, x):
        # x: bs x n x 1
        # f: bs x n x 1
        x, y, z, vx, vy, vz = [x[i] for i in range(self.num_dim_x)]
        f = np.zeros((self.num_dim_x,))
        f[0] = vx
        f[1] = vy
        f[2] = vz

        Fa = Fa_func(z, vx, vy, vz)

        f[3] = Fa[0] / mass
        f[4] = Fa[1] / mass
        f[5] = Fa[2] / mass - g

        return f

    def b_func(self, x):
        B = np.zeros((self.num_dim_x, self.num_dim_control))

        B[3, 0] = 1
        B[4, 1] = 1
        B[5, 2] = 1
        return B

    def system_reset(self):
        # with temp_seed(int(seed)):
        xref_0 = X_INIT_MIN + np.random.rand(len(X_INIT_MIN)) * (
            X_INIT_MAX - X_INIT_MIN
        )
        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (
            XE_INIT_MAX - XE_INIT_MIN
        )
        x_0 = xref_0 + xe_0
        Fa = Fa_func_np(xref_0.reshape(-1)).reshape(-1)

        freqs = list(range(1, 11))
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (
            0.5 * weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))
        ).tolist()

        xref = [xref_0]
        uref = []
        for i, _t in enumerate(self.t):
            u = np.array([0, 0, g]) - Fa / mass  # ref
            for freq, weight in zip(freqs, weights):
                u += np.array(
                    [
                        weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                        weight[1] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                        weight[2] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                    ]
                )
            u = np.clip(u, 0.75 * UREF_MIN.flatten(), 0.75 * UREF_MAX.flatten())

            x_t = xref[-1].copy()

            f_x = self.f_func(x_t)
            B_x = self.b_func(x_t)

            x_t = x_t + self.dt * (f_x + np.matmul(B_x, u[:, np.newaxis]).squeeze())

            termination = np.any(
                x_t[: self.pos_dimension] <= X_MIN.flatten()[: self.pos_dimension]
            ) or np.any(
                x_t[: self.pos_dimension] >= X_MAX.flatten()[: self.pos_dimension]
            )

            x_t = np.clip(x_t, X_MIN.flatten(), X_MAX.flatten())
            xref.append(x_t)
            uref.append(u)

            init_tracking_error = np.linalg.norm(x_0 - xref_0, ord=2)

            if termination:
                break

        return x_0, np.array(xref), np.array(uref), init_tracking_error, i

    def dynamic_fn(self, action):
        self.time_steps += 1

        f_x = self.f_func(self.x_t)
        B_x = self.b_func(self.x_t)

        self.x_t = self.x_t + self.dt * (
            f_x + np.matmul(B_x, action[:, np.newaxis]).squeeze()
        )

        noise = np.random.normal(loc=0.0, scale=self.sigma, size=self.num_dim_x)
        noise[self.pos_dimension :] = 0.0
        noise = np.clip(noise, -self.d_up, self.d_up)

        self.x_t += noise
        termination = np.any(
            self.x_t[: self.pos_dimension] <= X_MIN.flatten()[: self.pos_dimension]
        ) or np.any(
            self.x_t[: self.pos_dimension] >= X_MAX.flatten()[: self.pos_dimension]
        )
        self.x_t = np.clip(self.x_t, X_MIN.flatten(), X_MAX.flatten())
        self.state = np.concatenate(
            (self.x_t, self.xref[self.time_steps], self.uref[self.time_steps])
        )

        return termination

    def reward_fn(self, action):
        error = self.x_t - self.xref[self.time_steps]

        tracking_error = np.linalg.norm(
            self.state_weights * error,
            ord=2,
        )
        control_effort = np.linalg.norm(action, ord=2)

        reward = self.tracking_scaler / (tracking_error + 1) + self.control_scaler / (
            control_effort + 1
        )

        return reward, {
            "tracking_error": tracking_error,
            "control_effort": control_effort,
        }

    def reset(self, seed=None, options: dict | None = None):
        super().reset(seed=seed)
        self.time_steps = 0

        if options is None:
            (
                self.x_0,
                self.xref,
                self.uref,
                self.init_tracking_error,
                self.episode_len,
            ) = self.system_reset()
        else:
            if options.get("replace_x_0", True):
                xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (
                    XE_INIT_MAX - XE_INIT_MIN
                )
                x_0 = self.xref[0] + xe_0
                self.x_0 = x_0

        self.x_t = self.x_0.copy()
        self.state = np.concatenate(
            (self.x_t, self.xref[self.time_steps], self.uref[self.time_steps])
        )

        return self.state, {"x": self.x_t}

    def step(self, action):
        # policy output ranges [-1, 1]
        action = self.uref[self.time_steps] + action
        action = np.clip(action, UREF_MIN.flatten(), UREF_MAX.flatten())

        termination = self.dynamic_fn(action)
        reward, infos = self.reward_fn(action)

        truncation = self.time_steps == self.episode_len

        return (
            self.state,
            reward,
            termination,
            truncation,
            {
                "x": self.x_t,
                "tracking_error": infos["tracking_error"],
                "control_effort": infos["control_effort"],
                "relative_tracking_error": infos["tracking_error"]
                / self.init_tracking_error,
            },
        )

    def render(self, mode="human"):
        pass
