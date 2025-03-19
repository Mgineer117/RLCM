import numpy as np
import gymnasium as gym
from gymnasium import spaces

# PVTOL PARAMETERS
p_lim = np.pi / 3
pd_lim = np.pi / 3
vx_lim = 2.0
vz_lim = 1.0

X_MIN = np.array([-3.0, 0.0, -p_lim, -vx_lim, -vz_lim, -pd_lim]).reshape(-1, 1)
X_MAX = np.array([3.0, 6.0, p_lim, vx_lim, vz_lim, pd_lim]).reshape(-1, 1)

m = 0.486
J = 0.00383
g = 9.81
l = 0.25

lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim, -lim, -lim, -lim]).reshape(-1, 1)
XE_MAX = np.array([lim, lim, lim, lim, lim, lim]).reshape(-1, 1)

# for sampling ref
X_INIT_MIN = np.array([-1, 1.0, -0.1, 0.5, 0.0, 0.0])
X_INIT_MAX = np.array([1, 2.0, 0.1, 1.0, 0.0, 0.0])

XE_INIT_MIN = np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
XE_INIT_MAX = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

UREF_MIN = np.array([m * g / 2 - 1, m * g / 2 - 1]).reshape(-1, 1)
UREF_MAX = np.array([m * g / 2 + 1, m * g / 2 + 1]).reshape(-1, 1)

STATE_MIN = np.concatenate((X_MIN.flatten(), UREF_MIN.flatten()))
STATE_MAX = np.concatenate((X_MAX.flatten(), UREF_MAX.flatten()))

# position: 1.0, orientation: 0.25, velocity: 0.15
w = np.array([1.0, 1.0, 0.15, 0.05, 0.05, 0.05])  # relative importance


class PvtolEnv(gym.Env):
    def __init__(self, sigma: float = 0.0):
        super(PvtolEnv, self).__init__()
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """
        self.num_dim_x = 6
        self.num_dim_control = 2
        self.pos_dimension = 2

        self.tracking_scaler = 1.0
        self.control_scaler = 1e-1

        self.time_bound = 6.0
        self.dt = 0.03
        self.episode_len = int(self.time_bound / self.dt)
        self.t = np.arange(0, self.time_bound, self.dt)

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
        p_x, p_z, phi, v_x, v_z, dot_phi = [x[i] for i in range(self.num_dim_x)]
        f = np.zeros((self.num_dim_x,))
        f[0] = v_x * np.cos(phi) - v_z * np.sin(phi)
        f[1] = v_x * np.sin(phi) + v_z * np.cos(phi)
        f[2] = dot_phi
        f[3] = v_z * dot_phi - g * np.sin(phi)
        f[4] = -v_x * dot_phi - g * np.cos(phi)
        f[5] = 0
        return f

    def b_func(self, x):
        B = np.zeros((self.num_dim_x, self.num_dim_control))

        B[4, 0] = 1 / m
        B[4, 1] = 1 / m
        B[5, 0] = l / J
        B[5, 1] = -l / J
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

        freqs = list(range(1, 11))
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (
            0.1 * weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))
        ).tolist()

        xref = [xref_0]
        uref = []
        for i, _t in enumerate(self.t):
            u = 0.5 * np.array([m * g, m * g])  # ref
            for freq, weight in zip(freqs, weights):
                u += np.array(
                    [
                        weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                        weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                    ]
                )
            u = np.clip(u, UREF_MIN.flatten(), UREF_MAX.flatten())

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

            if termination:
                break

        return x_0, np.array(xref), np.array(uref), i

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
            (self.x_t - self.xref[self.time_steps], self.uref[self.time_steps])
        )

        return termination

    def reward_fn(self, action):
        tracking_error = np.linalg.norm(
            w * (self.x_t - self.xref[self.time_steps]),
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
            self.x_0, self.xref, self.uref, self.episode_len = self.system_reset()
        else:
            if options.get("replace_x_0", True):
                xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (
                    XE_INIT_MAX - XE_INIT_MIN
                )
                x_0 = self.xref[0] + xe_0
                self.x_0 = x_0

        self.x_t = self.x_0.copy()
        self.state = np.concatenate(
            (self.x_t - self.xref[self.time_steps], self.uref[self.time_steps])
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
            },
        )

    def render(self, mode="human"):
        pass
