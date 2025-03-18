import numpy as np
import gymnasium as gym
from gymnasium import spaces

# PVTOL PARAMETERS
p_lim = np.pi / 3
pd_lim = np.pi / 3
vx_lim = 2.0
vz_lim = 1.0

X_MIN = np.array([-35.0, -2.0, -p_lim, -vx_lim, -vz_lim, -pd_lim]).reshape(-1, 1)
X_MAX = np.array([0.0, 2.0, p_lim, vx_lim, vz_lim, pd_lim]).reshape(-1, 1)

m = 0.486
J = 0.00383
g = 9.81
l = 0.25

UREF_MIN = np.array([m * g / 2 - 1, m * g / 2 - 1]).reshape(-1, 1)
UREF_MAX = np.array([m * g / 2 + 1, m * g / 2 + 1]).reshape(-1, 1)

lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim, -lim, -lim, -lim]).reshape(-1, 1)
XE_MAX = np.array([lim, lim, lim, lim, lim, lim]).reshape(-1, 1)

# for sampling ref
X_INIT_MIN = np.array([0, 0, -0.1, 0.5, 0.0, 0.0])
X_INIT_MAX = np.array([0, 0, 0.1, 1.0, 0.0, 0.0])

XE_INIT_MIN = np.array(
    [
        -0.5,
    ]
    * 6
)
XE_INIT_MAX = np.array(
    [
        0.5,
    ]
    * 6
)

time_bound = 6.0
time_step = 0.03
t = np.arange(0, time_bound, time_step)

state_weights = np.array([1, 1, 0.1, 0.1, 0.1, 0.1])


class PvtolEnv(gym.Env):
    def __init__(self):
        super(PvtolEnv, self).__init__()
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """
        self.num_dim_x = 6
        self.num_dim_control = 2
        self.pos_dimension = 2

        self.reward_scaler = 1.0
        self.control_scaler = 1e-2

        self.time_bound = 6.0
        self.dt = 0.03
        self.episode_len = int(self.time_bound / self.dt)
        self.t = np.arange(0, self.time_bound, self.dt)

        self.observation_space = spaces.Box(
            low=X_MIN.flatten(), high=X_MAX.flatten(), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=UREF_MIN.flatten(), high=UREF_MAX.flatten(), dtype=np.float64
        )

        self.x_0, self.xref_0, self.xref, self.uref = self.system_reset(
            time_bound=self.time_bound, t=self.t
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

    def system_reset(self, time_bound, t):
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
        for _t in t:
            u = 0.5 * np.array([m * g, m * g])  # ref
            for freq, weight in zip(freqs, weights):
                u += np.array(
                    [
                        weight[0] * np.sin(freq * _t / time_bound * 2 * np.pi),
                        weight[0] * np.sin(freq * _t / time_bound * 2 * np.pi),
                    ]
                )
            f_x = self.f_func(xref[-1])
            B_x = self.b_func(xref[-1])

            xref.append(
                xref[-1] + self.dt * (f_x + np.matmul(B_x, u[:, np.newaxis]).squeeze())
            )
            uref.append(u)

        return x_0, xref_0, np.array(xref), np.array(uref)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_steps = 0
        self.x_t = self.x_0.copy()
        self.state = self.x_t - self.xref[self.time_steps]
        return self.state, {"x": self.x_t}

    def step(self, action):
        self.time_steps += 1

        f_x = self.f_func(self.x_t)
        B_x = self.b_func(self.x_t)

        self.x_t = self.x_t + self.dt * (
            f_x + np.matmul(B_x, action[:, np.newaxis]).squeeze()
        )
        noise = np.random.normal(loc=0.0, scale=0.03, size=self.num_dim_x)
        self.x_t += noise
        self.x_t = np.clip(self.x_t, X_MIN.flatten(), X_MAX.flatten())
        self.state = self.x_t - self.xref[self.time_steps]

        tracking_error = np.linalg.norm(self.state, ord=2)
        control_effort = np.linalg.norm(action, ord=2)

        reward = self.reward_scaler * (
            1 / (tracking_error + 1)
        ) + self.control_scaler * (1 / (control_effort + 1))
        termination = False
        truncation = self.time_steps == self.episode_len

        return (
            self.state.squeeze(),
            reward,
            termination,
            truncation,
            {
                "x": self.x_t,
                "tracking_error": tracking_error,
                "control_effort": control_effort,
            },
        )

    def render(self, mode="human"):
        pass
