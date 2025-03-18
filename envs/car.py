import numpy as np
import gymnasium as gym
from gymnasium import spaces

# CAR PARAMETERS
v_l = 1.0
v_h = 2.0

X_MIN = np.array([-5.0, -5.0, -np.pi, v_l]).reshape(-1, 1)
X_MAX = np.array([5.0, 5.0, np.pi, v_h]).reshape(-1, 1)

XE_MIN = np.array([-1, -1, -1, -1]).reshape(-1, 1)
XE_MAX = np.array([1, 1, 1, 1]).reshape(-1, 1)

UREF_MIN = np.array([-3.0, -3.0]).reshape(-1, 1)
UREF_MAX = np.array([3.0, 3.0]).reshape(-1, 1)

X_INIT_MIN = np.array([-2.0, -2.0, -1.0, 1.5])
X_INIT_MAX = np.array([2.0, 2.0, 1.0, 1.5])

XE_INIT_MIN = np.full((4,), -1.0)
XE_INIT_MAX = np.full((4,), 1.0)


class CarEnv(gym.Env):
    def __init__(self):
        super(CarEnv, self).__init__()
        """
        State: tracking error between current and reference trajectory
        Reward: The 2-norm of tracking error
        """
        self.num_dim_x = 4  # x, y, theta, v
        self.num_dim_control = 2  # u1 (angular acc), u2 (linear acc)
        self.pos_dimension = 2

        self.reward_scaler = 1.0
        self.control_scaler = 1e-1

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
        f = np.zeros((self.num_dim_x,))
        f[0] = x[3] * np.cos(x[2])
        f[1] = x[3] * np.sin(x[2])
        return f

    def b_func(self, x):
        B = np.zeros((self.num_dim_x, self.num_dim_control))
        B[2, 0] = 1
        B[3, 1] = 1
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
        weights = weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))

        xref = [xref_0]
        uref = []
        for _t in t:
            u = np.array([0.0, 0])
            for freq, weight in zip(freqs, weights):
                u += np.array(
                    [weight[0] * np.sin(freq * _t / time_bound * 2 * np.pi), 0]
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
        self.state = self.x_0.copy()
        return self.x_0, {}

    def step(self, action):
        self.time_steps += 1

        f_x = self.f_func(self.state)
        B_x = self.b_func(self.state)

        self.state = self.state + self.dt * (
            f_x + np.matmul(B_x, action[:, np.newaxis]).squeeze()
        )
        noise = np.random.normal(loc=0.0, scale=0.03, size=self.num_dim_x)
        self.state += noise
        self.state = np.clip(self.state, X_MIN.flatten(), X_MAX.flatten())

        tracking_error = np.linalg.norm(self.xref[self.time_steps] - self.state, ord=2)
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
            {"tracking_error": tracking_error, "control_effort": control_effort},
        )

    def render(self, mode="human"):
        pass
