import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import math
import time
import torch
import matplotlib.pyplot as plt

from policy.layers.c3m_networks import C3M_U
from policy.layers.ppo_networks import PPO_Actor

def get_actor(algo_name):
    if algo_name == "c3m":
        policy = C3M_U(x_dim=3,
                    state_dim=8,
                    effective_indices=np.array([0,1,2]),
                    action_dim=2,
                    task="turtlebot")
    elif algo_name in ("ppo", "mrl"):
        policy = PPO_Actor(input_dim = 8, hidden_dim = [64, 64], a_dim=2)

    policy.load_state_dict(torch.load(f"model/{algo_name}.pth")) 
    return policy.to(torch.float64).cpu()


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE, history=QoSHistoryPolicy.KEEP_LAST)
        robot_name = 'qualysis/tb3_3'
        self.subscription = self.create_subscription(
            PoseStamped,
            robot_name,  # or whatever your marker_deck_name is
            self.pose_callback,
            qos)
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.control_loop)  
        self.current_pose = None

        ### Algorithmic and Environmental parameters ###
        algo_name = "ppo"
        self.effective_indices = np.array([0,1,2])
        self.episode_len = 200

        policy = get_actor(algo_name)
        self.policy = policy
        
        ### Load the entire policy object ###
        self.state = []
        self.action = []
        self.ref_state = np.load('ref.npz')["state"]
        self.ref_action = np.load('ref.npz')["action"]

        self.internal_counter = 0

        time.sleep(5) # wait for the initialization of robots
        
    def pose_callback(self, msg):
        self.current_pose = msg

    def draw_plot(self):
        # Assuming self.state and self.ref_state are lists of [x, y] positions
        states = np.array(self.state)
        ref_states = np.array(self.ref_state)

        plt.figure(figsize=(8, 6), dpi=150)

        # Plot trajectories as lines
        plt.plot(states[:, 0], states[:, 1], color='blue', linestyle='-', linewidth=2, label='Trajectory')
        plt.plot(ref_states[:, 0], ref_states[:, 1], color='red', linestyle='--', linewidth=2, label='Reference Trajectory')

        # Mark start and end points
        plt.scatter(states[0, 0], states[0, 1], color='green', marker='o', s=100, label='Start')
        plt.scatter(states[-1, 0], states[-1, 1], color='black', marker='x', s=100, label='End')

        # Optionally, mark start and end of reference
        plt.scatter(ref_states[0, 0], ref_states[0, 1], color='orange', marker='o', s=100, label='Ref Start')
        plt.scatter(ref_states[-1, 0], ref_states[-1, 1], color='purple', marker='x', s=100, label='Ref End')

        # Add grid, legend, labels
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Trajectory Tracking')
        plt.tight_layout()

        # Save figure
        plt.savefig("traj.png", bbox_inches='tight')
        plt.close()

        data = {"state": np.array(self.state), "action":np.array(self.action)}
        np.savez('tracking_data.npz', data)
        
        self.state = []
        self.action = []

    def to_tensor(self, x: np.ndarray):
        return torch.from_numpy(x).to(torch.float64).unsqueeze(0)

    def get_action(self, x, xref, uref, x_trim, x_ref_trim):
        with torch.no_grad():
            try:
                a, _= self.policy(x, xref, uref, x_trim, x_ref_trim)            
            except:
                a = self.policy(x, xref, uref, x_trim, x_ref_trim)       
        a = a.cpu().numpy().squeeze() if a.shape[-1] > 1 else [a.item()]
        a += self.ref_action[self.internal_counter]
        a = np.clip(a, [0, -1.82], [0.22, 1.82])
        return a

    def control_loop(self):
        if self.current_pose is None:
            return
        pos_x = self.current_pose.pose.position.x
        pos_y = self.current_pose.pose.position.y
        yaw = self.current_pose.pose.orientation.z  # assuming yaw was stored in `.z`
        yaw = yaw % (2 * math.pi)

        x = self.to_tensor(np.array([pos_x, pos_y, yaw]))
        xref = self.to_tensor(self.ref_state[self.internal_counter])
        uref = self.to_tensor(self.ref_action[self.internal_counter])
        x_trim = self.to_tensor(x[self.effective_indices])
        x_ref_trim = self.to_tensor(xref[self.effective_indices])

        # Turn toward the goal
        a = self.get_action(x, xref, uref, x_trim, x_ref_trim)
        
        self.state.append(np.array([pos_x, pos_y, yaw]))
        self.action.append(a)
        print(f"x: {x.numpy()}, xref: {xref.numpy()} with a: {a}")
        
        # translate to the ros2 command
        vel_msg = Twist()
        vel_msg.linear.x = a[0]
        vel_msg.angular.z = a[1]
        self.publisher_.publish(vel_msg)

        self.internal_counter += 1

        if self.internal_counter >= (self.episode_len - 1):
            self.draw_plot()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()