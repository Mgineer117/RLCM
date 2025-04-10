import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
import math
import numpy as np
import time

class MDPDataCollector(Node):
    def __init__(self, total_steps=200, save_path='collected_data.npz'):
        super().__init__('mdp_data_collector')

        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE, history=QoSHistoryPolicy.KEEP_LAST)

        # Parameters
        self.total_steps = total_steps
        self.save_path = save_path

        # ROS 2 interfaces
        self.subscription = self.create_subscription(
            PoseStamped,
            'qualysis/tb3_3',  # your motion capture topic
            self.pose_callback,
            qos)

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

        # Data storage
        self.current_pose = None
        self.state_history = []
        self.action_history = []
        self.next_state_history = []

        # Control setup
        self.step_count = 0
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        self.previous_state = None  # For saving previous state

        # Sinusoidal angular velocity parameters
        time.sleep(5)
        self.start_time = time.time()
        self.angular_amplitude = np.random.uniform(0.3, 1.0)  # Random amplitude
        self.angular_frequency = np.random.uniform(0.2, 1.0)  # Random frequency (rad/s)

        self.get_logger().info(f"Sinusoidal angular velocity: amplitude={self.angular_amplitude:.3f}, frequency={self.angular_frequency:.3f}")

    def B_func(self, x, y, theta):
        B = np.zeros((3, 2))
        B[0, 0] = 0.8912 * np.cos(theta)
        B[1, 1] = 0.9055 * np.sin(theta)
        B[2, 1] = 0.7953
        return B

    def pose_callback(self, msg):
        self.current_pose = msg

    def control_loop(self):
        if self.current_pose is None:
            return  # wait for first pose update

        # Extract current state
        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y
        theta = self.current_pose.pose.orientation.z % (2 * math.pi)
        current_state = [x, y, theta]
    
        # Sinusoidal angular velocity
        linear_velocity, angular_velocity = self.get_motion()

        # If we have a previous state, save (state, action, next_state)
        if self.previous_state is not None and hasattr(self, 'previous_action'):
            self.state_history.append(self.previous_state)
            self.action_history.append(self.previous_action)
            self.next_state_history.append(current_state)

            # B = self.B_func(x, y, theta)
            # a = np.array([[linear_velocity, angular_velocity]]).T
            # # print(current_state)
            # x_dot = (np.array(current_state) - np.array(self.previous_state)) / 0.1
            # x_dot_approx = (B@a).T
            # print(x_dot-x_dot_approx)
            # print(angular_velocity)
        
        # Publish velocity command
        vel_msg = Twist()
        vel_msg.linear.x = linear_velocity
        vel_msg.angular.z = angular_velocity
        self.publisher_.publish(vel_msg)

        # Save current action for next step
        self.previous_action = [linear_velocity, angular_velocity]
        self.previous_state = current_state

        self.step_count += 1

        if self.step_count >= self.total_steps:
            self.finish_data_collection()

    def get_motion(self, pattern='constant_linear_sin_angular'):
        elapsed_time = time.time() - self.start_time

        if pattern == 'constant_linear_sin_angular':
            # 🚗 Constant speed, but turning left and right smoothly
            linear_velocity = 0.15
            angular_velocity = 0.4 * math.sin(0.5 * elapsed_time)

        elif pattern == 'changing_linear_sin_angular':
            # 🚗 Speed up and slow down smoothly, while also turning
            linear_velocity = 0.1 + 0.05 * math.sin(0.3 * elapsed_time)  # speed oscillates between 0.05 and 0.15
            angular_velocity = 0.4 * math.sin(0.5 * elapsed_time)

        elif pattern == 'constant_linear_step_angular':
            # 🚗 Constant speed, but angular velocity changes in a square wave (like zigzag)
            linear_velocity = 0.15
            angular_velocity = 0.4 * np.sign(math.sin(0.5 * elapsed_time))

        elif pattern == 'accelerating_spiral_out':
            # 🚗 Linear velocity slowly increases, angular velocity decreases
            linear_velocity = 0.1 + 0.01 * elapsed_time  # slowly accelerating
            angular_velocity = 0.5 / (1 + elapsed_time)  # decreasing turn, spiral out

        elif pattern == 'decelerating_spiral_in':
            # 🚗 Linear velocity slowly decreases, angular velocity increases
            linear_velocity = max(0.05, 0.15 - 0.01 * elapsed_time)
            angular_velocity = 0.1 + 0.05 * elapsed_time  # increasing turn, spiral in

        else:
            # Default fallback: straight line
            linear_velocity = 0.1
            angular_velocity = 0.0

        return linear_velocity, angular_velocity

    
    def finish_data_collection(self):
        self.get_logger().info("Finished data collection, stopping robot.")

        # Stop the robot
        stop_msg = Twist()
        self.publisher_.publish(stop_msg)

        # Prepare data
        states = np.array(self.state_history)
        actions = np.array(self.action_history)
        next_states = np.array(self.next_state_history)

        # Concatenate: [state, action, next_state]
        data = {
            'state': states,             # shape (N, 3): [x, y, theta]
            'action': actions,           # shape (N, 2): [linear_vel, angular_vel]
            'next_state': next_states    # shape (N, 3): [next_x, next_y, next_theta]
        }

        # Save to .npz
        np.savez(self.save_path, **data)
        self.get_logger().info(f"Data saved to {self.save_path}")
        rclpy.shutdown()
    
def main(args=None):
    rclpy.init(args=args)
    node = MDPDataCollector(total_steps=200, save_path='collected_data.npz')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
