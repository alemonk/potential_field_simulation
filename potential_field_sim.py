# Python simulation of potential field obstacle avoidance (pure Python)
#
# This script simulates a differential drive robot using a simple potential field
# controller that combines an attractive force toward the goal with a repulsive
# contribution from the closest obstacle. It also draws a static force grid
# (quiver) to visualize the combined force field across the workspace.
#
# Key design notes:
# - The controller uses only the single closest obstacle to compute repulsion.
# - Attractive force is a unit vector toward the target.
# - Repulsive magnitude falls off linearly with distance inside MAX_OBST_DIST.
# - Wheel speeds are derived from a desired forward speed and angular rate.
# - SMOOTH_ALPHA smooths wheel speed transitions to avoid abrupt changes.

import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import random
import numpy as np

# -------------------------------------------------------------
# Global parameters and constants
# -------------------------------------------------------------

# Simulation timings
DT = 0.1                  # time step in seconds
TOTAL_TIME = 1000.0       # total simulation time if not using target stop

# Robot geometry
WHEEL_BASE = 0.72         # meters, distance between left and right wheels

# Potential field constants
MAX_OBST_DIST = 3.0       # obstacles farther than this are ignored
MIN_OBST_DIST = 0.5       # if robot gets closer than this to an obstacle, stop or strong repulsion

# Desired velocity ranges
MIN_LIN_MPS = 0.75        # minimum commanded linear speed (m/s)
MAX_LIN_MPS = 1.0         # maximum allowed linear speed (m/s)
MAX_ANG_RADS = 1.0        # maximum allowed angular velocity (rad/s)

# Force combination gains
K_ANG_ATTRACT = 1.0       # weight for attractive force
K_REP_ANG = 3.0           # weight for repulsive force

# Frontal cone and smoothing parameters
FRONTAL_COS_CUTOFF = 0.0  # frontal check threshold used in controller (cosine of angle)
SMOOTH_ALPHA = 0.2        # smoothing factor for wheel command transitions (0..1)

# Initial robot state
INIT_X = 0.0
INIT_Y = 0.0
INIT_YAW = 0

# Environment: goal location
TARGET = (0.0, 100.0)

# Obstacle generation parameters
X_RANGE = (-10.0, 10.0)
Y_RANGE = (10.0, 90.0)
NUM_CLUSTERS = 15         # number of obstacle clusters
POINTS_PER_CLUSTER = 30   # points per cluster, controls density
STD_DEV = 0.5             # cluster spread (standard deviation)

# -------------------- Utility functions --------------------

def compute_total_force_field(x, y, obstacles, target_x, target_y):
    """
    Compute the combined force vector (attractive + repulsive) at an arbitrary
    world coordinate (x, y). This uses the same single-closest-obstacle rule
    as the controller so the visualization matches behavior.
    Returns a 2-element numpy array [fx, fy].
    """
    EPS = 1e-6

    # Attractive force: unit vector pointing from (x, y) to the target
    F_att = np.array([target_x - x, target_y - y], dtype=float)
    dist_to_goal = np.linalg.norm(F_att)
    F_att = F_att / (dist_to_goal + EPS)

    # Repulsive force: only the closest obstacle within MAX_OBST_DIST contributes
    closest_obs = None
    closest_dist = MAX_OBST_DIST

    for ox, oy in obstacles:
        d = math.hypot(ox - x, oy - y)
        if d < closest_dist:
            closest_dist = d
            closest_obs = (ox, oy)

    F_rep = np.zeros(2, dtype=float)
    if closest_obs is not None:
        ox, oy = closest_obs
        d_vec = np.array([x - ox, y - oy], dtype=float)  # vector from obstacle to query point
        d = np.linalg.norm(d_vec)

        # If inside a minimum safe distance, treat repulsion as zero to avoid NaNs.
        # The controller handles this case differently; this helper keeps things safe.
        if d < MIN_OBST_DIST:
            F_rep = np.array([0.0, 0.0])
        elif d < MAX_OBST_DIST:
            # linear falloff for repulsion magnitude
            rep_mag = (MAX_OBST_DIST - d) / MAX_OBST_DIST
            rep_mag = clampf(rep_mag, 0.0, 1.0)
            F_rep = d_vec / (d + EPS) * rep_mag
        else:
            # obstacle too far, no repulsion
            F_rep = np.zeros(2, dtype=float)

    # Combined force
    F_total = K_ANG_ATTRACT * F_att + K_REP_ANG * F_rep
    return F_total

def generate_clustered_obstacles(num_clusters, points_per_cluster, map_x_range, map_y_range, cluster_std_dev):
    """
    Create clustered obstacles by sampling points around random cluster centers.
    Returns a list of (x, y) tuples.
    """
    all_obstacles = []
    min_x, max_x = map_x_range
    min_y, max_y = map_y_range

    for _ in range(num_clusters):
        # Random cluster center inside the defined ranges
        center_x = random.uniform(min_x, max_x)
        center_y = random.uniform(min_y, max_y)

        # Sample Gaussian points around the center
        for _ in range(points_per_cluster):
            obstacle_x = np.random.normal(loc=center_x, scale=cluster_std_dev)
            obstacle_y = np.random.normal(loc=center_y, scale=cluster_std_dev)
            all_obstacles.append((obstacle_x, obstacle_y))

    return all_obstacles

# Generate obstacle set used by both controller and visualization
OBSTACLES = generate_clustered_obstacles(
    num_clusters=NUM_CLUSTERS,
    points_per_cluster=POINTS_PER_CLUSTER,
    map_x_range=X_RANGE,
    map_y_range=Y_RANGE,
    cluster_std_dev=STD_DEV
)

def clampf(v, lo, hi):
    """Clamp v to the closed interval [lo, hi]."""
    return max(lo, min(hi, v))

def wrapToPi(a):
    """
    Normalize angle to the interval [-pi, pi].
    Uses careful incremental subtraction/addition to avoid precision edge cases.
    """
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

# -------------------- Potential field avoidance controller --------------------

def computePotentialFieldAvoidance(
    robot_x, robot_y, robot_yaw,
    target_x, target_y,
    obstacles,
    cur_left_speed, cur_right_speed,
    wheel_base_m
):
    """
    Compute wheel commands for a differential drive robot based on a combined
    potential field. Returns (v_left, v_right).

    Steps:
    0) find the closest obstacle (within MAX_OBST_DIST)
    1) compute a unit attractive vector toward the goal
    2) compute repulsive vector from the closest obstacle using robot frame checks
    3) combine forces and derive desired heading and forward speed
    4) convert desired linear and angular velocities to wheel speeds
    5) smooth wheel outputs using exponential smoothing
    """

    # 0. determine which obstacle is relevant (closest within MAX_OBST_DIST)
    closest_obs = None
    closest_dist = MAX_OBST_DIST
    for ox, oy in obstacles:
        dx = ox - robot_x
        dy = oy - robot_y
        d = math.hypot(dx, dy)
        if d < closest_dist:
            closest_dist = d
            closest_obs = (ox, oy)

    EPS = 1e-6

    # ------------------- Attractive force -------------------
    # Unit vector pointing from robot to goal
    F_att = np.array([target_x - robot_x, target_y - robot_y], dtype=float)
    dist_to_goal = np.linalg.norm(F_att)
    F_att = F_att / (dist_to_goal + EPS)

    # ------------------- Repulsive force (closest obstacle only) -------------------
    F_rep = np.zeros(2, dtype=float)

    if closest_obs is not None:
        obs_x, obs_y = closest_obs

        # Vector from obstacle to robot in world frame, and its norm
        d_vec = np.array([robot_x - obs_x, robot_y - obs_y], dtype=float)
        d = np.linalg.norm(d_vec)

        # If robot is too close, return zero wheel speeds to avoid collision
        if d < MIN_OBST_DIST:
            return 0.0, 0.0

        # If within repulsion zone, compute repulsive contribution using a frontal check
        if d > EPS and d < MAX_OBST_DIST:
            # Express obstacle position relative to robot in robot frame:
            # xr is forward distance, yr is lateral distance (positive left)
            dx = obs_x - robot_x   # obstacle pos relative to robot in world frame
            dy = obs_y - robot_y
            cy = math.cos(robot_yaw)
            sy = math.sin(robot_yaw)
            xr = cy * dx + sy * dy
            yr = -sy * dx + cy * dy
            obs_dist = math.hypot(xr, yr)

            # Frontal check: only consider obstacles in front and within frontal cone
            if obs_dist > EPS and xr > 0 and (math.cos(math.atan2(yr, xr)) >= FRONTAL_COS_CUTOFF):
                # repulsion magnitude decreases with squared distance here
                rep_mag = (MAX_OBST_DIST - obs_dist**2) / MAX_OBST_DIST
                rep_mag = clampf(rep_mag, 0.0, 1.0)

                # Build repulsion vector in robot frame, pointing away from obstacle
                F_rep_robot = np.array([xr, yr], dtype=float)
                # invert and normalize, then scale with repulsion magnitude
                F_rep_robot = -F_rep_robot / (obs_dist + EPS) * rep_mag

                # Rotate repulsion back to world frame using robot yaw
                F_rep = np.array([
                    cy * F_rep_robot[0] - sy * F_rep_robot[1],
                    sy * F_rep_robot[0] + cy * F_rep_robot[1]
                ], dtype=float)

    # ------------------- Combine forces and compute motion commands -------------------
    F_total = K_ANG_ATTRACT * F_att + K_REP_ANG * F_rep

    # Desired heading and its magnitude
    desired_heading = math.atan2(F_total[1], F_total[0])
    force_magnitude = np.linalg.norm(F_total)

    # Scale forward speed with force magnitude while respecting a minimum speed
    desired_v = MIN_LIN_MPS + (1.0 - MIN_LIN_MPS) * min(force_magnitude, 1.0)

    # Desired angular velocity is proportional to heading error
    desired_omega = wrapToPi(desired_heading - robot_yaw)
    desired_omega = clampf(desired_omega, -MAX_ANG_RADS, MAX_ANG_RADS)

    # Penalize forward speed when a sharp turn is required
    ang_penalty = min(1.0, abs(desired_omega) / MAX_ANG_RADS)
    desired_v *= (1.0 - 0.5 * ang_penalty)

    # Convert linear and angular velocities to left and right wheel speeds
    v_left  = desired_v - desired_omega * (wheel_base_m * 0.5)
    v_right = desired_v + desired_omega * (wheel_base_m * 0.5)

    # Enforce wheel speed limits
    v_left  = clampf(v_left, -MAX_LIN_MPS, MAX_LIN_MPS)
    v_right = clampf(v_right, -MAX_LIN_MPS, MAX_LIN_MPS)

    # Smooth transition to new wheel commands to avoid abrupt jumps
    v_left  = SMOOTH_ALPHA * v_left  + (1.0 - SMOOTH_ALPHA) * cur_left_speed
    v_right = SMOOTH_ALPHA * v_right + (1.0 - SMOOTH_ALPHA) * cur_right_speed
    v_left  = clampf(v_left, -MAX_LIN_MPS, MAX_LIN_MPS)
    v_right = clampf(v_right, -MAX_LIN_MPS, MAX_LIN_MPS)

    return v_left, v_right

# -------------------- Robot simulator --------------------

class DifferentialDriveRobot:
    """
    Simple kinematic simulator for a differential drive robot.
    Stores pose and wheel speeds and integrates pose using a first-order Euler step.
    """
    def __init__(self, x, y, yaw, wheel_base):
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)
        self.wheel_base = float(wheel_base)
        self.vL = 0.0
        self.vR = 0.0

    def step(self, target, obstacles, dt):
        """
        Compute new wheel speeds using the controller and integrate the robot pose.
        Returns the updated pose and wheel speeds.
        """
        outL, outR = computePotentialFieldAvoidance(
            self.x, self.y, self.yaw,
            target[0], target[1],
            obstacles,
            self.vL, self.vR,
            self.wheel_base
        )
        # Update wheel speeds
        self.vL = outL
        self.vR = outR

        # Compute body linear velocity and angular rate from wheel speeds
        v = 0.5 * (self.vL + self.vR)
        omega = (self.vR - self.vL) / self.wheel_base

        # Integrate pose with simple Euler step
        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        self.yaw = wrapToPi(self.yaw + omega * dt)

        return (self.x, self.y, self.yaw, self.vL, self.vR)

# -------------------- Simulation parameters and run --------------------

steps = int(TOTAL_TIME / DT)

# Create robot instance with initial pose
robot = DifferentialDriveRobot(INIT_X, INIT_Y, INIT_YAW, WHEEL_BASE)

# Run simulation and record history in a list of dicts for DataFrame conversion
history = []
for i in range(steps):
    t = i * DT
    x, y, yaw, vL, vR = robot.step(TARGET, OBSTACLES, DT)
    history.append({
        "t": t, "x": x, "y": y, "yaw": yaw, "vL": vL, "vR": vR,
        "target_x": TARGET[0], "target_y": TARGET[1]
    })

# Convert history to pandas DataFrame for plotting and analysis
df = pd.DataFrame(history)

# -------------------- Plotting: trajectory, force grid, wheel speeds and yaw --------------------

# Prepare figure and axes for main animation
fig, ax = plt.subplots(figsize=(8, 8))

# Draw obstacles as black dots; useful visual reference
ax.scatter(
    [o[0] for o in OBSTACLES],
    [o[1] for o in OBSTACLES],
    s=50,
    marker="o",
    color="black",
    label="obstacles"
)

# Draw target as a green star
ax.scatter([TARGET[0]], [TARGET[1]], s=120, marker="*", color="green", label="target")

# Create an empty line that will be updated with the robot path in the animation
path_line, = ax.plot([], [], linewidth=1, color="red")

# Robot marker that moves along the path
robot_marker, = ax.plot([], [], marker=".", markersize=4, color="red", label="robot")

# Configure axes limits and aspect
ax.set_xlim(-30, 30)
ax.set_ylim(-5, 105)
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.legend()
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Potential Field Obstacle Avoidance - Animation")

# -------------------- Force grid (static) --------------------
# Create a grid of points across the workspace and compute the combined force at each point.
# The quiver overlay visualizes the vector field that the robot controller uses.
grid_x = np.linspace(-10, 10, 20)    # coarser in x for readability
grid_y = np.linspace(0, 100, 100)    # finer in y to follow the corridor to target

GX, GY = np.meshgrid(grid_x, grid_y)
U = np.zeros_like(GX)
V = np.zeros_like(GY)

# Populate U and V with force vector components computed by helper
for i in range(GX.shape[0]):
    for j in range(GX.shape[1]):
        fx, fy = compute_total_force_field(GX[i, j], GY[i, j], OBSTACLES, TARGET[0], TARGET[1])
        U[i, j] = fx
        V[i, j] = fy

# Draw the quiver for the force field. scale controls arrow length scaling.
ax.quiver(GX, GY, U, V, color="blue", alpha=0.2, scale=30)

# -------------------- Animation update function --------------------
def update(frame):
    """
    Animation callback that updates path and robot marker up to the given frame index.
    The quiver grid is static so it is not updated here.
    """
    xs = df["x"][:frame]
    ys = df["y"][:frame]
    path_line.set_data(xs, ys)

    # Set robot marker to the current robot pose
    robot_marker.set_data(df["x"][:frame+1], df["y"][:frame+1])
    return path_line, robot_marker

# Create animation. We step through the DataFrame in larger increments for speed.
anim = FuncAnimation(fig, update, frames=range(0, len(df), 10), interval=40, blit=True)

# Display the animation
plt.show()

# -------------------- Time series plots: wheel speeds and yaw --------------------
fig, axes = plt.subplots(2, 1, figsize=(10,6), sharex=True)

# Wheel speeds subplot
axes[0].plot(df["t"], df["vL"], label="vL (left)", linewidth=1.5)
axes[0].plot(df["t"], df["vR"], label="vR (right)", linewidth=1.5)
axes[0].set_ylabel("wheel speed [m/s]")
axes[0].grid(True)
axes[0].legend(loc="upper right")
axes[0].set_title("Wheel speeds")

# Heading (yaw) subplot
axes[1].plot(df["t"], df["yaw"], label="yaw", linewidth=1.5)
axes[1].set_ylabel("yaw [rad]")
axes[1].set_xlabel("time [s]")
axes[1].grid(True)
axes[1].legend(loc="upper right")
axes[1].set_title("Heading (yaw)")

plt.tight_layout()
plt.show()
