import math
import os
import random

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

LENGTH = 5
MAX_T = 30
DELTA_T = 0.1
OUTPUT_DIR = "./output"

matplotlib.use("Agg")

def figure_to_frame(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[..., :3].copy()

def write_video(frames):
    if not frames:
        print("No frames to write.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"pendulum_dt{DELTA_T}.mp4")
    fps = 1.0 / DELTA_T
    imageio.mimsave(out_path, frames, fps=fps, codec="libx264", quality=8)
    print(f"Wrote {out_path} ({len(frames)} frames @ {fps:.1f} fps)")

def render_pendulum(theta):

    y, x = LENGTH * math.cos(theta), LENGTH * math.sin(theta)

    fig, ax = plt.subplots()
    ax.plot([0, x], [0, -y], lw=10, color='blue', solid_capstyle='round')
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

def calculate_trajectory(curr_theta, curr_vel):

    theta_second_order = curr_theta + DELTA_T/2 * curr_vel
    vel_second_order = curr_vel - DELTA_T/2 * math.sin(curr_theta)

    theta_third_order = curr_theta + DELTA_T/2 * vel_second_order
    vel_third_order = curr_vel - DELTA_T/2 * math.sin(theta_second_order)

    theta_fourth_order = curr_theta + DELTA_T * vel_third_order
    vel_fourth_order = curr_vel - DELTA_T * math.sin(theta_third_order)
    
    k1_theta = curr_vel
    k2_theta = vel_second_order
    k3_theta = vel_third_order
    k4_theta = vel_fourth_order

    k1_vel = -1 * math.sin(curr_theta)
    k2_vel = -1 * math.sin(theta_second_order)
    k3_vel = -1 * math.sin(theta_third_order)
    k4_vel = -1 * math.sin(theta_fourth_order)

    next_theta = curr_theta + DELTA_T/6 * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
    next_vel = curr_vel + DELTA_T/6 * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)

    return next_theta, next_vel

def main():
    print("Hello from cse-291g-physical-simulation!")

    curr_theta = math.pi / 180 * random.randint(-90, 90)
    curr_vel = 0

    frames = []
    for step in range(0, math.ceil(MAX_T/DELTA_T) + 1):
        fig = render_pendulum(curr_theta)
        frames.append(figure_to_frame(fig))
        plt.close(fig)

        curr_theta, curr_vel = calculate_trajectory(curr_theta, curr_vel)

    write_video(frames)

if __name__ == "__main__":
    main()
