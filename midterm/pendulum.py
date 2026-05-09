import math
import os

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

LENGTH_1 = 5 # meters
LENGTH_2 = 3 # meters
WIDTH_1 = 1 # meters
WIDTH_2 = 1.5 # meters
MASS_1 = 1 # kg
MASS_2 = 2 # kg

MAX_T = 30
### Smaller delta T resulted in stability. System was unstable for 0.1 and 0.01
DELTA_T = 0.001
OUTPUT_DIR = "./output"
GRAVITY = 9.8 # Gravity of 9.8 m/s^2

SUBSAMPLE = 1

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

def rod_corners(pivot_x, pivot_y, theta, length, width):
    along = np.array([math.sin(theta), -math.cos(theta)])
    across = np.array([math.cos(theta),  math.sin(theta)])
    pivot = np.array([pivot_x, pivot_y])
    tip = pivot + length * along
    half_w = width / 2
    c1 = pivot - half_w * across
    c2 = pivot + half_w * across
    c3 = tip + half_w * across
    c4 = tip - half_w * across
    return np.array([c1, c2, c3, c4])

def render_pendulum(ax, state):

    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    theta1, theta2, _, _ = state

    origin_y2 = - LENGTH_1 * math.cos(theta1)
    origin_x2 = LENGTH_1 * math.sin(theta1)

    corners1 = rod_corners(0, 0, theta1, LENGTH_1, WIDTH_1)
    corners2 = rod_corners(origin_x2, origin_y2, theta2, LENGTH_2, WIDTH_2)

    ax.add_patch(plt.Polygon(corners1, color='steelblue', alpha=0.8))
    ax.add_patch(plt.Polygon(corners2, color='coral', alpha=0.8))

def state_diff(state):
    theta1, theta2, ang_vel1, ang_vel2 = state
    diff_theta1_theta2 = theta1 - theta2

    i_1 = (MASS_1 * LENGTH_1**2)/3 + (MASS_1 * WIDTH_1**2)/12
    i_2 = (MASS_2 * LENGTH_2**2)/3 + (MASS_2 * WIDTH_2**2)/12

    a_1 = i_1 + MASS_2 * LENGTH_1**2
    a_2 = i_2

    b = (MASS_2 * LENGTH_1 * LENGTH_2)/2

    c_1 = (MASS_1/2 + MASS_2) * GRAVITY * LENGTH_1
    c_2 = (MASS_2 * GRAVITY * LENGTH_2)/2

    m_11 = a_1
    m_12 = m_21 = b * math.cos(diff_theta1_theta2)
    m_22 = a_2

    m_q = np.array([[m_11, m_12], [m_21, m_22]])
    c_q_qdot = np.array([b * math.sin(diff_theta1_theta2) * ang_vel2 ** 2, -b * math.sin(diff_theta1_theta2) * ang_vel1 ** 2])
    g_q = np.array([c_1 * math.sin(theta1), c_2 * math.sin(theta2)])

    ang_acc_1, ang_acc_2 = np.linalg.solve(m_q, -c_q_qdot - g_q)

    return np.array([ang_vel1, ang_vel2, ang_acc_1, ang_acc_2])


def rk_4(curr_state):

    k_1 = state_diff(curr_state)
    k_2 = state_diff(curr_state + DELTA_T/2 * k_1)
    k_3 = state_diff(curr_state + DELTA_T/2 * k_2)
    k_4 = state_diff(curr_state + DELTA_T * k_3)

    return curr_state + DELTA_T/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
    

def main():
    curr_state = np.array([math.pi * 2 / 3, math.pi / 2, 0.0, 0.0])

    fig, ax = plt.subplots()

    frames = []
    for step in range(0, math.ceil(MAX_T/DELTA_T) + 1):

        if (0 == step % SUBSAMPLE):
            render_pendulum(ax, curr_state)
            frames.append(figure_to_frame(fig))
            plt.close(fig)

        curr_state = rk_4(curr_state)

    write_video(frames)

if __name__ == "__main__":
    main()
