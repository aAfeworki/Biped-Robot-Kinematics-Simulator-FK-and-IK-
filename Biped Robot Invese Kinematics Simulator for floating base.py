import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# Robot parameters
# -----------------------------
L1, L2 = 0.2, 0.2

# Base pose (x, y, z, roll, pitch, yaw)
base_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
base_defaults = base_pose[:]

# -----------------------------
# BIPED: FIXED FEET
# -----------------------------
feet_world = {
    "LEFT":  np.array([0.0,  0.2, -0.3]),
    "RIGHT": np.array([0.0, -0.2, -0.3])
}

# Hip offsets in body frame
hip_offsets = {
    "LEFT":  np.array([0.0,  0.2, 0.0]),
    "RIGHT": np.array([0.0, -0.2, 0.0])
}

# -----------------------------
# BODY 
# -----------------------------
l, w, h = 0.2, 0.2, 0.2

def get_box():
    return np.array([
        [ l,  w, 0],
        [ l, -w, 0],
        [-l, -w, 0],
        [-l,  w, 0],
        [ l,  w, h],
        [ l, -w, h],
        [-l, -w, h],
        [-l,  w, h]
    ])

# -----------------------------
# Rotation matrix (ZYX)
# -----------------------------
def rot_matrix(roll, pitch, yaw):
    Rx = np.array([
        [1,0,0],
        [0,np.cos(roll), -np.sin(roll)],
        [0,np.sin(roll),  np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch),0,np.sin(pitch)],
        [0,1,0],
        [-np.sin(pitch),0,np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw),0],
        [np.sin(yaw),  np.cos(yaw),0],
        [0,0,1]
    ])
    return Rz @ Ry @ Rx

# -----------------------------
# IK 
# -----------------------------
def ik(x, y, z):
    t1 = np.arctan2(y, -z)

    R = np.sqrt(y**2 + z**2)
    if R < 1e-6:
        R = 1e-6

    D = x**2 + R**2
    cos_t3 = (D - L1**2 - L2**2) / (2 * L1 * L2)
    cos_t3 = np.clip(cos_t3, -1.0, 1.0)

    t3 = np.arccos(cos_t3)

    alpha = np.arctan2(x, R)
    beta  = np.arctan2(L2*np.sin(t3), L1 + L2*np.cos(t3))

    t2 = alpha - beta

    return t1, t2, t3

# -----------------------------
# FK 
# -----------------------------
def fk(t1, t2, t3):
    X = L1*np.sin(t2) + L2*np.sin(t2+t3)
    R = L1*np.cos(t2) + L2*np.cos(t2+t3)
    Y = R*np.sin(t1)
    Z = -R*np.cos(t1)
    return np.array([X, Y, Z])

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

def draw():
    ax.clear()

    bx, by, bz, roll, pitch, yaw = base_pose
    Rb = rot_matrix(roll, pitch, yaw)
    base = np.array([bx, by, bz])

    b = get_box()
    b_world = (Rb @ b.T).T + base

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # top
        (4, 5), (5, 6), (6, 7), (7, 4),  # bottom
        (0, 4), (1, 5), (2, 6), (3, 7)  # verticals
    ]

    for e in edges:
        ax.plot(
            [b_world[e[0], 0], b_world[e[1], 0]],
            [b_world[e[0], 1], b_world[e[1], 1]],
            [b_world[e[0], 2], b_world[e[1], 2]],
            linewidth=3
        )

    angles_dict = {}

    for leg in feet_world:
        foot_w = feet_world[leg]
        hip_w = base + Rb @ hip_offsets[leg]

        # TRANSFORM 
        foot_local = Rb.T @ (foot_w - hip_w)

        t1, t2, t3 = ik(*foot_local)
        angles_dict[leg] = np.degrees([t1, t2, t3])

        knee = hip_w + Rb @ np.array([
            L1*np.sin(t2),
            L1*np.cos(t2)*np.sin(t1),
            -L1*np.cos(t2)*np.cos(t1)
        ])

        foot_calc = hip_w + Rb @ fk(t1,t2,t3)

        ax.plot([hip_w[0], knee[0]], [hip_w[1], knee[1]], [hip_w[2], knee[2]], marker='o')
        ax.plot([knee[0], foot_calc[0]], [knee[1], foot_calc[1]], [knee[2], foot_calc[2]], marker='o')

        ax.scatter(foot_w[0], foot_w[1], foot_w[2], s=30)

    # INFO BOX 
    info_text = "            θ1     θ2     θ3\n"
    for leg in ["LEFT","RIGHT"]:
        t1, t2, t3 = angles_dict[leg]
        info_text += f"{leg}: {t1:6.1f} {t2:6.1f} {t3:6.1f}\n"

    ax.text2D(
        0.02, 0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    ax.set_xlim([-0.6,0.6])
    ax.set_ylim([-0.6,0.6])
    ax.set_zlim([-0.6,0.4])

# -----------------------------
# TKINTER UI 
# -----------------------------
root = tk.Tk()
root.title("Biped Floating Base IK")
root.geometry("1100x650")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

control = tk.Frame(root)
control.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

def update(i, val):
    base_pose[i] = float(val)
    draw()
    canvas.draw_idle()

labels = ["X","Y","Z","Roll","Pitch","Yaw"]
limits = [
    (-0.27,0.27),
    (-0.27,0.27),
    (-0.3,0.1),
    (-0.54,0.54),
    (-np.pi/4,np.pi/4),
    (-1.52,1.52)
]

sliders = []

frame = tk.LabelFrame(control, text="Base Control")
frame.pack(fill="x", pady=6)

for i, name in enumerate(labels):
    s = tk.Scale(
        frame,
        from_=limits[i][0],
        to=limits[i][1],
        resolution=0.01,
        orient=tk.HORIZONTAL,
        label=name,
        command=lambda v, j=i: update(j,v)
    )
    s.set(base_pose[i])
    s.pack(fill="x")

    sliders.append((s, i))

def reset():
    for i in range(len(base_pose)):
        base_pose[i] = base_defaults[i]
    for s, i in sliders:
        s.set(base_pose[i])
    draw()
    canvas.draw()

tk.Button(control, text="Reset", command=reset).pack(fill="x", pady=10)

draw()
canvas.draw()
root.mainloop()