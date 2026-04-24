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

pos_limits = {
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "z": (-0.5, 0.0)
}

# -----------------------------
# TWO LEGS
# -----------------------------
legs = {
    "LEFT":  [0.0, 0.0, -0.3],
    "RIGHT": [0.0, 0.0, -0.3]
}

defaults = {k: v[:] for k, v in legs.items()}

# -----------------------------
# ELBOW MODE
# -----------------------------
elbow_up = False

# -----------------------------
# BODY (BOX)
# -----------------------------
box_length = 0.3
box_width  = 0.2
box_height = 0.2

def get_box():
    l = box_length/2
    w = box_width/2
    h = box_height

    return np.array([
        [ l,  w, 0],
        [ l, -w, 0],
        [-l, -w, 0],
        [-l,  w, 0],
        [ l,  w, -h],
        [ l, -w, -h],
        [-l, -w, -h],
        [-l,  w, -h]
    ])

# -----------------------------
# LEG BASE POSITIONS
# -----------------------------
base_pos = {
    "LEFT":  np.array([0.0,  box_width/2, -box_height]),
    "RIGHT": np.array([0.0, -box_width/2, -box_height])
}

# -----------------------------
# INVERSE KINEMATICS
# -----------------------------
def ik(x, y, z):
    global elbow_up

    t1 = np.arctan2(y, -z)

    R = np.sqrt(y**2 + z**2)
    R = max(R, 1e-6)

    D = x**2 + R**2
    cos_t3 = (D - L1**2 - L2**2) / (2 * L1 * L2)
    cos_t3 = np.clip(cos_t3, -1.0, 1.0)

    if elbow_up:
        t3 = -np.arccos(cos_t3)
    else:
        t3 = np.arccos(cos_t3)

    alpha = np.arctan2(x, R)
    beta  = np.arctan2(L2*np.sin(t3), L1 + L2*np.cos(t3))

    t2 = alpha - beta

    return t1, t2, t3

# -----------------------------
# FORWARD KINEMATICS
# -----------------------------
def fk(t1, t2, t3):
    X = L1*np.sin(t2) + L2*np.sin(t2+t3)
    R = L1*np.cos(t2) + L2*np.cos(t2+t3)
    Y = R*np.sin(t1)
    Z = -R*np.cos(t1)
    return np.array([X, Y, Z])

# -----------------------------
# PLOT
# -----------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

def draw():
    ax.clear()

    # Draw body
    b = get_box()
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for e in edges:
        ax.plot(
            [b[e[0],0], b[e[1],0]],
            [b[e[0],1], b[e[1],1]],
            [b[e[0],2], b[e[1],2]],
            linewidth=3
        )

    angles_dict = {}

    # Draw legs
    for leg, (x,y,z) in legs.items():
        base = base_pos[leg]

        t1, t2, t3 = ik(x, y, z)
        angles_dict[leg] = np.degrees([t1, t2, t3])

        knee = base + np.array([
            L1*np.sin(t2),
            L1*np.cos(t2)*np.sin(t1),
            -L1*np.cos(t2)*np.cos(t1)
        ])

        foot = base + fk(t1, t2, t3)

        ax.plot([base[0], knee[0]], [base[1], knee[1]], [base[2], knee[2]], marker='o')
        ax.plot([knee[0], foot[0]], [knee[1], foot[1]], [knee[2], foot[2]], marker='o')

    # Info box
    info = "           θ1     θ2     θ3\n"
    for leg in ["LEFT", "RIGHT"]:
        t1, t2, t3 = angles_dict[leg]
        info += f"{leg}: {t1:6.1f} {t2:6.1f} {t3:6.1f}\n"

    ax.text2D(
        0.02, 0.98,
        info,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    ax.set_xlim([-0.5,0.5])
    ax.set_ylim([-0.5,0.5])
    ax.set_zlim([-0.6,0.2])

# -----------------------------
# TKINTER UI
# -----------------------------
root = tk.Tk()
root.title("Biped IK Simulator")
root.geometry("1100x650")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

control = tk.Frame(root)
control.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

# -----------------------------
# UPDATE
# -----------------------------
def update(leg, i, val):
    legs[leg][i] = float(val)
    draw()
    canvas.draw_idle()

sliders = []

# -----------------------------
# SLIDERS
# -----------------------------
for leg in legs:
    f = tk.LabelFrame(control, text=leg)
    f.pack(fill="x", pady=6)

    for i, name in enumerate(["X","Y","Z"]):
        s = tk.Scale(
            f,
            from_=pos_limits[name.lower()][0],
            to=pos_limits[name.lower()][1],
            resolution=0.01,
            orient=tk.HORIZONTAL,
            label=name,
            command=lambda v, l=leg, j=i: update(l,j,v)
        )
        s.set(legs[leg][i])
        s.pack(side="left", expand=True, fill="x")

        sliders.append((s, leg, i))

# -----------------------------
# RESET
# -----------------------------
def reset():
    for l in legs:
        legs[l] = defaults[l][:]
    for s, l, i in sliders:
        s.set(legs[l][i])
    draw()
    canvas.draw()

tk.Button(control, text="Reset", command=reset).pack(fill="x", pady=5)

# -----------------------------
# ELBOW TOGGLE
# -----------------------------
def toggle_elbow():
    global elbow_up
    elbow_up = not elbow_up
    btn.config(text="Elbow: UP" if elbow_up else "Elbow: DOWN")
    draw()
    canvas.draw()

btn = tk.Button(control, text="Elbow: DOWN", command=toggle_elbow)
btn.pack(fill="x", pady=5)

# -----------------------------
draw()
canvas.draw()
root.mainloop()