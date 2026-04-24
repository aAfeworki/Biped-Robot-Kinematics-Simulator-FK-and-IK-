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

limits = {
    "theta1": (-np.pi/6, np.pi/6),
    "theta2": (-np.pi/2, np.pi/6),
    "theta3": (0, np.pi)
}

# -----------------------------
# TWO LEGS
# -----------------------------
legs = {
    "LEFT": [0.0, -0.5, 1.0],
    "RIGHT": [0.0, -0.5, 1.0]
}

defaults = {k: v[:] for k, v in legs.items()}

# -----------------------------
# BOX BODY (3D)
# -----------------------------
box_length = 0.2
box_width  = 0.2
box_height = 0.2

# 8 corners of the box
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
# LEG BASE POSITIONS (bottom of box)
# -----------------------------
base_pos = {
    "LEFT":  np.array([0.0,  box_width/2, -box_height]),
    "RIGHT": np.array([0.0, -box_width/2, -box_height])
}

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

    # ----- DRAW BOX -----
    b = get_box()

    edges = [
        (0,1),(1,2),(2,3),(3,0),  # top
        (4,5),(5,6),(6,7),(7,4),  # bottom
        (0,4),(1,5),(2,6),(3,7)   # verticals
    ]

    for e in edges:
        ax.plot(
            [b[e[0],0], b[e[1],0]],
            [b[e[0],1], b[e[1],1]],
            [b[e[0],2], b[e[1],2]],
            linewidth=3
        )

    # ----- DRAW LEGS -----
    for leg, (t1,t2,t3) in legs.items():
        base = base_pos[leg]

        knee = base + np.array([
            L1*np.sin(t2),
            L1*np.cos(t2)*np.sin(t1),
            -L1*np.cos(t2)*np.cos(t1)
        ])

        foot = base + fk(t1,t2,t3)

        ax.plot([base[0], knee[0]], [base[1], knee[1]], [base[2], knee[2]], marker='o')
        ax.plot([knee[0], foot[0]], [knee[1], foot[1]], [knee[2], foot[2]], marker='o')

    ax.set_xlim([-0.4,0.4])
    ax.set_ylim([-0.4,0.4])
    ax.set_zlim([-0.4,0.4])

# -----------------------------
# TKINTER UI
# -----------------------------
root = tk.Tk()
root.title("Biped FK Simulator")
root.geometry("1100x650")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

control = tk.Frame(root)
control.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

# -----------------------------
# Update
# -----------------------------
def update(leg, i, val):
    legs[leg][i] = float(val)
    draw()
    canvas.draw_idle()

# -----------------------------
# Store sliders
# -----------------------------
sliders = []

# -----------------------------
# Sliders 
# -----------------------------
for leg in legs:
    f = tk.LabelFrame(control, text=leg)
    f.pack(fill="x", pady=6)

    for i, name in enumerate(["θ1","θ2","θ3"]):
        s = tk.Scale(
            f,
            from_=limits[f"theta{i+1}"][0],
            to=limits[f"theta{i+1}"][1],
            resolution=0.01,
            orient=tk.HORIZONTAL,
            label=name,
            command=lambda v, l=leg, j=i: update(l,j,v)
        )
        s.set(legs[leg][i])
        s.pack(side="left", expand=True, fill="x")

        sliders.append((s, leg, i))

# -----------------------------
# Reset
# -----------------------------
def reset():
    for l in legs:
        legs[l] = defaults[l][:]
    for s, l, i in sliders:
        s.set(legs[l][i])
    draw()
    canvas.draw()

tk.Button(control, text="Reset", command=reset).pack(fill="x", pady=10)

# -----------------------------
draw()
canvas.draw()
root.mainloop()