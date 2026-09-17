import matplotlib.pyplot as plt
import math

readings = [
    (0, 8190), (15, 8190), (30, 8190), (45, 8191), (60, 458), (75, 211),
    (90, 176), (105, 155), (120, 183), (135, 207), (150, 280), (165, 431),
    (180, 798), (195, 887), (210, 984), (225, 417), (240, 452), (255, 381),
    (270, 251), (285, 8190), (300, 8190), (315, 355), (330, 489), (345, 259),
]

MAX_VALID_MM = 2000  # treat near-8190 as "no target", not a real wall

xs, ys = [], []
for angle_deg, dist_mm in readings:
    if dist_mm >= MAX_VALID_MM:
        continue
    rad = math.radians(angle_deg)
    xs.append(dist_mm * math.cos(rad))
    ys.append(dist_mm * math.sin(rad))

plt.figure(figsize=(6, 6))
plt.plot(xs, ys, 'o-')
plt.plot(0, 0, 'rx', label='sensor')
plt.gca().set_aspect('equal')
plt.title("Room slice (top-down)")
plt.xlabel("mm"); plt.ylabel("mm")
plt.grid(True); plt.legend()
plt.show()
