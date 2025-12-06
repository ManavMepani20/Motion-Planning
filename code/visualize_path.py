#!/usr/bin/env python3
# =======================================================================================
# visualize_path.py
# Visualizes the solution path, obstacles, and uncertainty ellipses for the CCRRT planner.
#
# Features:
# - Reads static and dynamic obstacles from file (obstacles.txt)
# - Reads planned path (solution_path.txt), RRT tree (rrt_tree.txt), and covariances (covariances.txt)
# - Animates the robot's path, showing uncertainty as an ellipse at each step
# - Animates dynamic obstacles using their trajectory (obstacle_trajectory.txt)
# - Supports saving the animation as a video (MP4) or displaying interactively
#
# File formats:
# - Obstacles: x, y, w, h[, dynamic]
# - Path: x, y, theta, v (theta/v may be unused)
# - Covariances: 2x2 matrix per line (flattened)
# - RRT tree: x1 y1 t1 x2 y2 t2 per edge
# - Obstacle trajectory: t obs_id x y r per line
#
# This script helps visualize how the CCRRT algorithm handles static/dynamic obstacles
# and propagates uncertainty (chance constraints) along the path.
# =======================================================================================

import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, FFMpegWriter

def read_obstacles(file_path):
    static_obstacles = []
    dynamic_obstacles = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            try:
                # Static circle: x, y, r, "circle"
                if parts[3].strip() == "circle":
                    x = float(parts[0])
                    y = float(parts[1])
                    r = float(parts[2])
                    static_obstacles.append(('circle', x, y, r))
                # Dynamic rectangle: x, y, w, h dynamic
                elif "dynamic" in parts[3]:
                    x = float(parts[0])
                    y = float(parts[1])
                    w = float(parts[2])
                    h = float(parts[3].split()[0])
                    dynamic_obstacles.append(('rect', x, y, w, h))
                # Fallback: treat as rectangle (legacy)
                else:
                    x = float(parts[0])
                    y = float(parts[1])
                    w = float(parts[2])
                    h = float(parts[3])
                    static_obstacles.append(('rect', x, y, w, h))
            except ValueError:
                continue
    return static_obstacles, dynamic_obstacles

def read_path(file_path):
    path = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, t, v = map(float, parts[:4])
            theta = 0.0  # Always set orientation to 0
            path.append((x, y, theta, v))
    return path

def read_covariances(file_path):
    covariances = []
    with open(file_path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 4:
                covariances.append(((vals[0], vals[1]), (vals[2], vals[3])))
    return covariances

def read_tree(file_path):
    segs = []
    with open(file_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                x1, y1, x2, y2 = map(float, parts)
            elif len(parts) == 6:
                x1, y1, _, x2, y2, _ = map(float, parts)
            else:
                continue
            segs.append(((x1, y1), (x2, y2)))
    return segs

def read_obstacle_trajectory(file_path):
    trajectory = {}
    with open(file_path, 'r') as f:
        for line in f:
            t, obs_id, x, y, r = map(float, line.strip().split())
            if obs_id not in trajectory:
                trajectory[obs_id] = []
            trajectory[obs_id].append((t, x, y, r))
    return trajectory

def animate_car_path(obstacles, path, covariances, trajectory_file=None, save_video=False):
    fig, ax = plt.subplots()
    
    # Separate static and dynamic obstacles
    static_obstacles, dynamic_obstacles = obstacles
    print(f"[DEBUG] Found {len(static_obstacles)} static and {len(dynamic_obstacles)} dynamic obstacles")
    
    # Load dynamic obstacle trajectories
    dynamic_trajectories = {}
    if trajectory_file and os.path.exists(trajectory_file):
        dynamic_trajectories = read_obstacle_trajectory(trajectory_file)
        print(f"[INFO] Loaded trajectory data from {trajectory_file}")
        for obs_id, traj in dynamic_trajectories.items():
            print(f"  Obstacle {obs_id}: {len(traj)} points")
    else:
        print(f"[WARN] No trajectory file found at '{trajectory_file}'")

    # Draw static obstacles
    static_patches = []
    for obst in static_obstacles:
        if obst[0] == 'circle':
            _, x, y, r = obst
            circ = patches.Circle((x, y), r, linewidth=1, edgecolor='black', facecolor='gray', alpha=0.7)
            static_patches.append(circ)
            ax.add_patch(circ)
        elif obst[0] == 'rect':
            _, x, y, w, h = obst
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor='gray', alpha=0.7)
            static_patches.append(rect)
            ax.add_patch(rect)

    # Initialize dynamic obstacles (rectangles)
    dynamic_patches = []
    for i, obst in enumerate(dynamic_obstacles):
        if obst[0] == 'rect':
            _, x, y, w, h = obst
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='red', facecolor='pink')
            dynamic_patches.append(rect)
            ax.add_patch(rect)

    # Draw RRT tree
    for (x1, y1), (x2, y2) in read_tree("rrt_tree.txt"):
        ax.plot([x1, x2], [y1, y2], color='lightgray', linewidth=0.5, zorder=0)

    # Plot full path
    xs = [state[0] for state in path]
    ys = [state[1] for state in path]
    ax.plot(xs, ys, color='blue', label='Path')

    # Start and goal
    if path:
        ax.plot(xs[0], ys[0], marker='s', color='green', markersize=10, label='Start')
        ax.plot(xs[-1], ys[-1], marker='*', color='red', markersize=10, label='Goal')

    ax.set_aspect('equal', 'box')

    # --- FIX: Set fixed axis limits so visualization does not scale with goal ---
    # Compute global min/max for all path and obstacles

    # Gather all x, y, w, h for both circle and rect obstacles
    all_x = xs[:]
    all_y = ys[:]
    all_w = []
    all_h = []

    for obst in static_obstacles:
        if obst[0] == 'circle':
            _, x, y, r = obst
            all_x.append(x)
            all_y.append(y)
            all_w.append(r)
            all_h.append(r)
        elif obst[0] == 'rect':
            _, x, y, w, h = obst
            all_x.append(x)
            all_y.append(y)
            all_w.append(w)
            all_h.append(h)
    for obst in dynamic_obstacles:
        if obst[0] == 'rect':
            _, x, y, w, h = obst
            all_x.append(x)
            all_y.append(y)
            all_w.append(w)
            all_h.append(h)

    min_x = min(all_x) - max(all_w, default=0) - 1
    max_x = max(all_x) + max(all_w, default=0) + 1
    min_y = min(all_y) - max(all_h, default=0) - 1
    max_y = max(all_y) + max(all_h, default=0) + 1

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    # --- END FIX ---

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Animated Car Path')
    ax.legend(loc='upper right')

    # Animation elements
    path_line, = ax.plot([], [], color='red', linewidth=2)

    car_length = 0.5
    car_width = 0.3
    car_patch = patches.Rectangle((0, 0), car_length, car_width, angle=0.0, color='red', zorder=5)
    ax.add_patch(car_patch)

    cov_ellipse = Ellipse((0, 0), 0, 0, edgecolor='orange', facecolor='none', lw=0.8, alpha=0.6)
    ax.add_patch(cov_ellipse)

    def init():
        path_line.set_data([], [])
        return (path_line, car_patch, cov_ellipse, 
                *static_patches, *dynamic_patches)

    def update(frame):
        x, y, theta, _ = path[frame]
        path_line.set_data(xs[:frame + 1], ys[:frame + 1])

        # Update car position
        dx = -car_length / 2
        dy = -car_width / 2
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        offset = rot @ np.array([dx, dy])
        car_patch.set_xy((x + offset[0], y + offset[1]))
        car_patch.angle = np.degrees(theta)

        # Update covariance ellipse
        cov = np.array(covariances[frame])
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * 1.96 * np.sqrt(vals)
        cov_ellipse.set_center((x, y))
        cov_ellipse.width = width
        cov_ellipse.height = height
        cov_ellipse.angle = angle

        # Update dynamic obstacles
        current_time = frame * 0.1  # Keep original timing
        for obs_id, traj in dynamic_trajectories.items():
            if int(obs_id) < len(dynamic_patches):  # Make sure obstacle exists
                # Find closest trajectory points
                for i in range(len(traj)-1):
                    if traj[i][0] <= current_time <= traj[i+1][0]:
                        t1, x1, y1, r1 = traj[i]
                        t2, x2, y2, r2 = traj[i+1]
                        
                        # Scale the displacement by 1/10th
                        center_x = (x1 + x2) / 2  # Center point
                        center_y = (y1 + y2) / 2
                        x1_scaled = center_x + (x1 - center_x) / 10
                        x2_scaled = center_x + (x2 - center_x) / 10
                        y1_scaled = center_y + (y1 - center_y) / 10
                        y2_scaled = center_y + (y2 - center_y) / 10
                        
                        # Linear interpolation with scaled coordinates
                        alpha = (current_time - t1) / (t2 - t1)
                        x = x1_scaled + alpha * (x2_scaled - x1_scaled)
                        y = y1_scaled + alpha * (y2_scaled - y1_scaled)
                        w = h = 2 * r1  # Keep original size
                        
                        # Update obstacle position
                        dynamic_patches[int(obs_id)].set_xy((x - w/2, y - h/2))
                        dynamic_patches[int(obs_id)].set_width(w)
                        dynamic_patches[int(obs_id)].set_height(h)
                        break

        return (path_line, car_patch, cov_ellipse, 
                *static_patches, *dynamic_patches)

    ani = FuncAnimation(fig, update, frames=len(path),
                        init_func=init, blit=True, interval=50, repeat=False)

    if save_video:
        print("[INFO] Saving animation to car_path_animation.mp4 ...")
        ani.save("car_path_animation.mp4", writer=FFMpegWriter(fps=20))
        print("[DONE] MP4 animation saved.")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Animate and save car path with uncertainty.")
    parser.add_argument("--pathfile", type=str, required=True, help="Path file with x y theta v per line")
    parser.add_argument("--obstacles", type=str, required=True, help="Obstacle file in x,y,w,h format per line")
    parser.add_argument("--trajectory", type=str, help="Dynamic obstacle trajectory file")
    parser.add_argument("--savevideo", action="store_true", help="Save animation as MP4 instead of showing it")
    args = parser.parse_args()

    path = read_path(args.pathfile)
    obstacles = read_obstacles(args.obstacles)

    cov_file = "covariances.txt"
    covariances = []
    if os.path.exists(cov_file):
        covariances = read_covariances(cov_file)
        print(f"[INFO] Loaded {len(covariances)} covariance entries from '{cov_file}'")
    else:
        print(f"[INFO] No covariance file found ('{cov_file}'). Ellipses will be zero-sized.")
    
    if len(covariances) < len(path):
        missing = len(path) - len(covariances)
        print(f"[WARN] Only {len(covariances)} covariances for {len(path)} path points; padding {missing} zeros.")
        zero_cov = ((0.0, 0.0), (0.0, 0.0))
        covariances.extend([zero_cov] * missing)

    animate_car_path(obstacles, path, covariances, 
                    trajectory_file=args.trajectory,
                    save_video=args.savevideo)

if __name__ == "__main__":
    main()
