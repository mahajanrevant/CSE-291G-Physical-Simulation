"""
================================================================================
ADVANCED CLOTH SIMULATION — St. Venant-Kirchhoff Continuum Mechanics
================================================================================
 
Implements the force-evaluation pipeline from Prof. Chern's CSE 291 notes:
 
    Input phi -> F = d phi -> C = F^T F -> E = 1/2(C - I)
              -> S = lambda tr(E) I + 2mu E -> P = F S -> f = div(P)
 
Each function is cross-referenced to the CONCEPTS we worked through:
 
    Concept 1  : Flow map phi (M -> W), material vs world space
    Concept 2  : Deformation gradient F = d phi          (3x2 per triangle)
    Concept 3  : Right Cauchy-Green C = F^T F            (2x2, removes rotation)
    Concept 4  : Green-St Venant strain E = 1/2(C - I)   (2x2, zero at rest)
    Concept 5  : StVK energy Psi = lambda/2 tr(E)^2 + mu tr(E^2)
    Concept 6  : 2nd Piola stress S = lambda tr(E) I + 2mu E
    Concept 7  : 1st Piola stress P = F S                (3x2, back to world)
    Concept 8  : Force f = div(P)
    Concept 9  : Discretization — F = Ds @ Dm^{-1}, per-vertex force
    Concept 10 : Symplectic Euler time integration
    Concept 11 : Pinning constraints
 
Dependencies: numpy, matplotlib
================================================================================
"""
 
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend: works over SSH with no display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
 
 
# ==============================================================================
# CONCEPT 1 & 9 : THE CLOTH MESH (material space M, discretized)
# ==============================================================================
class ClothMesh:
    """
    The discrete material space M.
 
    Stores BOTH:
      - reference (material) coordinates  : the flat rest sheet, coords (X, Y)
      - world coordinates x               : current 3D positions = phi(material)
 
    The flow map phi (Concept 1) is represented implicitly: phi maps each
    material vertex to its current world position x[i]. Between vertices, phi
    is affine on each triangle, which is what makes F constant per triangle
    (Concept 9).
    """
 
    def __init__(self, width=1.0, height=0.6, nx=14, ny=9, density=1.0):
        self.nx, self.ny = nx, ny
 
        # --- Material (reference) coordinates: a flat 2D grid -----------------
        Xs = np.linspace(0.0, width,  nx)
        Ys = np.linspace(0.0, height, ny)
        Xg, Yg = np.meshgrid(Xs, Ys)
        self.material_2d = np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float64)
        self.N = self.material_2d.shape[0]
 
        # --- World coordinates: initial configuration ------------------------
        self.x = np.zeros((self.N, 3), dtype=np.float64)
        self.x[:, 0] = self.material_2d[:, 0]
        self.x[:, 1] = self.material_2d[:, 1]
        self.x[:, 2] = 0.0
 
        self.v = np.zeros((self.N, 3), dtype=np.float64)
 
        # --- Triangulate the grid (Concept 9) --------------------------------
        self.triangles = self._triangulate(nx, ny)
        self.T = self.triangles.shape[0]
 
        # --- Precompute per-triangle rest quantities (Concept 9) -------------
        self.Dm_inv = np.zeros((self.T, 2, 2), dtype=np.float64)
        self.rest_area = np.zeros(self.T, dtype=np.float64)
        self._precompute_rest()
 
        # --- Lumped vertex masses (Concept 9) --------------------------------
        self.m = np.zeros(self.N, dtype=np.float64)
        for t in range(self.T):
            share = density * self.rest_area[t] / 3.0
            for vid in self.triangles[t]:
                self.m[vid] += share
        self.m[self.m < 1e-12] = 1e-12
 
    @staticmethod
    def _triangulate(nx, ny):
        tris = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                v00 = j * nx + i
                v10 = j * nx + (i + 1)
                v01 = (j + 1) * nx + i
                v11 = (j + 1) * nx + (i + 1)
                tris.append([v00, v10, v01])
                tris.append([v10, v11, v01])
        return np.array(tris, dtype=np.int64)
 
    def _precompute_rest(self):
        """
        CONCEPT 9 — the rest-shape correction Dm^{-1}.
 
        Dm columns = material edge vectors: Dm = [P1-P0 | P2-P0]  (2x2).
        F = Ds @ Dm^{-1}. Dm^{-1} converts world edge vectors into the true
        gradient d phi / dX. This is the discrete reference-metric correction
        C_hat = (flat^M)^{-1} C, since flat^M = Dm^T Dm.
        """
        for t, (i0, i1, i2) in enumerate(self.triangles):
            P0 = self.material_2d[i0]
            P1 = self.material_2d[i1]
            P2 = self.material_2d[i2]
            Dm = np.column_stack([P1 - P0, P2 - P0])
            self.Dm_inv[t] = np.linalg.inv(Dm)
            self.rest_area[t] = 0.5 * abs(np.linalg.det(Dm))
 
 
# ==============================================================================
# THE SIMULATOR
# ==============================================================================
class ContinuumClothSimulator:
    def __init__(self, cloth, lam=80.0, mu=80.0,
                 gravity=(0.0, 0.0, -9.8), damping=0.4, dt=2e-3):
        self.cloth = cloth
        self.lam = lam
        self.mu = mu
        self.gravity = np.array(gravity, dtype=np.float64)
        self.damping = damping
        self.dt = dt
        self.forces = np.zeros_like(cloth.x)
        self.x_old = cloth.x.copy()
 
    def compute_elastic_forces(self):
        """
        CONCEPTS 2-8, per triangle:
            F = Ds Dm^{-1}  (2/9)   C = F^T F  (3)   E = 1/2(C - I)  (4)
            S = lam tr(E) I + 2 mu E  (6)   P = F S  (7)
            [f1|f2] = -A P Dm^{-T}    (8/9)
        """
        self.forces.fill(0.0)
        x = self.cloth.x
        I2 = np.eye(2)
 
        for t in range(self.cloth.T):
            i0, i1, i2 = self.cloth.triangles[t]
 
            # CONCEPT 2 & 9 : deformation gradient F (3x2)
            Ds = np.column_stack([x[i1] - x[i0], x[i2] - x[i0]])
            F = Ds @ self.cloth.Dm_inv[t]
 
            # CONCEPT 3 : right Cauchy-Green C (2x2)
            C = F.T @ F
 
            # CONCEPT 4 : strain E (2x2)
            E = 0.5 * (C - I2)
 
            # CONCEPT 6 : 2nd Piola stress S (2x2)
            S = self.lam * np.trace(E) * I2 + 2.0 * self.mu * E
 
            # CONCEPT 7 : 1st Piola stress P (3x2)
            P = F @ S
 
            # CONCEPT 8 & 9 : force = div(P) -> vertices
            A = self.cloth.rest_area[t]
            H = -A * (P @ self.cloth.Dm_inv[t].T)
            f1 = H[:, 0]
            f2 = H[:, 1]
            f0 = -(f1 + f2)
 
            self.forces[i0] += f0
            self.forces[i1] += f1
            self.forces[i2] += f2
 
    def add_gravity_and_damping(self):
        self.forces += self.cloth.m[:, None] * self.gravity[None, :]
        self.forces += -self.damping * self.cloth.m[:, None] * self.cloth.v
 
    def step(self, pinned_indices=None, target_fn=None, t_now=0.0):
        """
        CONCEPT 10 & 11. Symplectic Euler:
            v <- v + dt f / m   then   x <- x + dt v
        Pinned vertices placed directly at targets; velocity back-derived.
        """
        cloth = self.cloth
        self.x_old = cloth.x.copy()
 
        self.compute_elastic_forces()
        self.add_gravity_and_damping()
 
        pinned = set() if pinned_indices is None else set(pinned_indices)
 
        free = np.array([i for i in range(cloth.N) if i not in pinned], dtype=np.int64)
        a = self.forces[free] / cloth.m[free][:, None]
        cloth.v[free] += self.dt * a
        cloth.x[free] += self.dt * cloth.v[free]
 
        if pinned_indices is not None and target_fn is not None:
            for i in pinned_indices:
                target = target_fn(i, t_now)
                cloth.x[i] = target
                cloth.v[i] = (cloth.x[i] - self.x_old[i]) / self.dt
 
 
# ==============================================================================
# PINNING SETUP (CONCEPT 11)
# ==============================================================================
def make_edge_pins(cloth, lift=0.45, sway_amp=0.18, sway_speed=1.6,
                   ramp_time=1.0):
    """
    CONCEPT 11. Pin the left and right columns and animate them.
 
    IMPORTANT (stability): the prescribed motion must be SMOOTH in time.
    The pinned velocity is back-derived as (x_target(t) - x_old)/dt, so any
    instantaneous jump in the target produces an enormous spurious velocity
    that detonates the explicit integrator. We therefore:
      - start the targets exactly at the rest position (no jump at t=0), and
      - ramp the lift/sway in over `ramp_time` seconds using a smooth
        ease-in (1 - cos) profile.
    """
    nx, ny = cloth.nx, cloth.ny
    left_col  = [j * nx + 0        for j in range(ny)]
    right_col = [j * nx + (nx - 1) for j in range(ny)]
    pinned = left_col + right_col
 
    rest = {i: cloth.x[i].copy() for i in pinned}
    left_set = set(left_col)
 
    def target_fn(i, t):
        base = rest[i].copy()
        # smooth ease-in: 0 at t=0, 1 at t>=ramp_time, with zero initial slope
        if t < ramp_time:
            ramp = 0.5 * (1.0 - np.cos(np.pi * t / ramp_time))
        else:
            ramp = 1.0
        base[2] += lift * ramp
        phase = 0.0 if i in left_set else np.pi
        base[0] += sway_amp * ramp * np.sin(sway_speed * t + phase)
        base[2] += 0.05 * ramp * np.sin(0.8 * sway_speed * t)
        return base
 
    return pinned, target_fn
 
 
# ==============================================================================
# VISUALIZATION
# ==============================================================================
def animate(cloth, sim, pinned, target_fn,
            n_frames=240, substeps=8, save_path=None):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    pinned_arr = np.array(pinned)
 
    def update(frame):
        ax.clear()
        for s in range(substeps):
            t_now = (frame * substeps + s) * sim.dt
            sim.step(pinned_indices=pinned, target_fn=target_fn, t_now=t_now)
 
        x = cloth.x
        ax.plot_trisurf(x[:, 0], x[:, 1], x[:, 2],
                        triangles=cloth.triangles,
                        color=(0.35, 0.55, 0.85), alpha=0.85,
                        edgecolor=(0.2, 0.3, 0.5), linewidth=0.2)
        ax.scatter(x[pinned_arr, 0], x[pinned_arr, 1], x[pinned_arr, 2],
                   color='crimson', s=25, depthshade=False)
        ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.3, 0.9); ax.set_zlim(-0.9, 0.7)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.set_title(f'StVK cloth - frame {frame}')
        ax.view_init(elev=18, azim=-60)
        return ()
 
    anim = FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False)
    if save_path:
        anim.save(save_path, writer='pillow', fps=24, dpi=90)
        print(f"Saved animation to {save_path}")
    return fig, anim
 
 
# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    # Tuned for a soft, visibly draping cloth that stays numerically stable
    # under explicit (symplectic Euler) integration:
    #   - density 5.0 makes gravity pull hard enough to sag nicely
    #   - lam = mu = 20 is soft enough to drape, stiff enough to look like cloth
    #   - dt = 1e-3 sits safely under the explicit stability limit 2*sqrt(m/k)
    #   - the pin ramp (see make_edge_pins) avoids the start-of-sim velocity spike
    cloth = ClothMesh(width=1.0, height=0.6, nx=14, ny=9, density=5.0)
    sim = ContinuumClothSimulator(cloth, lam=20.0, mu=20.0,
                                  gravity=(0, 0, -9.8), damping=0.6, dt=1e-3)
    pinned, target_fn = make_edge_pins(cloth, lift=0.45, sway_amp=0.18,
                                       sway_speed=1.6, ramp_time=1.0)
    # substeps=16 because dt is small; 16 physics steps per drawn frame
    # Headless (SSH): use the non-interactive Agg backend and save to a GIF
    # instead of plt.show().
    fig, anim = animate(cloth, sim, pinned, target_fn,
                        n_frames=240, substeps=16,
                        save_path='cloth_simulation_advanced.gif')