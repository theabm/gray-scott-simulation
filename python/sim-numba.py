# Gray–Scott reaction–diffusion (2D) with MPI halo exchange (mpi4py)
# Weak-scaling friendly, single file, supports local/global seeding.

# To run with good configuration:
# mpirun -n 4 python sim.py  --steps 2000 --print-every 50 --seed-mode local --periodic --viz-every 50 --viz-gif

from mpi4py import MPI
import numpy as np
import argparse
import time
import os
import matplotlib
matplotlib.use("Agg")           # headless
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# numba for JIT acceleration of the inner loop
import numba



# --------------------------------------------------------------------
# CLI parsing
# --------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Gray–Scott 2D (MPI, weak-scaling oriented)")
    ap.add_argument("--nx_local", type=int, default=256, help="local interior width (X)")
    ap.add_argument("--ny_local", type=int, default=256, help="local interior height (Y)")
    ap.add_argument("--px", type=int, default=0, help="processes along X (columns)")
    ap.add_argument("--py", type=int, default=0, help="processes along Y (rows)")
    ap.add_argument("--periodic", action="store_true", help="use periodic BCs in both dims")
    ap.add_argument("--steps", type=int, default=500, help="number of time steps")
    ap.add_argument("--dt", type=float, default=1.0, help="time step")
    ap.add_argument("--Du", type=float, default=0.16, help="diffusion coeff for U")
    ap.add_argument("--Dv", type=float, default=0.08, help="diffusion coeff for V")
    ap.add_argument("--F",  type=float, default=0.060, help="feed rate")
    ap.add_argument("--k",  type=float, default=0.062, help="kill rate")
    ap.add_argument("--print-every", type=int, default=50, help="rank0 prints progress every N steps")
    ap.add_argument("--seed", type=int, default=0, help="base RNG seed (per-rank offset applied)")

    # Seeding options
    ap.add_argument("--seed-mode", choices=["local", "global"], default="local",
                    help="how to place the perturbation square")
    ap.add_argument("--seed-frac", nargs=2, type=float, default=[0.4, 0.6],
                    help="fractional bounds (low high) of seeded square in [0,1]")

    # Visualization options
    ap.add_argument("--viz-every", type=int, default=0,
                help="save a PNG every N steps (0 = off)")
    ap.add_argument("--viz-field", choices=["U", "V", "both"], default="V",
                help="which field to visualize")
    ap.add_argument("--viz-outdir", type=str, default="frames",
                help="directory to save frames")
    ap.add_argument("--viz-clim", nargs=2, type=float, default=None,
                help="color limits vmin vmax (default: autoscale)")
    ap.add_argument("--viz-gif", action="store_true",
                help="assemble saved frames into out.gif (rank0 only)")


    return ap.parse_args()


# --------------------------------------------------------------------
# MPI helpers
# --------------------------------------------------------------------
def make_cartesian(comm_world, px, py, periodic):
    size = comm_world.Get_size()
    dims = [py, px]
    dims = MPI.Compute_dims(size, dims)
    py, px = dims
    periods = (1 if periodic else 0, 1 if periodic else 0)
    cart = comm_world.Create_cart(dims=(py, px), periods=periods, reorder=True)
    coords = cart.Get_coords(cart.Get_rank())
    return cart, (py, px), coords


# --------------------------------------------------------------------
# Array allocation
# --------------------------------------------------------------------
def allocate_fields(ny_local, nx_local):
    shape = (ny_local + 2, nx_local + 2)  # +2 for ghosts
    U  = np.empty(shape, dtype=np.float32)
    V  = np.empty(shape, dtype=np.float32)
    Un = np.empty(shape, dtype=np.float32)
    Vn = np.empty(shape, dtype=np.float32)
    return U, V, Un, Vn


# --------------------------------------------------------------------
# Seeding helpers
# --------------------------------------------------------------------
def seed_local_fraction(U, V, frac_lo=0.4, frac_hi=0.6, uval=0.50, vval=0.25):
    """Seed a square in the *local interior* defined by fractional coords."""
    Ui = U[1:-1, 1:-1]
    Vi = V[1:-1, 1:-1]
    nyi, nxi = Ui.shape

    yy, xx = np.ogrid[0:nyi, 0:nxi]
    y_norm = (yy + 0.5) / nyi
    x_norm = (xx + 0.5) / nxi

    mask = (y_norm > frac_lo) & (y_norm < frac_hi) & \
           (x_norm > frac_lo) & (x_norm < frac_hi)

    Ui[mask] = uval
    Vi[mask] = vval
    return U,V


def seed_global_fraction(U, V, coords, px, py, nx, ny,
                         frac_lo=0.4, frac_hi=0.6, uval=0.50, vval=0.25):
    """Seed a square in *global* fractional coordinates."""
    ry, rx = coords
    NX, NY = px * nx, py * ny
    x0, x1 = rx * nx, (rx + 1) * nx
    y0, y1 = ry * ny, (ry + 1) * ny

    Ui = U[1:-1, 1:-1]
    Vi = V[1:-1, 1:-1]
    nyi, nxi = Ui.shape

    yy, xx = np.ogrid[0:nyi, 0:nxi]
    xg = (x0 + (xx + 0.5)) / NX
    yg = (y0 + (yy + 0.5)) / NY

    mask = (yg > frac_lo) & (yg < frac_hi) & \
           (xg > frac_lo) & (xg < frac_hi)

    Ui[mask] = uval
    Vi[mask] = vval
    return U, V


# --------------------------------------------------------------------
# Initialization
# --------------------------------------------------------------------
def initialize(U, V, coords, rng, args, px, py):
    U.fill(1.0)
    V.fill(0.0)

    # random noise in the interior
    Ui = U[1:-1, 1:-1]
    Vi = V[1:-1, 1:-1]
    noise = 0.02 * rng.random(Ui.shape)
    Ui -= noise
    Vi += noise

    lo, hi = args.seed_frac
    if args.seed_mode == "local":
        U,V = seed_local_fraction(U, V, lo, hi)
    elif args.seed_mode == "global":
        U,V = seed_global_fraction(U, V, coords, px, py, args.nx_local, args.ny_local, lo, hi)

# --------------------------------------------------------------------
# Von Neumann ghost initialization
# --------------------------------------------------------------------
def apply_neumann_boundary_ghosts(cart, A):
    """
    Enforce zero-flux (Neumann) ONLY on physical boundaries.
    Interior ranks (with real neighbors) do nothing.

    Only needed for non-periodic BCs.
    """
    ny, nx = A.shape
    first_row, last_row = 1, ny - 2
    first_col, last_col = 1, nx - 2

    up,   down  = cart.Shift(0, 1)  # Y-dim neighbors
    left, right = cart.Shift(1, 1)  # X-dim neighbors

    # Top boundary: no UP neighbor
    if up == MPI.PROC_NULL:
        A[0, first_col:last_col+1] = A[1, first_col:last_col+1]
        A[0, 0]    = A[1, 1]        # corners (cosmetic)
        A[0, -1]   = A[1, -2]

    # Bottom boundary: no DOWN neighbor
    if down == MPI.PROC_NULL:
        A[-1, first_col:last_col+1] = A[-2, first_col:last_col+1]
        A[-1, 0]   = A[-2, 1]
        A[-1, -1]  = A[-2, -2]

    # Left boundary: no LEFT neighbor
    if left == MPI.PROC_NULL:
        A[first_row:last_row+1, 0] = A[first_row:last_row+1, 1]
        A[0, 0]     = A[1, 1]
        A[-1, 0]    = A[-2, 1]

    # Right boundary: no RIGHT neighbor
    if right == MPI.PROC_NULL:
        A[first_row:last_row+1, -1] = A[first_row:last_row+1, -2]
        A[0, -1]    = A[1, -2]
        A[-1, -1]   = A[-2, -2]


# --------------------------------------------------------------------
# Halo exchange
# --------------------------------------------------------------------
def exchange_halo(cart, A):
    """
    Exchange ghost cells with interior neighbors.
    Works for both periodic and non-periodic BCs.
    """

    ny, nx = A.shape
    first_row, last_row = 1, ny - 2
    first_col, last_col = 1, nx - 2
    status = MPI.Status()

    up, down = cart.Shift(0, 1)
    left, right = cart.Shift(1, 1)

    # rows
    cart.Sendrecv(A[last_row, first_col:last_col+1], dest=down, sendtag=101,
                  recvbuf=A[0, first_col:last_col+1], source=up, recvtag=101, status=status)
    cart.Sendrecv(A[first_row, first_col:last_col+1], dest=up, sendtag=102,
                  recvbuf=A[ny-1, first_col:last_col+1], source=down, recvtag=102, status=status)

    # cols
    col_len = ny - 2
    send_right = np.ascontiguousarray(A[first_row:last_row+1, last_col])
    recv_left  = np.empty(col_len, dtype=A.dtype)
    cart.Sendrecv(sendbuf = send_right, dest=right, sendtag=201, recvbuf=recv_left,  source=left, recvtag=201, status=status)
    A[first_row:last_row+1, 0] = recv_left

    send_left  = np.ascontiguousarray(A[first_row:last_row+1, first_col])
    recv_right = np.empty(col_len, dtype=A.dtype)
    cart.Sendrecv(sendbuf=send_left, dest=left, sendtag=202, recvbuf=recv_right, source=right, recvtag=202, status=status)
    A[first_row:last_row+1, nx-1] = recv_right

# --------------------------------------------------------------------
# update ghost cells
# --------------------------------------------------------------------
def update_ghosts(cart, A, periodic):
    # Always exchange with interior neighbors
    exchange_halo(cart, A)

    # Only BCs on physical edges (no-ops for interior ranks, or if periodic)
    if not periodic:
        apply_neumann_boundary_ghosts(cart, A)


# --------------------------------------------------------------------
# Gray–Scott update
# --------------------------------------------------------------------
@numba.njit(parallel=False, fastmath = True, cache=True)
def step_gray_scott_fused(U, V, Un, Vn, Du, Dv, F, k, dt):
    ny, nx = U.shape
    # interior: [1:ny-1, 1:nx-1]

    for i in numba.prange(1, ny-1):
        for j in range(1, nx-1):

            ui = U[i, j]
            vi = V[i, j]

            uvv = ui * vi * vi

            Lu = (-4.0 * ui + U[i, j-1] + U[i, j+1] + U[i-1, j] + U[i+1, j])
            Lv = (-4.0 * vi + V[i, j-1] + V[i, j+1] + V[i-1, j] + V[i+1, j])

            Un[i,j] = ui + dt * (Du * Lu - uvv + F * (1.0 - ui))
            Vn[i,j] = vi + dt * (Dv * Lv + uvv - (F + k) * vi)


def gather_global_field(cart, A_interior, nx, ny, px, py):
    """
    Gather each rank's interior (ny x nx) to rank 0 and assemble the global (NY x NX).
    A_interior must be contiguous (U[1:-1,1:-1] or V[1:-1,1:-1]).
    """
    rank = cart.Get_rank()
    size = cart.Get_size()
    coords = cart.Get_coords(rank)

    # Gather tiles and their coords to rank 0
    tiles  = cart.gather(np.ascontiguousarray(A_interior), root=0)
    allcs  = cart.gather(coords, root=0)

    if rank != 0:
        return None

    NY, NX = py * ny, px * nx
    G = np.empty((NY, NX), dtype=A_interior.dtype)

    # Place each tile into its global slot using its (ry, rx)
    for tile, (ry, rx) in zip(tiles, allcs):
        G[ry*ny:(ry+1)*ny, rx*nx:(rx+1)*nx] = tile
    return G


def save_frame(step, cart, U, V, nx, ny, px, py, which="V", outdir="frames", clim=None):
    """
    Gather global field(s) and save a PNG on rank 0.
    which: "U" | "V" | "both"
    clim: (vmin, vmax) or None for autoscale
    """
    rank = cart.Get_rank()
    os.makedirs(outdir, exist_ok=True)

    Ui = U[1:-1, 1:-1]
    Vi = V[1:-1, 1:-1]

    if which in ("U", "both"):
        Ug = gather_global_field(cart, Ui, nx, ny, px, py)
    else:
        Ug = None

    if which in ("V", "both"):
        Vg = gather_global_field(cart, Vi, nx, ny, px, py)
    else:
        Vg = None

    if rank != 0:
        return

    # Prepare figure
    if which == "both":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        ims = []
        ims.append(axes[0].imshow(Ug, origin="lower", interpolation="nearest"))
        axes[0].set_title("U")
        ims.append(axes[1].imshow(Vg, origin="lower", interpolation="nearest"))
        axes[1].set_title("V")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        if clim:
            ims[0].set_clim(*clim)
            ims[1].set_clim(*clim)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
        data = Ug if which == "U" else Vg
        im = ax.imshow(data, origin="lower", interpolation="nearest")
        if clim:
            im.set_clim(*clim)
        ax.set_title(which)
        ax.set_xticks([]); ax.set_yticks([])

    fname = os.path.join(outdir, f"step_{step:06d}.png")
    fig.savefig(fname, dpi=120)
    plt.close(fig)

def assemble_gif(outdir="frames", outfile="out.gif", fps=10):
    """
    Assemble all PNGs in outdir into a GIF.
    Only rank 0 should call this.
    """
    files = sorted(f for f in os.listdir(outdir) if f.endswith(".png"))
    if not files:
        print(f"[viz] No PNGs found in {outdir}, skipping GIF.")
        return
    print(f"[viz] Assembling {len(files)} frames into {outfile}")
    frames = []
    for fn in files:
        path = os.path.join(outdir, fn)
        frames.append(imageio.imread(path))
    imageio.mimsave(outfile, frames, fps=fps)
    print(f"[viz] Wrote {outfile}")

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cart, (py, px), coords = make_cartesian(comm, args.px, args.py, args.periodic)
    nx, ny = args.nx_local, args.ny_local
    NX, NY = px * nx, py * ny

    U, V, Un, Vn = allocate_fields(ny, nx)
    rng = np.random.default_rng(args.seed + rank)
    initialize(U, V, coords, rng, args, px, py)

    update_ghosts(cart, U, args.periodic)
    update_ghosts(cart, V, args.periodic)
    # after initialize(...)
    if args.viz_every and args.viz_every > 0:
        save_frame(0, cart, U, V, nx, ny, px, py,
               which=args.viz_field, outdir=args.viz_outdir, clim=args.viz_clim)

    # make sure scalars match dtype
    Du = np.float32(args.Du); Dv = np.float32(args.Dv)
    F  = np.float32(args.F);  k  = np.float32(args.k)
    dt = np.float32(args.dt)

    t0 = time.time()
    for step in range(1, args.steps + 1):

        halo_start = time.perf_counter()
        update_ghosts(cart, U, args.periodic)
        update_ghosts(cart, V, args.periodic)   
        halo_end = time.perf_counter()

        gss_start = time.perf_counter()
        step_gray_scott_fused(U, V, Un, Vn, Du, Dv, F, k, dt)
        gss_end = time.perf_counter()

        U, Un = Un, U
        V, Vn = Vn, V
        
        if args.viz_every and (step % args.viz_every == 0):
            save_frame(step, cart, U, V, nx, ny, px, py,
                   which=args.viz_field, outdir=args.viz_outdir, clim=args.viz_clim)

        if args.print_every and (step % args.print_every == 0):
            comm.Barrier()
            local_sum = float(V[1:-1, 1:-1].sum())
            global_sum = comm.allreduce(local_sum, op=MPI.SUM)
            if rank == 0:
                elapsed = time.time() - t0
                gss_elapsed = (gss_end - gss_start)*1e3
                halo_elapsed = (halo_end - halo_start)*1e3
                print(f"[step {step:6d}] ranks={size} grid={py}x{px} "
                      f"N={NY}x{NX} local={ny}x{nx} Vsum={global_sum:.6e} "
                      f"elapsed={elapsed:.2f}s "
                      f"GSS_time={gss_elapsed:.2f}ms "
                      f"halo_time={halo_elapsed:.2f}ms"
                      )

    comm.Barrier()
    if rank == 0:
        if args.viz_gif and args.viz_every and args.viz_every > 0:
            gif_path = os.path.join(args.viz_outdir, "out.gif")
            assemble_gif(args.viz_outdir, gif_path, fps=10)
        print("DONE.")


if __name__ == "__main__":
    main()
