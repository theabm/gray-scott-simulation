# Gray–Scott reaction–diffusion (2D) with MPI halo exchange (mpi4py)
# Weak-scaling friendly, single file, supports local/global seeding.

# TODO: mpi cuda aware transfers
# TODO: add numba cuda support for comparison using @cuda.jit
# TODO: add raw kernel (ElementwiseKernel - easiest - or RawKernel - harder) -- Note that this should be the same as above.

# To run with good configuration:
# mpirun -n 4 python sim.py  --steps 2000 --print-every 50 --seed-mode local --periodic --viz-every 50 --viz-gif

from re import A
from mpi4py import MPI
import argparse
import time
import os
import matplotlib
matplotlib.use("Agg")           # headless
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# GPU support - MPI not CUDA aware so we need to do host<->device copies ourselves
import numpy as np
import cupy as cp


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
    U  = cp.empty(shape, dtype=cp.float64)
    V  = cp.empty(shape, dtype=cp.float64)
    Un = cp.empty(shape, dtype=cp.float64)
    Vn = cp.empty(shape, dtype=cp.float64)
    return U, V, Un, Vn


# --------------------------------------------------------------------
# Seeding helpers
# --------------------------------------------------------------------
def seed_local_fraction(U, V, frac_lo=0.4, frac_hi=0.6, uval=0.50, vval=0.25):
    """Seed a square in the *local interior* defined by fractional coords."""
    Ui = U[1:-1, 1:-1]
    Vi = V[1:-1, 1:-1]
    nyi, nxi = Ui.shape

    yy, xx = cp.ogrid[0:nyi, 0:nxi]
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

    yy, xx = cp.ogrid[0:nyi, 0:nxi]
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
# Pinned memory allocation helper
# --------------------------------------------------------------------
def _pinned_empty_like(nbytes, dtype, nelements):
    """Create a NumPy array backed by pinned host memory with same shape/dtype."""

    # use cupy cuda allocator to allocate pinned memory in host (not device)
    # this will make host<->device transfers faster
    mem = cp.cuda.alloc_pinned_memory(nbytes)     

    # create a NumPy view into the pinned memory - numpy not responsible for freeing.
    host = np.frombuffer(mem, dtype=dtype, count=nelements)
    # return reshaped view and the mem handle (to keep alive)
    return host.reshape((nelements,)), mem

# --------------------------------------------------------------------
# Halo exchange
# --------------------------------------------------------------------
def exchange_halo(cart, A: cp.ndarray, B: cp.ndarray, staging: dict):
    assert A.shape == B.shape
    assert A.dtype == B.dtype

    ny, nx = A.shape
    first_row, last_row = 1, ny - 2
    first_col, last_col = 1, nx - 2
    status = MPI.Status()

    # neighbors in each direction
    up, down = cart.Shift(0, 1)
    left, right = cart.Shift(1, 1)

    if "rows" not in staging:
        # number of elements to send/recv in each direction
        nr_halo = (nx-2)*2
        nr_halo_bytes = nr_halo * A.dtype.itemsize

        staging["rows"] = {}
        staging["rows"]["send"], staging["rows"]["send_mem"] = _pinned_empty_like(nr_halo_bytes, A.dtype, nr_halo)
        staging["rows"]["recv"], staging["rows"]["recv_mem"] = _pinned_empty_like(nr_halo_bytes, A.dtype, nr_halo)

    if "cols" not in staging:
        nc_halo = (ny-2)*2
        nc_halo_bytes = nc_halo * A.dtype.itemsize

        staging["cols"] = {}
        staging["cols"]["send"], staging["cols"]["send_mem"] = _pinned_empty_like(nc_halo_bytes, A.dtype, nc_halo)
        staging["cols"]["recv"], staging["cols"]["recv_mem"] = _pinned_empty_like(nc_halo_bytes, A.dtype, nc_halo)

    # rows
    R = staging["rows"]
    sendbuf = R["send"]
    recvbuf = R["recv"]

    # send down
    cp.asnumpy(A[last_row, first_col:last_col+1], out = sendbuf[:nx-2])
    cp.asnumpy(B[last_row, first_col:last_col+1], out = sendbuf[nx-2:])

    cart.Sendrecv(sendbuf, dest=down, sendtag=101,
                  recvbuf=recvbuf, source=up, recvtag=101, status=status)
                  
    A[0, first_col:last_col+1].set(recvbuf[:nx-2])
    B[0, first_col:last_col+1].set(recvbuf[nx-2:])

    # send up
    cp.asnumpy(A[first_row, first_col:last_col+1], out = sendbuf[:nx-2])
    cp.asnumpy(B[first_row, first_col:last_col+1], out = sendbuf[nx-2:])

    cart.Sendrecv(sendbuf, dest=up, sendtag=102,
                  recvbuf=recvbuf, source=down, recvtag=102, status=status)

    A[ny-1, first_col:last_col+1].set(recvbuf[:nx-2])
    B[ny-1, first_col:last_col+1].set(recvbuf[nx-2:])

    # cols
    C = staging["cols"]
    sendbuf = C["send"]
    recvbuf = C["recv"]

    # send right
    send_right_A = cp.ascontiguousarray(A[first_row:last_row+1, last_col])
    send_right_B = cp.ascontiguousarray(B[first_row:last_row+1, last_col])
    cp.asnumpy(send_right_A, out = sendbuf[:ny-2])
    cp.asnumpy(send_right_B, out = sendbuf[ny-2:])

    cart.Sendrecv(sendbuf = sendbuf, dest=right, sendtag=201, recvbuf=recvbuf,  source=left, recvtag=201, status=status)
    A[first_row:last_row+1, 0] = cp.asarray(recvbuf[:ny-2])
    B[first_row:last_row+1, 0] = cp.asarray(recvbuf[ny-2:])

    # send left
    send_left_A = cp.ascontiguousarray(A[first_row:last_row+1, first_col])
    send_left_B = cp.ascontiguousarray(B[first_row:last_row+1, first_col])
    cp.asnumpy(send_left_A, out = sendbuf[:ny-2])
    cp.asnumpy(send_left_B, out = sendbuf[ny-2:])
    cart.Sendrecv(sendbuf=sendbuf, dest=left, sendtag=202, recvbuf=recvbuf, source=right, recvtag=202, status=status)
    A[first_row:last_row+1, nx-1] = cp.asarray(recvbuf[:ny-2])
    B[first_row:last_row+1, nx-1] = cp.asarray(recvbuf[ny-2:])

# --------------------------------------------------------------------
# update ghost cells
# --------------------------------------------------------------------
def update_ghosts(cart, A, B, periodic, staging):
    # Always exchange with interior neighbors
    exchange_halo(cart, A, B, staging)

    # Only BCs on physical edges (no-ops for interior ranks, or if periodic)
    if not periodic:
        apply_neumann_boundary_ghosts(cart, A)
        apply_neumann_boundary_ghosts(cart, B)


# --------------------------------------------------------------------
# Gray–Scott update
# --------------------------------------------------------------------
def step_gray_scott(U, V, Un, Vn, Du, Dv, F, k, dt):
    Ui = U[1:-1, 1:-1]
    Vi = V[1:-1, 1:-1]

    Lu = (-4.0 * Ui
          + U[1:-1, 0:-2]
          + U[1:-1, 2:  ]
          + U[0:-2, 1:-1]
          + U[2:  , 1:-1])

    Lv = (-4.0 * Vi
          + V[1:-1, 0:-2]
          + V[1:-1, 2:  ]
          + V[0:-2, 1:-1]
          + V[2:  , 1:-1])

    uvv = Ui * Vi * Vi
    Un[1:-1, 1:-1] = Ui + dt * (Du * Lu - uvv + F * (1.0 - Ui))
    Vn[1:-1, 1:-1] = Vi + dt * (Dv * Lv + uvv - (F + k) * Vi)

def gather_global_field(cart, A_interior, nx, ny, px, py):
    """
    Gather each rank's interior (ny x nx) to rank 0 and assemble the global (NY x NX).
    A_interior must be contiguous (U[1:-1,1:-1] or V[1:-1,1:-1]).
    """
    rank = cart.Get_rank()
    size = cart.Get_size()
    coords = cart.Get_coords(rank)

    tile_h = cp.asnumpy(A_interior)

    # Gather tiles and their coords to rank 0
    tiles  = cart.gather(tile_h, root=0)
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

    # allocate on GPU device
    U, V, Un, Vn = allocate_fields(ny, nx)
    rng = cp.random.default_rng(args.seed + rank)
    initialize(U, V, coords, rng, args, px, py)

    staging = {}
    
    update_ghosts(cart, U, V, args.periodic, staging)
    # after initialize(...)
    if args.viz_every and args.viz_every > 0:
        save_frame(0, cart, U, V, nx, ny, px, py,
               which=args.viz_field, outdir=args.viz_outdir, clim=args.viz_clim)


    t0 = time.time()
    for step in range(1, args.steps + 1):

        halo_start = time.perf_counter()
        update_ghosts(cart, U, V, args.periodic, staging)
        halo_end = time.perf_counter()

        gss_start = time.perf_counter()
        step_gray_scott(U, V, Un, Vn, args.Du, args.Dv, args.F, args.k, args.dt)
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
