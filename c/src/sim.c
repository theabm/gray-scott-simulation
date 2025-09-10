// Gray–Scott reaction–diffusion (2D) with MPI halo exchange (C)
// -----------------------------------------------------------------------------
// Intent: closely mirror the Python mpi4py version's logic & update order.
// - Same parameters & defaults (see CLI flags below)
// - Same domain decomposition: 2D Cartesian grid with (py x px) ranks
// - Same ghost layout: arrays sized (ny+2) x (nx+2) with 1-cell halo
// - Same halo exchange pattern + optional Neumann (zero-flux) on physical edges
// - Same seeding options: local/global square via fractional coordinates
// - Same update order each step: exchange ghosts U, exchange ghosts V, update,
//   swap, optional gather+save frame, optional progress print from rank 0.
// Differences vs Python for visualization only:
// - Frames are saved as PGM (8-bit grayscale) instead of PNG; "both" writes
//   two images per step with suffix _U / _V. You can convert to GIF with e.g.:
//     convert frames/step_*.pgm out.gif
//   (the --viz-gif flag will just print a hint on rank 0.)
//
// Build:
//   mpicc -O3 -std=c11 -Wall -Wextra -o gray_scott_mpi gray_scott_mpi.c
// Run (example):
//   mpirun -n 4 ./gray_scott_mpi --steps 2000 --print-every 50 --seed-mode local --periodic --viz-every 50 --viz-gif
// -----------------------------------------------------------------------------

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

// -----------------------------------------------
// Utility: directory creation (no error if exists)
// -----------------------------------------------
static void ensure_dir(const char *path) {
    if (mkdir(path, 0777) != 0 && errno != EEXIST) {
        fprintf(stderr, "[warn] mkdir %s failed: %s\n", path, strerror(errno));
    }
}

// -----------------------------------------------
// Simple RNG (xorshift64*) for portability
// -----------------------------------------------
typedef struct { unsigned long long s; } rng64;

static void rng_seed(rng64 *r, unsigned long long seed) {
    // avoid zero state
    r->s = seed ? seed : 88172645463393265ull;
}

static inline unsigned long long xorshift64s(rng64 *r) {
    unsigned long long x = r->s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->s = x;
    return x * 2685821657736338717ull;
}

static inline double rng_uniform01(rng64 *r) {
    // 53-bit mantissa to [0,1)
    return (xorshift64s(r) >> 11) * (1.0/9007199254740992.0);
}

// -----------------------------------------------
// Config / CLI
// -----------------------------------------------
typedef struct {
    int nx_local;         // interior width (X)
    int ny_local;         // interior height (Y)
    int px;               // processes along X (columns)
    int py;               // processes along Y (rows)
    int periodic;         // bool
    int steps;
    double dt, Du, Dv, F, k;
    int print_every;
    int seed;

    // seeding
    enum { SEED_LOCAL, SEED_GLOBAL } seed_mode;
    double seed_lo, seed_hi; // fraction bounds

    // viz
    int viz_every;            
    enum { VIZ_U, VIZ_V, VIZ_BOTH } viz_field;
    char viz_outdir[256];
    double viz_vmin, viz_vmax;
    int viz_gif;              // if set, print hint on rank 0
} Config;

static void set_defaults(Config *c) {
    c->nx_local = 256;
    c->ny_local = 256;
    c->px = 0; c->py = 0;
    c->periodic = 0;
    c->steps = 500;
    c->dt = 1.0; c->Du = 0.16; c->Dv = 0.08; c->F = 0.060; c->k = 0.062;
    c->print_every = 50;
    c->seed = 0;
    c->seed_mode = SEED_LOCAL;
    c->seed_lo = 0.4; c->seed_hi = 0.6;
    c->viz_every = 0;
    c->viz_field = VIZ_V;
    strcpy(c->viz_outdir, "frames");
    c->viz_vmin = 0.0; c->viz_vmax = 1.0;
    c->viz_gif = 0;
}

static int streq(const char *a, const char *b){ return strcmp(a,b)==0; }

static void parse_args(Config *c, int argc, char **argv, int rank) {
    set_defaults(c);
    for (int i=1; i<argc; ++i) {
        if (streq(argv[i], "--nx_local") && i+1<argc) c->nx_local = atoi(argv[++i]);
        else if (streq(argv[i], "--ny_local") && i+1<argc) c->ny_local = atoi(argv[++i]);
        else if (streq(argv[i], "--px") && i+1<argc) c->px = atoi(argv[++i]);
        else if (streq(argv[i], "--py") && i+1<argc) c->py = atoi(argv[++i]);
        else if (streq(argv[i], "--periodic")) c->periodic = 1;
        else if (streq(argv[i], "--steps") && i+1<argc) c->steps = atoi(argv[++i]);
        else if (streq(argv[i], "--dt") && i+1<argc) c->dt = atof(argv[++i]);
        else if (streq(argv[i], "--Du") && i+1<argc) c->Du = atof(argv[++i]);
        else if (streq(argv[i], "--Dv") && i+1<argc) c->Dv = atof(argv[++i]);
        else if (streq(argv[i], "--F")  && i+1<argc) c->F  = atof(argv[++i]);
        else if (streq(argv[i], "--k")  && i+1<argc) c->k  = atof(argv[++i]);
        else if (streq(argv[i], "--print-every") && i+1<argc) c->print_every = atoi(argv[++i]);
        else if (streq(argv[i], "--seed") && i+1<argc) c->seed = atoi(argv[++i]);
        else if (streq(argv[i], "--seed-mode") && i+1<argc) {
            const char *s = argv[++i];
            if (streq(s, "local")) c->seed_mode = SEED_LOCAL;
            else if (streq(s, "global")) c->seed_mode = SEED_GLOBAL;
        }
        else if (streq(argv[i], "--seed-frac") && i+2<argc) {
            c->seed_lo = atof(argv[++i]);
            c->seed_hi = atof(argv[++i]);
        }
        else if (streq(argv[i], "--viz-every") && i+1<argc) c->viz_every = atoi(argv[++i]);
        else if (streq(argv[i], "--viz-field") && i+1<argc) {
            const char *s = argv[++i];
            if (streq(s, "U")) c->viz_field = VIZ_U;
            else if (streq(s, "V")) c->viz_field = VIZ_V;
            else if (streq(s, "both")) c->viz_field = VIZ_BOTH;
        }
        else if (streq(argv[i], "--viz-outdir") && i+1<argc) {
            strncpy(c->viz_outdir, argv[++i], sizeof(c->viz_outdir)-1);
            c->viz_outdir[sizeof(c->viz_outdir)-1] = '\0';
        }
        else if (streq(argv[i], "--viz-gif")) c->viz_gif = 1;
        else if (streq(argv[i], "-h") || streq(argv[i], "--help")) {
            if (rank == 0) {
                printf("Gray–Scott 2D (MPI)\n");
                printf("Options:\n");
                printf("  --nx_local INT       [256]\n");
                printf("  --ny_local INT       [256]\n");
                printf("  --px INT             [auto]\n");
                printf("  --py INT             [auto]\n");
                printf("  --periodic           [false]\n");
                printf("  --steps INT          [500]\n");
                printf("  --dt, --Du, --Dv, --F, --k  [1.0, 0.16, 0.08, 0.06, 0.062]\n");
                printf("  --print-every INT    [50]\n");
                printf("  --seed INT           [0]\n");
                printf("  --seed-mode local|global  [local]\n");
                printf("  --seed-frac LO HI    [0.4 0.6]\n");
                printf("  --viz-every INT      [0=off]\n");
                printf("  --viz-field U|V|both [V]\n");
                printf("  --viz-outdir PATH    [frames]\n");
                printf("  --viz-gif            [off]\n");
            }
            MPI_Finalize();
            exit(0);
        }
    }
}

// -----------------------------------------------
// Indexing helpers for (ny+2) x (nx+2) row-major arrays
// -----------------------------------------------
#define AT(A, nx, i, j)  (A[((i)*(nx+2)) + (j)])   // 0..ny+1, 0..nx+1

// -----------------------------------------------
// Initialization & seeding
// -----------------------------------------------

// Seed a square in local fractional coords (lo,hi) in [0,1]
static void seed_local_fraction(double *U, double *V, int nx, int ny, double lo, double hi) {
    for (int i=1; i<=ny; ++i) {
        for (int j=1; j<=nx; ++j) {
            double y_norm = ( (i-1) + 0.5 ) / (double)ny;
            double x_norm = ( (j-1) + 0.5 ) / (double)nx;
            if (y_norm>lo && y_norm<hi && x_norm>lo && x_norm<hi) {
                AT(U,nx,i,j) = 0.50;
                AT(V,nx,i,j) = 0.25;
            }
        }
    }
}

// Seed a square in global fractional coords (lo,hi) in [0,1]
static void seed_global_fraction(double *U, double *V,
                                 int nx, int ny, int px, int py,
                                 int rx, int ry, double lo, double hi) {
    int NX = px * nx, NY = py * ny;
    int x0 = rx * nx, y0 = ry * ny;
    for (int i=1; i<=ny; ++i) {
        for (int j=1; j<=nx; ++j) {
            double xg = (x0 + (j - 0.5)) / (double)NX;
            double yg = (y0 + (i - 0.5)) / (double)NY;
            if (yg>lo && yg<hi && xg>lo && xg<hi) {
                AT(U,nx,i,j) = 0.50;
                AT(V,nx,i,j) = 0.25;
            }
        }
    }
}

// Initialize U=1, V=0 + small noise + seed square
static void initialize(double *U, double *V, int nx, int ny,
                       int rx, int ry, int px, int py,
                       rng64 *rng, const Config *cfg) {
    // Fill
    for (int i=0; i<ny+2; ++i) {
        for (int j=0; j<nx+2; ++j) {
            AT(U,nx,i,j) = 1.0;
            AT(V,nx,i,j) = 0.0;
        }
    }
    // Add small noise in interior
    for (int i=1; i<=ny; ++i) {
        for (int j=1; j<=nx; ++j) {
            double noise = 0.02 * rng_uniform01(rng);
            AT(U,nx,i,j) -= noise;
            AT(V,nx,i,j) += noise;
        }
    }
    // Seed square
    if (cfg->seed_mode == SEED_LOCAL) {
        seed_local_fraction(U,V,nx,ny,cfg->seed_lo,cfg->seed_hi);
    } else {
        seed_global_fraction(U,V,nx,ny,px,py,rx,ry,cfg->seed_lo,cfg->seed_hi);
    }
}

// -----------------------------------------------
// Neumann BC: copy interior edge into ghost *only* on physical boundaries
// -----------------------------------------------
static void apply_neumann_boundary_ghosts(MPI_Comm cart, double *A, int nx, int ny) {
    int up, down, left, right;
    MPI_Cart_shift(cart, 0, 1, &up, &down);   // Y-dim
    MPI_Cart_shift(cart, 1, 1, &left, &right);// X-dim

    // Top boundary: no UP neighbor
    if (up == MPI_PROC_NULL) {
        for (int j=1; j<=nx; ++j) AT(A,nx,0,j) = AT(A,nx,1,j);
        AT(A,nx,0,0) = AT(A,nx,1,1);
        AT(A,nx,0,nx+1) = AT(A,nx,1,nx);
    }
    // Bottom boundary: no DOWN neighbor
    if (down == MPI_PROC_NULL) {
        for (int j=1; j<=nx; ++j) AT(A,nx,ny+1,j) = AT(A,nx,ny,j);
        AT(A,nx,ny+1,0) = AT(A,nx,ny,1);
        AT(A,nx,ny+1,nx+1) = AT(A,nx,ny,nx);
    }
    // Left boundary
    if (left == MPI_PROC_NULL) {
        for (int i=1; i<=ny; ++i) AT(A,nx,i,0) = AT(A,nx,i,1);
        AT(A,nx,0,0) = AT(A,nx,1,1);
        AT(A,nx,ny+1,0) = AT(A,nx,ny,1);
    }
    // Right boundary
    if (right == MPI_PROC_NULL) {
        for (int i=1; i<=ny; ++i) AT(A,nx,i,nx+1) = AT(A,nx,i,nx);
        AT(A,nx,0,nx+1) = AT(A,nx,1,nx);
        AT(A,nx,ny+1,nx+1) = AT(A,nx,ny,nx);
    }
}

// -----------------------------------------------
// Halo exchange (Sendrecv rows then columns)
// -----------------------------------------------
static void exchange_halo(MPI_Comm cart, double *A, int nx, int ny) {
    int up, down, left, right;
    MPI_Cart_shift(cart, 0, 1, &up, &down);   // Y-dim neighbors
    MPI_Cart_shift(cart, 1, 1, &left, &right);// X-dim neighbors

    MPI_Status status;

    // send last interior row to DOWN, receive into top ghost from UP
    MPI_Sendrecv(&AT(A, nx, ny, 1), nx, MPI_DOUBLE, down, 101,
                 &AT(A, nx, 0, 1), nx, MPI_DOUBLE, up, 101,
                 cart, &status);

    // send first interior row to UP, receive bottom ghost from DOWN
    MPI_Sendrecv(&AT(A, nx, 1, 1), nx, MPI_DOUBLE, up, 102,
                 &AT(A, nx, ny + 1, 1), nx, MPI_DOUBLE, down, 102,
                 cart, &status);

    // Columns: pack/unpack contiguous buffers of length ny
    double *send = (double *)malloc(sizeof(double) * ny);
    double *recv = (double *)malloc(sizeof(double) * ny);

    // pack last interior column
    for (int i = 0; i < ny; ++i) send[i] = AT(A, nx, i+1, nx);

    // send last interior column to RIGHT, receive into first ghost from LEFT
    MPI_Sendrecv(send, ny, MPI_DOUBLE, right, 201,
                 recv, ny, MPI_DOUBLE, left, 201,
                 cart, &status);
    // unpack column and copy to first ghost column
    for (int i = 0; i < ny; ++i) AT(A, nx, i+1, 0) = recv[i];

    // pack first interior column
    for (int i = 0; i < ny; ++i) send[i] = AT(A, nx, i+1, 1);

    // send first interior column to LEFT, receive into last ghost from RIGHT
    MPI_Sendrecv(send, ny, MPI_DOUBLE, left, 202,
                 recv, ny, MPI_DOUBLE, right, 202,
                 cart, &status);

    // unpack column and copy to last ghost column
    for (int i = 0; i < ny; ++i) AT(A, nx, i+1, nx+1) = recv[i];

    free(send);
    free(recv);
}

static void update_ghosts(MPI_Comm cart, double *A, int nx, int ny, int periodic) {
    exchange_halo(cart, A, nx, ny);
    if (!periodic) apply_neumann_boundary_ghosts(cart, A, nx, ny);
}

// -----------------------------------------------
// Gray–Scott update on interior cells
// -----------------------------------------------
static void step_gray_scott(const double *U, const double *V, double *Un, double *Vn,
                            int nx, int ny, double Du, double Dv, double F, double k, double dt) {
    for (int i=1; i<=ny; ++i) {
        for (int j=1; j<=nx; ++j) {
            double Ui = AT(U,nx,i,j);
            double Vi = AT(V,nx,i,j);
            double Lu = (-4.0*Ui
                        + AT(U,nx,i,  j-1)
                        + AT(U,nx,i,  j+1)
                        + AT(U,nx,i-1,j)
                        + AT(U,nx,i+1,j));
            double Lv = (-4.0*Vi
                        + AT(V,nx,i,  j-1)
                        + AT(V,nx,i,  j+1)
                        + AT(V,nx,i-1,j)
                        + AT(V,nx,i+1,j));
            double uvv = Ui * Vi * Vi;
            AT(Un,nx,i,j) = Ui + dt*(Du*Lu - uvv + F*(1.0 - Ui));
            AT(Vn,nx,i,j) = Vi + dt*(Dv*Lv + uvv - (F + k)*Vi);
        }
    }
}

// -----------------------------------------------
// Gather helpers (contiguous interior tiles) -> rank 0 assembles global field
// -----------------------------------------------
static double* gather_global_field(MPI_Comm cart, const double *A, int nx, int ny, int px, int py) {
    int rank, size; MPI_Comm_rank(cart, &rank); MPI_Comm_size(cart, &size);

    // Pack interior into a contiguous tile
    int tile_elems = nx*ny;
    double *tile = (double*)malloc(sizeof(double)*tile_elems);
    for (int i=0; i<ny; ++i) memcpy(tile + i*nx, &AT(A,nx,i+1,1), sizeof(double)*nx);

    // gather tiles
    double *all_tiles = NULL;
    if (rank == 0) all_tiles = (double*)malloc(sizeof(double)*tile_elems*size);
    MPI_Gather(tile, tile_elems, MPI_DOUBLE, all_tiles, tile_elems, MPI_DOUBLE, 0, cart);

    // gather coords (ry, rx) for each rank
    int coords[2]; MPI_Cart_coords(cart, rank, 2, coords);
    int *all_coords = NULL;
    if (rank == 0) all_coords = (int*)malloc(sizeof(int)*2*size);
    MPI_Gather(coords, 2, MPI_INT, all_coords, 2, MPI_INT, 0, cart);

    free(tile);

    if (rank != 0) return NULL;

    // reorder gathered array into grid layout according to coords
    int NX = px*nx, NY = py*ny;
    double *G = (double*)malloc(sizeof(double)*NX*NY);
    // place tiles
    for (int r=0; r<size; ++r) {
        int ry = all_coords[2*r+0];
        int rx = all_coords[2*r+1];
        double *src = all_tiles + r*tile_elems;
        for (int i=0; i<ny; ++i) {
            memcpy(G + (ry*ny + i)*NX + (rx*nx), src + i*nx, sizeof(double)*nx);
        }
    }
    free(all_tiles); free(all_coords);
    return G; // owned by caller (rank 0)
}

// -----------------------------------------------
// Write PGM (8-bit grayscale) for a 2D scalar field
// -----------------------------------------------
static void write_pgm(const char *fname, const double *G, int NX, int NY, double vmin, double vmax) {
    // autoscale
    vmin = 1e300; vmax = -1e300;
    for (int i=0;i<NX*NY;++i) { if (G[i]<vmin) vmin=G[i]; if (G[i]>vmax) vmax=G[i]; }
    if (vmax <= vmin) { vmin = 0.0; vmax = 1.0; }

    FILE *f = fopen(fname, "wb");
    if (!f) { fprintf(stderr, "[viz] failed to open %s\n", fname); return; }
    fprintf(f, "P5\n%d %d\n255\n", NX, NY);
    // map to 0..255, origin lower-left to mimic imshow(origin="lower")
    for (int i=NY-1; i>=0; --i) {
        for (int j=0; j<NX; ++j) {
            double v = G[i*NX + j];
            double t = (v - vmin) / (vmax - vmin);
            if (t<0) t=0; 
            if (t>1) t=1;
            unsigned char b = (unsigned char)(t*255.0 + 0.5);
            fwrite(&b, 1, 1, f);
        }
    }
    fclose(f);
}

static void save_frame(int step, MPI_Comm cart, double *U, double *V,
                       int nx, int ny, int px, int py,
                       int which, const char *outdir,
                       double vmin, double vmax) {
    int rank; MPI_Comm_rank(cart, &rank);
    ensure_dir(outdir);

    double *Ug = NULL, *Vg = NULL;
    if (which == VIZ_U || which == VIZ_BOTH) Ug = gather_global_field(cart, U, nx, ny, px, py);
    if (which == VIZ_V || which == VIZ_BOTH) Vg = gather_global_field(cart, V, nx, ny, px, py);

    if (rank != 0) { if (Ug) free(Ug); if (Vg) free(Vg); return; }

    char path[512];
    int NX = px*nx, NY = py*ny;

    if (which == VIZ_U) {
        snprintf(path, sizeof(path), "%s/step_%06d.pgm", outdir, step);
        write_pgm(path, Ug, NX, NY, vmin, vmax);
    } else if (which == VIZ_V) {
        snprintf(path, sizeof(path), "%s/step_%06d.pgm", outdir, step);
        write_pgm(path, Vg, NX, NY, vmin, vmax);
    } else {
        snprintf(path, sizeof(path), "%s/step_%06d_U.pgm", outdir, step);
        write_pgm(path, Ug, NX, NY, vmin, vmax);
        snprintf(path, sizeof(path), "%s/step_%06d_V.pgm", outdir, step);
        write_pgm(path, Vg, NX, NY, vmin, vmax);
    }
    if (Ug) free(Ug); 
    if (Vg) free(Vg);
}

// -----------------------------------------------
// Main
// -----------------------------------------------
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm world = MPI_COMM_WORLD;
    int rank, size; MPI_Comm_rank(world, &rank); MPI_Comm_size(world, &size);

    Config cfg; parse_args(&cfg, argc, argv, rank);

    // Cart topology (py x px). If px/py=0, let MPI choose via Dims_create.
    int dims[2] = { cfg.py, cfg.px };
    MPI_Dims_create(size, 2, dims); // may overwrite zeros
    cfg.py = dims[0]; cfg.px = dims[1];
    int periods[2] = { cfg.periodic, cfg.periodic };
    MPI_Comm cart; MPI_Cart_create(world, 2, dims, periods, 1, &cart);
    int coords[2]; MPI_Cart_coords(cart, rank, 2, coords);
    int ry = coords[0], rx = coords[1];

    int nx = cfg.nx_local, ny = cfg.ny_local;
    int NX = cfg.px * nx, NY = cfg.py * ny;

    // Allocate fields (ny+2) x (nx+2)
    size_t total = (size_t)(nx+2)*(ny+2);
    double *U  = (double*)malloc(sizeof(double)*total);
    double *V  = (double*)malloc(sizeof(double)*total);
    double *Un = (double*)malloc(sizeof(double)*total);
    double *Vn = (double*)malloc(sizeof(double)*total);
    if (!U||!V||!Un||!Vn) { fprintf(stderr, "[rank %d] alloc failure\n", rank); MPI_Abort(world, 1); }

    rng64 rng; rng_seed(&rng, (unsigned long long)(cfg.seed + rank));
    initialize(U,V,nx,ny,rx,ry,cfg.px,cfg.py,&rng,&cfg);

    // Initial ghost update + optional frame 0
    update_ghosts(cart, U, nx, ny, cfg.periodic);
    update_ghosts(cart, V, nx, ny, cfg.periodic);
    if (cfg.viz_every>0) {
        save_frame(0, cart, U, V, nx, ny, cfg.px, cfg.py,
                   cfg.viz_field, cfg.viz_outdir, cfg.viz_vmin, cfg.viz_vmax);
    }

    double t0 = MPI_Wtime();

    for (int step=0; step<cfg.steps; ++step) {
        double halo_start = MPI_Wtime();
        update_ghosts(cart, U, nx, ny, cfg.periodic);
        update_ghosts(cart, V, nx, ny, cfg.periodic);
        double halo_end = MPI_Wtime();

        double gss_start = MPI_Wtime();
        step_gray_scott(U, V, Un, Vn, nx, ny, cfg.Du, cfg.Dv, cfg.F, cfg.k, cfg.dt);
        double gss_end = MPI_Wtime();

        // swap
        double *tmp = U; U = Un; Un = tmp;
        tmp = V; V = Vn; Vn = tmp;

        if (cfg.viz_every>0 && (step % cfg.viz_every == 0)) {
            save_frame(step, cart, U, V, nx, ny, cfg.px, cfg.py,
                       cfg.viz_field, cfg.viz_outdir, cfg.viz_vmin, cfg.viz_vmax);
        }

        if (cfg.print_every>0 && (step % cfg.print_every == 0)) {
            MPI_Barrier(cart);
            // sum V interior
            double local_sum = 0.0;
            for (int i=1;i<=ny;++i) for (int j=1;j<=nx;++j) local_sum += AT(V,nx,i,j);
            double global_sum = 0.0; MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, cart);
            if (rank == 0) {
                double elapsed = MPI_Wtime() - t0;
                double gss_ms = (gss_end - gss_start)*1e3;
                double halo_ms = (halo_end - halo_start)*1e3;
                printf("[step %6d] ranks=%d grid=%dx%d N=%dx%d local=%dx%d Vsum=%.6e elapsed=%.2fs GSS_time=%.2fms halo_time=%.2fms\n",
                       step, size, cfg.py, cfg.px, NY, NX, ny, nx, global_sum, elapsed, gss_ms, halo_ms);
                fflush(stdout);
            }
        }
    }

    MPI_Barrier(cart);
    if (rank == 0) {
        if (cfg.viz_gif && cfg.viz_every>0) {
            printf("[viz] Frames written to %s (PGM).\n", cfg.viz_outdir);
            printf("[viz] To assemble a GIF: convert %s/step_*.pgm out.gif\n", cfg.viz_outdir);
        }
        printf("DONE.\n");
    }

    free(U); free(V); free(Un); free(Vn);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
