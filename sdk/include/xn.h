/* ═══════════════════════════════════════════════════════════════════════════
 *  xn.h ─ the cortex ABI.  C linkage, System V x86-64, no allocation.
 *
 *  Mirrors engine/xn.inc byte for byte.  Link against build/libxn.a.
 *  Every routine is reentrant except the xn_io_* transmitter (and xn_emit,
 *  which uses it): one shared output buffer per process.
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifndef XN_H
#define XN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define XN_ABI_VERSION  1
#define XN_MAX_LAYERS   8
#define XN_MAX_NEURONS  64
#define XN_MAX_INPUTS   6

enum xn_flag   { XN_F_WEIGHTS = 1, XN_F_TRAINED = 2 };
enum xn_target { XN_T_NONE, XN_T_XOR, XN_T_XNOR, XN_T_AND, XN_T_OR,
                 XN_T_NAND, XN_T_NOR, XN_T_TRUTH };
enum xn_status { XN_OK, XN_EIO, XN_ESYNTAX, XN_ESECTION, XN_EKEY, XN_EVALUE,
                 XN_ELIMIT, XN_ENOLAYERS, XN_EOUTPUT, XN_ETARGET, XN_EINPUTS,
                 XN_EWEIGHTS, XN_EARG };

/* one rank of XOR cells: cell n fires when popcnt((x ^ w[n]) & fan) >= t[n] */
typedef struct xn_layer {
    uint32_t neurons;
    uint32_t fan_in;
    uint32_t nw;                        /* weights given by the .xn file    */
    uint32_t nt;                        /* thresholds given by the .xn file */
    uint64_t w[XN_MAX_NEURONS];
    uint8_t  t[XN_MAX_NEURONS];
} xn_layer;

typedef struct xn_model {
    uint32_t magic;                     /* 'XN01' */
    uint32_t version;
    uint32_t inputs;
    uint32_t nlayers;
    uint64_t truth;                     /* target truth word, bit x = f(x) */
    uint64_t seed;
    uint64_t epochs;                    /* mutations per restart */
    uint32_t restarts;
    uint32_t verbose;
    uint32_t flags;                     /* xn_flag */
    uint32_t err_line;                  /* 1-based line of a parse error, 0 = whole model */
    uint32_t target;                    /* xn_target */
    uint32_t used;                      /* restarts consumed by the last xn_train */
    uint64_t loss;
    uint64_t steps;
    char     name[32];
    xn_layer layers[XN_MAX_LAYERS];
} xn_model;

/* ── lifecycle ─────────────────────────────────────────────────────────── */
void        xn_model_init(xn_model *m);
int         xn_parse(xn_model *m, const char *text, size_t len);
int         xn_load(xn_model *m, const char *path);
int         xn_emit(const xn_model *m, int fd);
int         xn_option_set(xn_model *m, const char *key, size_t keylen,
                          const char *val, size_t vallen);

/* ── learning and listening ────────────────────────────────────────────── */
uint64_t    xn_train(xn_model *m);                  /* -> best loss       */
uint64_t    xn_forward(const xn_model *m, uint64_t row); /* -> 0 | 1      */
uint64_t    xn_truth(const xn_model *m);            /* cortex truth word  */
uint64_t    xn_loss(const xn_model *m);             /* rows that disagree */
uint64_t    xn_layer_eval(const xn_layer *l, uint64_t v);

/* ── the primitive ─────────────────────────────────────────────────────── */
uint64_t    xn_cell(uint64_t x, uint64_t w, uint64_t fan, uint32_t t);
uint64_t    xn_parity(uint64_t x);
uint64_t    xn_hamming(uint64_t a, uint64_t b);
uint64_t    xn_fanmask(uint32_t k);
uint64_t    xn_rowmask(uint32_t inputs);
uint64_t    xn_rng_next(uint64_t *state);
uint64_t    xn_rng_below(uint64_t *state, uint64_t n);
uint64_t    xn_target_truth(uint32_t kind, uint32_t inputs);

/* ── text ──────────────────────────────────────────────────────────────── */
int         xn_text_eq(const char *p, size_t len, const char *cstr);

/* ── axn: swarms of agents ─────────────────────────────────────────────────
 * An agent is its id: seed = xn_agent_seed(model.seed, id).  Each lives one
 * life (one restart, model.epochs mutations) on raw kernel threads; only the
 * champion, ordered by (loss, mutations, id), survives and is re-lived into
 * the model.  Results do not depend on the thread count.                   */
#define XN_MAX_THREADS 256

typedef struct xn_swarm {
    uint64_t agents;                    /* in: agents to spawn (0 → 1)              */
    uint32_t threads;                   /* in: 0 → every cpu; out: threads used     */
    uint32_t lock_;                     /* internal                                 */
    uint64_t next_;                     /* internal                                 */
    const xn_model *model_;             /* internal                                 */
    uint64_t best_id;
    uint64_t best_loss;
    uint64_t best_steps;                /* the champion's mutations                 */
    uint64_t contact;                   /* agents that reached loss 0               */
    uint64_t steps;                     /* mutations across the swarm               */
    uint64_t ns;                        /* wall time                                */
} xn_swarm;

uint64_t    xn_swarm_run(xn_model *m, xn_swarm *s);    /* -> champion loss */
uint64_t    xn_agent_seed(uint64_t base_seed, uint64_t id);
uint32_t    xn_cpus(void);
uint64_t    xn_clock_ns(void);

/* ── identity ──────────────────────────────────────────────────────────── */
const char *xn_strerror(int status);
const char *xn_version(void);
uint32_t    xn_model_size(void);
uint32_t    xn_abi_version(void);
uint32_t    xn_swarm_size(void);
uint32_t    xn_swarm_stack(void);       /* bytes of stack per spawned thread      */
uint32_t    xn_agent_footprint(void);   /* peak stack bytes of one living agent   */
uint32_t    xn_io_capacity(void);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus)
static_assert(sizeof(xn_layer) == 592, "xn_layer drifted from engine/xn.inc");
static_assert(sizeof(xn_model) == 4848, "xn_model drifted from engine/xn.inc");
static_assert(sizeof(xn_swarm) == 80, "xn_swarm drifted from engine/xn.inc");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(xn_layer) == 592, "xn_layer drifted from engine/xn.inc");
_Static_assert(sizeof(xn_model) == 4848, "xn_model drifted from engine/xn.inc");
_Static_assert(sizeof(xn_swarm) == 80, "xn_swarm drifted from engine/xn.inc");
#endif

#endif /* XN_H */
