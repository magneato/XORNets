; ═══════════════════════════════════════════════════════════════════════════
;  xn.inc ─ the cortex ABI.  Every byte here is mirrored in sdk/include/xn.h.
;  Change one, change both.  The C++ tests static_assert the offsets.
; ═══════════════════════════════════════════════════════════════════════════

%ifndef XN_INC
%define XN_INC

XN_ABI_VERSION  equ 1
XN_MAGIC        equ 'XN01'

XN_MAX_LAYERS   equ 8
XN_MAX_NEURONS  equ 64
XN_MAX_INPUTS   equ 6               ; 2^6 = 64 rows: the whole dataset fits in one register

; ── layer: one rank of XOR cells ─────────────────────────────────────────────
struc layer
    .neurons    resd 1              ; cells in this layer (1..64)
    .fan_in     resd 1              ; width of the incoming bit-vector
    .nw         resd 1              ; weights supplied by the .xn file
    .nt         resd 1              ; thresholds supplied by the .xn file
    .w          resq XN_MAX_NEURONS ; cell n fires when popcnt((x ⊕ w[n]) & fan) >= t[n]
    .t          resb XN_MAX_NEURONS
endstruc
XN_LAYER_SIZE   equ layer_size                      ; 592
XN_LAYERS_BYTES equ XN_LAYER_SIZE * XN_MAX_LAYERS   ; 4736

; ── model: the whole organism ────────────────────────────────────────────────
struc model
    .magic      resd 1
    .version    resd 1
    .inputs     resd 1
    .nlayers    resd 1
    .truth      resq 1              ; target truth word: bit x = f(x)
    .seed       resq 1
    .epochs     resq 1              ; mutation steps per restart
    .restarts   resd 1
    .verbose    resd 1
    .flags      resd 1
    .err_line   resd 1
    .target     resd 1              ; XN_T_*
    .used       resd 1              ; restarts consumed by the last train
    .loss       resq 1
    .steps      resq 1
    .name       resb 32
    .layers     resb XN_LAYERS_BYTES
endstruc
XN_MODEL_SIZE   equ model_size                      ; 4848

; ── swarm: the tournament ledger shared by every agent thread ────────────────
;  An agent is nothing but its id: seed = splitmix64(model.seed, id).  It lives
;  one life (one restart, model.epochs mutations) and only the champion's
;  (loss, mutations, id) survives.  Lower is better, compared in that order.
struc swarm
    .agents     resq 1              ; in: agents to spawn (0 → 1)
    .threads    resd 1              ; in: 0 → every cpu we may run on; out: used
    .lock       resd 1              ; internal spinlock
    .next       resq 1              ; internal: next agent id to hand out
    .model      resq 1              ; internal: the base organism
    .best_id    resq 1
    .best_loss  resq 1
    .best_steps resq 1
    .contact    resq 1              ; agents that reached loss 0
    .steps      resq 1              ; mutations across the whole swarm
    .ns         resq 1              ; wall time
endstruc
XN_SWARM_SIZE   equ swarm_size                      ; 80
XN_MAX_THREADS  equ 256

; flags
XN_F_WEIGHTS    equ 1               ; every layer carries weights + thresholds
XN_F_TRAINED    equ 2               ; xn_train has run

; targets
XN_T_NONE       equ 0
XN_T_XOR        equ 1
XN_T_XNOR       equ 2
XN_T_AND        equ 3
XN_T_OR         equ 4
XN_T_NAND       equ 5
XN_T_NOR        equ 6
XN_T_TRUTH      equ 7

; errors
XN_OK           equ 0
XN_EIO          equ 1
XN_ESYNTAX      equ 2
XN_ESECTION     equ 3
XN_EKEY         equ 4
XN_EVALUE       equ 5
XN_ELIMIT       equ 6
XN_ENOLAYERS    equ 7
XN_EOUTPUT      equ 8
XN_ETARGET      equ 9
XN_EINPUTS      equ 10
XN_EWEIGHTS     equ 11
XN_EARG         equ 12
XN_ECOUNT       equ 13

; linux x86-64 syscalls
SYS_read        equ 0
SYS_write       equ 1
SYS_open        equ 2
SYS_close       equ 3
SYS_fstat       equ 5
SYS_mmap        equ 9
SYS_munmap      equ 11
SYS_exit        equ 60
SYS_clone       equ 56
SYS_futex       equ 202
SYS_getaffinity equ 204
SYS_clock_gettime equ 228

O_RDONLY        equ 0
PROT_READ       equ 1
MAP_PRIVATE     equ 2
MAP_ANONYMOUS   equ 0x20
MAP_STACK       equ 0x20000
PROT_RW         equ 3
STAT_SIZE       equ 144
STAT_ST_SIZE    equ 48

%macro NOEXECSTACK 0
    section .note.GNU-stack noalloc noexec nowrite progbits
%endmacro

%endif
