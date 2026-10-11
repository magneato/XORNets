; ═══════════════════════════════════════════════════════════════════════════
;  agents.asm ─ axn: a swarm of tiny cortexes, and only the best survives.
;
;  No agent is ever stored.  Agent i *is* its seed,
;
;      seed(i) = splitmix64(model.seed + (i + 1) · φ)
;
;  so spawning one costs nothing.  It wakes inside a per-thread scratch copy
;  of the organism, lives a single life (one restart, model.epochs mutations
;  of stochastic bit-flip search), and is judged by (loss, mutations, id).
;  Then the next agent wakes in the same bytes.  A billion agents need no
;  more memory than one.
;
;  Workers are raw kernel threads: clone(CLONE_VM|CLONE_THREAD|…) onto an
;  mmap'd stack, agents handed out in chunks by lock xadd, joined on the
;  kernel-cleared tid word with futex.  The calling thread works too.
;  The ordering is total, so the champion is the same for any thread count.
;  Afterwards the champion is re-lived from its seed into the caller's model:
;  an agent is reproducible from nothing but its number.
; ═══════════════════════════════════════════════════════════════════════════
%include "xn.inc"
default rel

extern xn_train

global xn_swarm_run, xn_agent_seed, xn_cpus, xn_clock_ns
global xn_swarm_size, xn_agent_footprint, xn_swarm_stack

CHUNK           equ 64              ; agents per grab from the shared counter
STACK_SIZE      equ 131072          ; per spawned thread
CLONE_FLAGS     equ 0x100 | 0x200 | 0x400 | 0x800 | 0x10000 | 0x40000 | 0x100000 | 0x200000
                ;   VM     FS      FILES   SIGHAND THREAD   SYSVSEM  PARENT_SETTID CHILD_CLEARTID
FUTEX_WAIT      equ 0
CLOCK_MONOTONIC equ 1

; worker frame: the scratch organism, then a few locals
W_STEPS         equ XN_MODEL_SIZE
W_CONTACT       equ XN_MODEL_SIZE + 8
W_END           equ XN_MODEL_SIZE + 16
W_FRAME         equ XN_MODEL_SIZE + 40             ; ≡ 8 mod 16: aligned after 6 pushes

; peak stack of one live agent: worker frame + xn_train frame + the leaf calls
AGENT_FOOTPRINT equ (W_FRAME + 56) + (XN_LAYERS_BYTES + 24 + 56) + 128

; launcher frame: one stack base per spawned thread
L_FRAME         equ 8 * XN_MAX_THREADS + 8

section .text

xn_swarm_size:
    mov     eax, XN_SWARM_SIZE
    ret

xn_swarm_stack:
    mov     eax, STACK_SIZE
    ret

xn_agent_footprint:
    mov     eax, AGENT_FOOTPRINT
    ret

; xn_agent_seed(rdi base seed, rsi id) -> rax   splitmix64: the agent's genome
xn_agent_seed:
    lea     rax, [rsi + 1]
    mov     rdx, 0x9E3779B97F4A7C15
    imul    rax, rdx
    add     rax, rdi
    mov     rdx, rax
    shr     rdx, 30
    xor     rax, rdx
    mov     rdx, 0xBF58476D1CE4E5B9
    imul    rax, rdx
    mov     rdx, rax
    shr     rdx, 27
    xor     rax, rdx
    mov     rdx, 0x94D049BB133111EB
    imul    rax, rdx
    mov     rdx, rax
    shr     rdx, 31
    xor     rax, rdx
    ret

; xn_cpus() -> eax   cpus in our affinity mask (at least 1)
xn_cpus:
    sub     rsp, 136
    xor     edi, edi
    mov     esi, 128
    mov     rdx, rsp
    mov     eax, SYS_getaffinity
    syscall
    xor     r8d, r8d
    test    rax, rax
    jle     .one
    xor     edx, edx
.word:
    cmp     rdx, rax
    jae     .counted
    popcnt  r9, [rsp + rdx]
    add     r8, r9
    add     rdx, 8
    jmp     .word
.counted:
    test    r8d, r8d
    jnz     .out
.one:
    mov     r8d, 1
.out:
    mov     eax, r8d
    add     rsp, 136
    ret

; xn_clock_ns() -> rax   monotonic nanoseconds
xn_clock_ns:
    sub     rsp, 24
    mov     edi, CLOCK_MONOTONIC
    mov     rsi, rsp
    mov     eax, SYS_clock_gettime
    syscall
    imul    rax, [rsp], 1000000000
    add     rax, [rsp + 8]
    add     rsp, 24
    ret

; ── one worker: wake agents until the counter runs dry ──────────────────────
; worker(rdi *swarm)
worker:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, W_FRAME
    mov     rbx, rdi
    mov     rsi, [rbx + swarm.model]
    mov     rdi, rsp
    mov     ecx, XN_MODEL_SIZE
    rep     movsb                   ; the scratch organism every agent inhabits
    mov     dword [rsp + model.restarts], 1     ; one life each
    xor     eax, eax
    mov     [rsp + W_STEPS], rax
    mov     [rsp + W_CONTACT], rax
    mov     r12, -1                 ; local champion: loss
    mov     r13, -1                 ;                 mutations
    mov     r14, -1                 ;                 id

.grab:
    mov     eax, CHUNK
    lock xadd [rbx + swarm.next], rax
    mov     rcx, [rbx + swarm.agents]
    cmp     rax, rcx
    jae     .dry
    lea     rdx, [rax + CHUNK]
    cmp     rdx, rcx
    cmova   rdx, rcx
    mov     [rsp + W_END], rdx
    mov     r15, rax                ; agent id

.agent:
    cmp     r15, [rsp + W_END]
    jae     .grab
    mov     rax, [rbx + swarm.model]
    mov     rdi, [rax + model.seed]
    mov     rsi, r15
    call    xn_agent_seed
    mov     [rsp + model.seed], rax
    mov     rdi, rsp
    call    xn_train                ; the agent's whole life
    mov     rbp, [rsp + model.steps]
    add     [rsp + W_STEPS], rbp
    test    rax, rax
    jnz     .judge
    inc     qword [rsp + W_CONTACT]
.judge:                             ; ids only grow here, so ties keep the elder
    cmp     rax, r12
    jb      .crown
    ja      .next
    cmp     rbp, r13
    jae     .next
.crown:
    mov     r12, rax
    mov     r13, rbp
    mov     r14, r15
.next:
    inc     r15
    jmp     .agent

.dry:
    mov     rax, [rsp + W_STEPS]
    lock add [rbx + swarm.steps], rax
    mov     rax, [rsp + W_CONTACT]
    lock add [rbx + swarm.contact], rax
.spin:
    mov     eax, 1
    xchg    eax, [rbx + swarm.lock]
    test    eax, eax
    jz      .held
    pause
    jmp     .spin
.held:
    cmp     r12, [rbx + swarm.best_loss]
    jb      .win
    ja      .release
    cmp     r13, [rbx + swarm.best_steps]
    jb      .win
    ja      .release
    cmp     r14, [rbx + swarm.best_id]
    jae     .release
.win:
    mov     [rbx + swarm.best_loss], r12
    mov     [rbx + swarm.best_steps], r13
    mov     [rbx + swarm.best_id], r14
.release:
    mov     dword [rbx + swarm.lock], 0         ; a plain store releases on x86
    add     rsp, W_FRAME
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; a spawned thread begins here, on its own stack: [rsp] = *swarm
thread_entry:
    pop     rdi
    sub     rsp, 8
    call    worker
    xor     edi, edi
    mov     eax, SYS_exit           ; this thread only; the kernel clears our tid
    syscall

; ── xn_swarm_run(rdi *model, rsi *swarm) -> rax champion loss ───────────────
;   on return the model holds the champion, re-lived from its seed
xn_swarm_run:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, L_FRAME
    mov     rbp, rdi
    mov     rbx, rsi
    mov     [rbx + swarm.model], rbp

    mov     rax, [rbx + swarm.agents]
    test    rax, rax
    jnz     .agents_ok
    mov     eax, 1
    mov     [rbx + swarm.agents], rax
.agents_ok:
    mov     eax, [rbx + swarm.threads]
    test    eax, eax
    jnz     .threads_ok
    call    xn_cpus
.threads_ok:
    cmp     eax, XN_MAX_THREADS
    jbe     .cap1
    mov     eax, XN_MAX_THREADS
.cap1:
    cmp     rax, [rbx + swarm.agents]
    jbe     .cap2
    mov     eax, [rbx + swarm.agents]   ; never more threads than agents
.cap2:
    mov     [rbx + swarm.threads], eax

    xor     eax, eax
    mov     [rbx + swarm.lock], eax
    mov     [rbx + swarm.next], rax
    mov     [rbx + swarm.contact], rax
    mov     [rbx + swarm.steps], rax
    dec     rax
    mov     [rbx + swarm.best_id], rax
    mov     [rbx + swarm.best_loss], rax
    mov     [rbx + swarm.best_steps], rax

    call    xn_clock_ns
    mov     r15, rax                ; t0

    mov     r12d, [rbx + swarm.threads]
    dec     r12d                    ; threads to spawn; the caller is one of them
    xor     r13d, r13d              ; spawned
.spawn:
    cmp     r13d, r12d
    jae     .spawned
    xor     edi, edi
    mov     esi, STACK_SIZE
    mov     edx, PROT_RW
    mov     r10d, MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK
    mov     r8, -1
    xor     r9d, r9d
    mov     eax, SYS_mmap
    syscall
    cmp     rax, -4096
    ja      .spawned                ; out of memory: fight with who we have
    mov     r14, rax
    mov     [rsp + r13 * 8], r14
    mov     dword [r14], -1         ; tid word lives at the stack's floor
    lea     rsi, [r14 + STACK_SIZE - 16]
    mov     [rsi], rbx
    mov     edi, CLONE_FLAGS
    mov     rdx, r14                ; parent tid
    mov     r10, r14                ; child tid, cleared + futex-woken at exit
    xor     r8d, r8d
    mov     eax, SYS_clone
    syscall
    test    rax, rax
    jz      thread_entry            ; ← the child wakes up here
    js      .clone_failed
    inc     r13d
    jmp     .spawn
.clone_failed:
    mov     rdi, r14
    mov     esi, STACK_SIZE
    mov     eax, SYS_munmap
    syscall
.spawned:
    lea     eax, [r13d + 1]
    mov     [rbx + swarm.threads], eax

    mov     rdi, rbx
    call    worker                  ; the caller joins the swarm

    xor     r12d, r12d
.join:
    cmp     r12d, r13d
    jae     .joined
    mov     r14, [rsp + r12 * 8]
.wait:
    mov     edx, [r14]
    test    edx, edx
    jz      .reap
    mov     rdi, r14
    mov     esi, FUTEX_WAIT
    xor     r10d, r10d
    mov     eax, SYS_futex
    syscall
    jmp     .wait
.reap:
    mov     rdi, r14
    mov     esi, STACK_SIZE
    mov     eax, SYS_munmap
    syscall
    inc     r12d
    jmp     .join

.joined:
    call    xn_clock_ns
    sub     rax, r15
    mov     [rbx + swarm.ns], rax

    ; re-live the champion in the caller's organism, then restore its options
    mov     r12, [rbp + model.seed]
    mov     r13d, [rbp + model.restarts]
    mov     rdi, r12
    mov     rsi, [rbx + swarm.best_id]
    call    xn_agent_seed
    mov     [rbp + model.seed], rax
    mov     dword [rbp + model.restarts], 1
    mov     rdi, rbp
    call    xn_train
    mov     [rbp + model.seed], r12
    mov     [rbp + model.restarts], r13d
    mov     rax, [rbx + swarm.best_loss]

    add     rsp, L_FRAME
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

NOEXECSTACK
