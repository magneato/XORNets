; ═══════════════════════════════════════════════════════════════════════════
;  xn.asm ─ the console.  _start, argv, the report, the prompt.  No libc.
;
;      xn --model models/first-contact.xn
;      xn --help --model models/first-contact.xn
;      xn --model m.xn --infer 10
;      xn --model m.xn --emit > m.trained.xn
;      xn --model m.xn --prompt
;
;  exit codes: 0 signal acquired · 1 trained but loss > 0 · 2 bad command
;              line · 3 the .xn file was rejected
; ═══════════════════════════════════════════════════════════════════════════
%include "xn.inc"
default rel

extern xn_load, xn_emit, xn_train, xn_forward, xn_truth, xn_option_set
extern xn_strerror, xn_version, xn_rowmask
extern xn_io_flush, xn_io_putn, xn_io_puts, xn_io_putc, xn_io_dec, xn_io_hex
extern xn_io_bits, xn_io_glyphs, xn_io_spaces, xn_io_strlen, xn_io_capacity
extern xn_swarm_run, xn_agent_seed, xn_cpus, xn_swarm_size, xn_agent_footprint, xn_swarm_stack
extern xn_model_size, xn_text_u64

global _start

F_HELP      equ 1
F_VERSION   equ 2
F_EMIT      equ 4
F_PROMPT    equ 8
F_TRAIN     equ 16
F_QUIET     equ 32
F_STATS     equ 64
MAX_OV      equ 8
LINE_MAX    equ 255

section .rodata
a_help:     db "--help", 0
a_h:        db "-h", 0
a_version:  db "--version", 0
a_model:    db "--model", 0
a_infer:    db "--infer", 0
a_emit:     db "--emit", 0
a_prompt:   db "--prompt", 0
a_train:    db "--train", 0
a_quiet:    db "--quiet", 0
a_seed:     db "--seed", 0
a_epochs:   db "--epochs", 0
a_restarts: db "--restarts", 0
a_verbose:  db "--verbose", 0
a_agents:   db "--agents", 0
a_threads:  db "--threads", 0
a_stats:    db "--stats", 0
w_q:        db "q", 0
w_quit:     db "quit", 0
w_exit:     db "exit", 0

target_names:
    db "none", 0, "xor", 0, "xnor", 0, "and", 0, "or", 0, "nand", 0, "nor", 0, "truth", 0

help_text:
    db 10
    db "  xn ─ (x)or (n)ets · a 1-bit cortex that only speaks ⊕", 10
    db 10
    db "  usage   xn --model FILE.xn [options]", 10
    db 10
    db "    --model FILE   load model, layers and options from a .xn file", 10
    db "    --infer BITS   evaluate one row, e.g. --infer 10  (row bits, msb first)", 10
    db "    --prompt       open the console: type rows, read the cortex's reply", 10
    db "    --emit         print the trained organism as .xn, weights included", 10
    db "    --train        retrain even if the file already carries weights", 10
    db "    --seed N       override [options] seed", 10
    db "    --epochs N     override [options] epochs   (mutations per restart)", 10
    db "    --restarts N   override [options] restarts", 10
    db "    --verbose N    override [options] verbose  (0 hides the anatomy)", 10
    db "    --agents N     axn: spawn N agents, one life each, keep the champion", 10
    db "    --threads N    swarm threads (default: every cpu we may use)", 10
    db "    --stats        the size of every object in the system, then exit", 10
    db "    --quiet        essentials only", 10
    db "    --version      print the version", 10
    db "    --help         this transmission (add --model to describe a model)", 10
    db 10
    db "  cell    y = popcnt((x ⊕ w) & fan) >= t      learning: bit-flip search", 10
    db "  exit    0 signal · 1 noise · 2 bad arguments · 3 rejected .xn", 10
    db 0

banner:
    db 10
    db "  ⊕━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⊕", 10
    db "     x n  ·  (x)or (n)ets  ·  1-bit cortex  ·  ", 0
banner2:
    db 10
    db "  ⊕━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⊕", 10, 0

this_model: db 10, "  ── this model ──", 10, 0
m_model:    db "  model    ", 0
m_inputs:   db 10, "  inputs   ", 0
m_rows:     db " · rows ", 0
m_target:   db " · target ", 0
m_cortex:   db 10, "  cortex   ", 0
m_arrow:    db " → ", 0
m_options:  db 10, "  options  seed ", 0
m_epochs:   db " · epochs ", 0
m_restarts: db " · restarts ", 0
m_verbose:  db " · verbose ", 0
m_wfile:    db 10, "  weights  carried in the file", 10, 0
m_wlearn:   db 10, "  weights  to be learned", 10, 0
m_open:     db " (", 0
m_close:    db ")", 0

r_train:    db 10, "  ◇ listening · stochastic bit-flip search", 10, 0
r_skip:     db 10, "  ◇ weights carried in the file · training skipped", 10, 0
r_restart:  db "    restarts ", 0
r_steps:    db " · mutations ", 0
r_layer:    db 10, "  layer ", 0
r_fanin:    db " · fan-in ", 0
r_cells:    db " · cells ", 0
r_n:        db "    n", 0
r_w:        db "  w ", 0
r_t:        db "  t ", 0
r_table:    db 10, "  row", 0
r_tcols:    db "target  cortex", 10, 0
r_ok:       db "✓", 10, 0
r_bad:      db "✗", 10, 0
r_loss:     db 10, "  loss ", 0
r_slash:    db "/", 0
r_signal:   db "  →  SIGNAL ACQUIRED ⊕ reply ", 0
r_noise:    db "  →  NOISE · contact not yet established", 10, 0

c_open1:    db 10, "  ⊕ console open · speak ", 0
c_open2:    db " bits (e.g. ", 0
c_open3:    db ") · q closes the channel", 10, 0
c_prompt:   db "xn⊕ ▸ ", 0
c_reply:    db "      → ", 0
c_target:   db "   target ", 0
c_huh1:     db "      ? the cortex hears only ", 0
c_huh2:     db " bits of 0/1", 10, 0
c_bye:      db "  ⊕ channel closed", 10, 0

i_head:     db "xn ⊕ ", 0

e_xn:       db "xn: ", 0
e_colon:    db ": ", 0
e_unknown:  db "unknown option '", 0
e_missing:  db "missing value for ", 0
e_try:      db "' (try --help)", 10, 0
e_nomodel:  db "no --model given (try --help)", 10, 0
e_many:     db "too many overrides", 10, 0
e_number:   db ": wants a positive number (threads: 1..256)", 10, 0
e_bits1:    db "--infer needs ", 0
e_bits2:    db " bits of 0/1, msb first", 10, 0
s_nl:       db 10, 0
s_sp2:      db "  ", 0

%macro PUTS 1
    lea     rdi, [%1]
    call    xn_io_puts
%endmacro

%macro ARGIS 2
    mov     rdi, rbx
    lea     rsi, [%1]
    call    cstreq
    test    eax, eax
    jnz     %2
%endmacro

section .bss
alignb 64
the_model:  resb XN_MODEL_SIZE
linebuf:    resb LINE_MAX + 1
opt_model:  resq 1
opt_infer:  resq 1
opt_flags:  resd 1
opt_threads: resd 1
opt_agents: resq 1
alignb 16
the_swarm:  resb XN_SWARM_SIZE
ov_count:   resd 1
ov_key:     resq MAX_OV
ov_val:     resq MAX_OV

section .text

_start:
    mov     rdi, [rsp]              ; argc
    lea     rsi, [rsp + 8]          ; argv
    and     rsp, -16
    call    main
    mov     ebx, eax
    call    xn_io_flush
    mov     edi, ebx
    mov     eax, SYS_exit
    syscall

; cstreq(rdi a, rsi b) -> eax 1 if equal
cstreq:
    mov     al, [rdi]
    cmp     al, [rsi]
    jne     .no
    test    al, al
    jz      .yes
    inc     rdi
    inc     rsi
    jmp     cstreq
.yes:
    mov     eax, 1
    ret
.no:
    xor     eax, eax
    ret

; nth_str(rdi base, esi n) -> rax   n-th string of a NUL-separated pack
nth_str:
    mov     rax, rdi
.skip:
    test    esi, esi
    jz      .done
.chr:
    inc     rax
    cmp     byte [rax - 1], 0
    jne     .chr
    dec     esi
    jmp     .skip
.done:
    ret

; parse_bits(rdi ptr, rsi len, edx n) -> rax value, edx 0 ok | 1 bad
parse_bits:
    mov     ecx, edx
    cmp     rsi, rcx
    jne     .bad
    xor     eax, eax
.bit:
    movzx   ecx, byte [rdi]
    sub     ecx, '0'
    cmp     ecx, 1
    ja      .bad
    shl     rax, 1
    or      rax, rcx
    inc     rdi
    dec     rsi
    jnz     .bit
    xor     edx, edx
    ret
.bad:
    mov     edx, 1
    ret

; ── main(edi argc, rsi argv) -> eax exit code ───────────────────────────────
main:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8
    mov     r12d, edi
    mov     r13, rsi
    mov     r14d, 1

.arg:
    cmp     r14d, r12d
    jae     .args_done
    mov     rbx, [r13 + r14 * 8]
    ARGIS   a_help, .a_help
    ARGIS   a_h, .a_help
    ARGIS   a_version, .a_version
    ARGIS   a_model, .a_model
    ARGIS   a_infer, .a_infer
    ARGIS   a_emit, .a_emit
    ARGIS   a_prompt, .a_prompt
    ARGIS   a_train, .a_train
    ARGIS   a_quiet, .a_quiet
    ARGIS   a_seed, .a_override
    ARGIS   a_epochs, .a_override
    ARGIS   a_restarts, .a_override
    ARGIS   a_verbose, .a_override
    ARGIS   a_agents, .a_agents
    ARGIS   a_threads, .a_threads
    ARGIS   a_stats, .a_stats
    PUTS    e_xn
    PUTS    e_unknown
    mov     rdi, rbx
    call    xn_io_puts
    PUTS    e_try
    mov     eax, 2
    jmp     .ret

.a_help:
    or      dword [opt_flags], F_HELP
    jmp     .next
.a_version:
    or      dword [opt_flags], F_VERSION
    jmp     .next
.a_emit:
    or      dword [opt_flags], F_EMIT
    jmp     .next
.a_prompt:
    or      dword [opt_flags], F_PROMPT
    jmp     .next
.a_train:
    or      dword [opt_flags], F_TRAIN
    jmp     .next
.a_quiet:
    or      dword [opt_flags], F_QUIET
    jmp     .next
.a_stats:
    or      dword [opt_flags], F_STATS
    jmp     .next
.a_agents:
    call    .number
    test    rax, rax
    jz      .bad_number
    mov     [opt_agents], rax
    jmp     .next
.a_threads:
    call    .number
    test    rax, rax
    jz      .bad_number
    cmp     rax, XN_MAX_THREADS
    ja      .bad_number
    mov     [opt_threads], eax
    jmp     .next

; local: the next argument as a number (decimal, 0x, 0b, '_' groups); bail on junk
.number:
    inc     r14d
    cmp     r14d, r12d
    jae     .number_missing
    mov     rax, [r13 + r14 * 8]
    push    rax
    mov     rdi, rax
    call    xn_io_strlen
    pop     rdi
    mov     rsi, rax
    call    xn_text_u64
    test    edx, edx
    jnz     .bad_number_frame
    ret
.number_missing:
    add     rsp, 8
    jmp     .missing_body
.bad_number_frame:
    add     rsp, 8
.bad_number:
    PUTS    e_xn
    mov     rdi, rbx
    call    xn_io_puts
    PUTS    e_number
    mov     eax, 2
    jmp     .ret
.a_model:
    call    .value
    mov     [opt_model], rax
    jmp     .next
.a_infer:
    call    .value
    mov     [opt_infer], rax
    jmp     .next
.a_override:
    call    .value
    mov     ecx, [ov_count]
    cmp     ecx, MAX_OV
    jae     .too_many
    lea     rdx, [ov_val]
    mov     [rdx + rcx * 8], rax
    lea     rdx, [ov_key]
    lea     rax, [rbx + 2]          ; "--seed" → "seed"
    mov     [rdx + rcx * 8], rax
    inc     dword [ov_count]
    jmp     .next
.next:
    inc     r14d
    jmp     .arg

; local: fetch the argument after rbx, or bail with exit 2
.value:
    inc     r14d
    cmp     r14d, r12d
    jae     .missing
    mov     rax, [r13 + r14 * 8]
    ret
.missing:
    add     rsp, 8
.missing_body:
    PUTS    e_xn
    PUTS    e_missing
    mov     rdi, rbx
    call    xn_io_puts
    PUTS    s_nl
    mov     eax, 2
    jmp     .ret
.too_many:
    PUTS    e_xn
    PUTS    e_many
    mov     eax, 2
    jmp     .ret

.args_done:
    test    dword [opt_flags], F_VERSION
    jz      .no_version
    call    xn_version
    mov     rdi, rax
    call    xn_io_puts
    PUTS    s_nl
    xor     eax, eax
    jmp     .ret
.no_version:
    test    dword [opt_flags], F_STATS
    jz      .no_stats
    cmp     qword [opt_model], 0
    jne     .load
    xor     edi, edi
    call    stats
    xor     eax, eax
    jmp     .ret
.no_stats:
    cmp     qword [opt_model], 0
    jne     .load
    test    dword [opt_flags], F_HELP
    jnz     .help_only
    cmp     r12d, 1
    je      .help_only
    PUTS    e_xn
    PUTS    e_nomodel
    mov     eax, 2
    jmp     .ret
.help_only:
    PUTS    help_text
    xor     eax, eax
    jmp     .ret

; ── load the organism ──
.load:
    lea     rdi, [the_model]
    mov     rsi, [opt_model]
    call    xn_load
    test    eax, eax
    jz      .loaded
    mov     ebx, eax
    PUTS    e_xn
    mov     rdi, [opt_model]
    call    xn_io_puts
    lea     rax, [the_model]
    mov     eax, [rax + model.err_line]
    test    eax, eax
    jz      .no_line
    mov     edi, ':'
    call    xn_io_putc
    lea     rax, [the_model]
    mov     edi, [rax + model.err_line]
    call    xn_io_dec
.no_line:
    PUTS    e_colon
    mov     edi, ebx
    call    xn_strerror
    mov     rdi, rax
    call    xn_io_puts
    PUTS    s_nl
    mov     eax, 3
    jmp     .ret

.loaded:
    lea     r15, [the_model]
    xor     ebp, ebp
.override:
    cmp     ebp, [ov_count]
    jae     .overridden
    lea     rax, [ov_key]
    mov     rbx, [rax + rbp * 8]
    mov     rdi, rbx
    call    xn_io_strlen
    mov     r12, rax                ; key length
    lea     rax, [ov_val]
    mov     r13, [rax + rbp * 8]
    mov     rdi, r13
    call    xn_io_strlen
    mov     r8, rax
    mov     rcx, r13
    mov     rdx, r12
    mov     rsi, rbx
    mov     rdi, r15
    call    xn_option_set
    test    eax, eax
    jz      .ov_ok
    mov     r12d, eax
    PUTS    e_xn
    mov     edi, '-'
    call    xn_io_putc
    mov     edi, '-'
    call    xn_io_putc
    mov     rdi, rbx
    call    xn_io_puts
    PUTS    e_colon
    mov     edi, r12d
    call    xn_strerror
    mov     rdi, rax
    call    xn_io_puts
    PUTS    s_nl
    mov     eax, 2
    jmp     .ret
.ov_ok:
    inc     ebp
    jmp     .override

.overridden:
    test    dword [opt_flags], F_STATS
    jz      .no_model_stats
    mov     rdi, r15
    call    stats
    xor     eax, eax
    jmp     .ret
.no_model_stats:
    test    dword [opt_flags], F_HELP
    jz      .no_help
    PUTS    help_text
    PUTS    this_model
    call    summary
    xor     eax, eax
    jmp     .ret

.no_help:
    ; r14d = 1 when the cortex must learn before it can speak
    xor     r14d, r14d
    test    dword [r15 + model.flags], XN_F_WEIGHTS
    setz    r14b
    test    dword [opt_flags], F_TRAIN
    jnz     .must_learn
    cmp     qword [opt_agents], 0   ; a swarm always learns from scratch
    je      .decided
.must_learn:
    mov     r14d, 1
.decided:
    test    dword [opt_flags], F_EMIT
    jz      .no_emit
    call    .maybe_train
    mov     rdi, r15
    mov     esi, 1
    call    xn_emit
    xor     eax, eax
    jmp     .ret

.no_emit:
    mov     rbx, [opt_infer]
    test    rbx, rbx
    jz      .report
    mov     rdi, rbx
    call    xn_io_strlen
    mov     rsi, rax
    mov     rdi, rbx
    mov     edx, [r15 + model.inputs]
    call    parse_bits
    test    edx, edx
    jnz     .bad_bits
    mov     r12, rax                ; row
    call    .maybe_train
    mov     rdi, r15
    mov     rsi, r12
    call    xn_forward
    mov     r13d, eax
    test    dword [opt_flags], F_QUIET
    jnz     .infer_quiet
    PUTS    i_head
    mov     rdi, rbx
    call    xn_io_puts
    PUTS    m_arrow
.infer_quiet:
    lea     edi, [r13d + '0']
    call    xn_io_putc
    PUTS    s_nl
    xor     eax, eax
    jmp     .ret
.bad_bits:
    PUTS    e_xn
    PUTS    e_bits1
    mov     edi, [r15 + model.inputs]
    call    xn_io_dec
    PUTS    e_bits2
    mov     eax, 2
    jmp     .ret

; local: train iff r14d says so
.maybe_train:
    test    r14d, r14d
    jz      .mt_done
    sub     rsp, 8
    mov     rax, [opt_agents]
    test    rax, rax
    jnz     .mt_swarm
    mov     rdi, r15
    call    xn_train
    add     rsp, 8
.mt_done:
    ret
.mt_swarm:                          ; axn: spawn the swarm, keep the champion
    lea     rsi, [the_swarm]
    mov     [rsi + swarm.agents], rax
    mov     eax, [opt_threads]
    mov     [rsi + swarm.threads], eax
    mov     rdi, r15
    call    xn_swarm_run
    add     rsp, 8
    ret

; ── the full report ──
.report:
    test    dword [opt_flags], F_QUIET
    jnz     .quiet_report
    PUTS    banner
    call    xn_version
    lea     rdi, [rax + 3]          ; "0.1.0 …"
    mov     rsi, 5
    call    xn_io_putn
    PUTS    banner2
    call    summary
    test    r14d, r14d
    jz      .skip_train
    cmp     qword [opt_agents], 0
    je      .solo
    call    .maybe_train
    call    swarm_report
    jmp     .anatomy
.solo:
    PUTS    r_train
    call    .maybe_train
    PUTS    r_restart
    mov     edi, [r15 + model.used]
    call    xn_io_dec
    PUTS    r_steps
    mov     rdi, [r15 + model.steps]
    call    xn_io_dec
    PUTS    s_nl
    jmp     .anatomy
.skip_train:
    PUTS    r_skip
.anatomy:
    cmp     dword [r15 + model.verbose], 0
    je      .table
    call    anatomy
.table:
    call    truth_table
    jmp     .verdict

.quiet_report:
    call    .maybe_train

.verdict:
    mov     rdi, r15
    call    xn_truth
    mov     rbx, rax                ; cortex truth
    mov     edi, [r15 + model.inputs]
    call    xn_rowmask
    mov     rcx, rbx
    xor     rcx, [r15 + model.truth]
    and     rcx, rax
    popcnt  r12, rcx                ; loss
    PUTS    r_loss
    mov     rdi, r12
    call    xn_io_dec
    PUTS    r_slash
    mov     ecx, [r15 + model.inputs]
    mov     edi, 1
    shl     edi, cl
    mov     r13d, edi               ; rows
    call    xn_io_dec
    test    r12, r12
    jnz     .noise
    PUTS    r_signal
    xor     ebp, ebp                ; reply: row 0 first
.reply:
    cmp     ebp, r13d
    jae     .replied
    bt      rbx, rbp
    setc    dil
    add     dil, '0'
    call    xn_io_putc
    inc     ebp
    jmp     .reply
.replied:
    PUTS    s_nl
    jmp     .console
.noise:
    PUTS    r_noise

.console:
    test    dword [opt_flags], F_PROMPT
    jz      .done
    call    console
.done:
    xor     eax, eax
    test    r12, r12
    setnz   al

.ret:
    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ── summary: name, shape, options ───────────────────────────────────────────
summary:
    push    rbx
    push    rbp
    push    r12
    lea     rbx, [the_model]
    PUTS    m_model
    lea     rdi, [rbx + model.name]
    call    xn_io_puts
    PUTS    m_inputs
    mov     edi, [rbx + model.inputs]
    call    xn_io_dec
    PUTS    m_rows
    mov     ecx, [rbx + model.inputs]
    mov     edi, 1
    shl     edi, cl
    call    xn_io_dec
    PUTS    m_target
    lea     rdi, [target_names]
    mov     esi, [rbx + model.target]
    call    nth_str
    mov     rdi, rax
    call    xn_io_puts
    PUTS    m_open
    mov     rdi, [rbx + model.truth]
    call    xn_io_hex
    PUTS    m_close
    PUTS    m_cortex
    mov     edi, [rbx + model.inputs]
    call    xn_io_dec
    lea     rbp, [rbx + model.layers]
    xor     r12d, r12d
.shape:
    cmp     r12d, [rbx + model.nlayers]
    jae     .shaped
    PUTS    m_arrow
    mov     edi, [rbp + layer.neurons]
    call    xn_io_dec
    add     rbp, XN_LAYER_SIZE
    inc     r12d
    jmp     .shape
.shaped:
    PUTS    m_options
    mov     rdi, [rbx + model.seed]
    call    xn_io_hex
    PUTS    m_epochs
    mov     rdi, [rbx + model.epochs]
    call    xn_io_dec
    PUTS    m_restarts
    mov     edi, [rbx + model.restarts]
    call    xn_io_dec
    PUTS    m_verbose
    mov     edi, [rbx + model.verbose]
    call    xn_io_dec
    test    dword [rbx + model.flags], XN_F_WEIGHTS
    jz      .learn
    PUTS    m_wfile
    jmp     .out
.learn:
    PUTS    m_wlearn
.out:
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ── anatomy: every cell, its weight glyphs and threshold ────────────────────
anatomy:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    lea     r14, [the_model]
    lea     rbx, [r14 + model.layers]
    xor     ebp, ebp
.layer:
    cmp     ebp, [r14 + model.nlayers]
    jae     .done
    PUTS    r_layer
    mov     edi, ebp
    call    xn_io_dec
    PUTS    r_fanin
    mov     edi, [rbx + layer.fan_in]
    call    xn_io_dec
    PUTS    r_cells
    mov     edi, [rbx + layer.neurons]
    call    xn_io_dec
    PUTS    s_nl
    xor     r12d, r12d
.cell:
    cmp     r12d, [rbx + layer.neurons]
    jae     .next
    PUTS    r_n
    mov     edi, r12d
    call    xn_io_dec
    mov     edi, 3
    cmp     r12d, 10
    adc     edi, 0                  ; pad n0..n9 one wider
    sub     edi, 1
    call    xn_io_spaces
    PUTS    r_w
    mov     rdi, [rbx + layer.w + r12 * 8]
    mov     esi, [rbx + layer.fan_in]
    call    xn_io_glyphs
    PUTS    r_t
    movzx   edi, byte [rbx + layer.t + r12]
    call    xn_io_dec
    PUTS    s_nl
    inc     r12d
    jmp     .cell
.next:
    add     rbx, XN_LAYER_SIZE
    inc     ebp
    jmp     .layer
.done:
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ── truth table: every row, the wish and the answer ─────────────────────────
truth_table:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    lea     rbx, [the_model]
    mov     rdi, rbx
    call    xn_truth
    mov     r13, rax                ; cortex
    mov     r14d, [rbx + model.inputs]
    PUTS    r_table
    lea     edi, [r14d + 1]         ; align "target" over the first value column
    cmp     r14d, 3
    jae     .head
    mov     edi, 3
.head:
    call    xn_io_spaces
    PUTS    r_tcols
    mov     ecx, r14d
    mov     r12d, 1
    shl     r12d, cl                ; rows
    xor     ebp, ebp
.row:
    cmp     ebp, r12d
    jae     .done
    PUTS    s_sp2
    mov     edi, ebp
    mov     esi, r14d
    call    xn_io_bits
    mov     edi, 4
    cmp     r14d, 3
    jae     .wide
    mov     edi, 5
    sub     edi, r14d
    add     edi, 1
.wide:
    call    xn_io_spaces
    mov     rax, [rbx + model.truth]
    bt      rax, rbp
    setc    dil
    add     dil, '0'
    call    xn_io_putc
    mov     edi, 7
    call    xn_io_spaces
    bt      r13, rbp
    setc    dil
    add     dil, '0'
    call    xn_io_putc
    mov     edi, 7
    call    xn_io_spaces
    mov     rax, [rbx + model.truth]
    xor     rax, r13
    bt      rax, rbp
    jc      .miss
    PUTS    r_ok
    jmp     .nextrow
.miss:
    PUTS    r_bad
.nextrow:
    inc     ebp
    jmp     .row
.done:
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ── console: a line-at-a-time conversation over stdin ──────────────────────
console:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    lea     rbx, [the_model]
    PUTS    c_open1
    mov     edi, [rbx + model.inputs]
    call    xn_io_dec
    PUTS    c_open2
    mov     edi, 1
    mov     esi, [rbx + model.inputs]
    call    xn_io_bits
    PUTS    c_open3
.prompt:
    PUTS    c_prompt
    call    xn_io_flush
    xor     r12d, r12d              ; length
    xor     r13d, r13d              ; eof seen
.read:
    sub     rsp, 8
    mov     rsi, rsp
    xor     edi, edi
    mov     edx, 1
    mov     eax, SYS_read
    syscall
    movzx   ecx, byte [rsp]
    add     rsp, 8
    test    rax, rax
    jle     .eof
    cmp     cl, 10
    je      .line
    cmp     r12d, LINE_MAX
    jae     .read
    lea     rdx, [linebuf]
    mov     [rdx + r12], cl
    inc     r12d
    jmp     .read
.eof:
    mov     r13d, 1
    test    r12d, r12d
    jnz     .line
    PUTS    s_nl
    jmp     .close
.line:
    lea     rbp, [linebuf]
.ltrim:                             ; trim spaces, tabs and CR on both ends
    test    r12d, r12d
    jz      .trimmed
    mov     al, [rbp]
    cmp     al, ' '
    je      .lcut
    cmp     al, 9
    je      .lcut
    cmp     al, 13
    jne     .rtrim
.lcut:
    inc     rbp
    dec     r12d
    jmp     .ltrim
.rtrim:
    mov     al, [rbp + r12 - 1]
    cmp     al, ' '
    je      .rcut
    cmp     al, 9
    je      .rcut
    cmp     al, 13
    jne     .trimmed
.rcut:
    dec     r12d
    jnz     .rtrim
.trimmed:
    mov     byte [rbp + r12], 0
    test    r12d, r12d
    jz      .again
    mov     rdi, rbp
    lea     rsi, [w_q]
    call    cstreq
    test    eax, eax
    jnz     .close
    mov     rdi, rbp
    lea     rsi, [w_quit]
    call    cstreq
    test    eax, eax
    jnz     .close
    mov     rdi, rbp
    lea     rsi, [w_exit]
    call    cstreq
    test    eax, eax
    jnz     .close
    mov     rdi, rbp
    mov     esi, r12d
    mov     edx, [rbx + model.inputs]
    call    parse_bits
    test    edx, edx
    jnz     .huh
    mov     r14, rax                ; row
    mov     rdi, rbx
    mov     rsi, r14
    call    xn_forward
    mov     r12d, eax
    PUTS    c_reply
    lea     edi, [r12d + '0']
    call    xn_io_putc
    PUTS    c_target
    mov     rax, [rbx + model.truth]
    bt      rax, r14
    setc    dil
    add     dil, '0'
    call    xn_io_putc
    PUTS    s_sp2
    mov     rax, [rbx + model.truth]
    bt      rax, r14
    setc    al
    cmp     al, r12b
    jne     .wrong
    PUTS    r_ok
    jmp     .again
.wrong:
    PUTS    r_bad
    jmp     .again
.huh:
    PUTS    c_huh1
    mov     edi, [rbx + model.inputs]
    call    xn_io_dec
    PUTS    c_huh2
.again:
    test    r13d, r13d
    jz      .prompt
.close:
    PUTS    c_bye
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ═══════════════════════════════════════════════════════════════════════════
;  axn report and --stats
; ═══════════════════════════════════════════════════════════════════════════
section .rodata
w_head:     db 10, "  ◇ swarm · ", 0
w_agents:   db " agents · ", 0
w_threads:  db " threads · one life of ", 0
w_life:     db " mutations each", 10, 0
w_contact:  db "    contact   ", 0
w_open:     db " agents (", 0
w_pct:      db "%)", 10, 0
w_champ:    db "    champion  agent #", 0
w_dot:      db " · ", 0
w_muts:     db " mutations · loss ", 0
w_seed:     db " · seed ", 0
w_pace:     db "    pace      ", 0
w_ms:       db " ms · ", 0
w_aps:      db " agents/s · ", 0
w_total:    db " mutations", 10, 0

proc_exe:   db "/proc/self/exe", 0
st_head:    db 10, "  ⊕ xn stats · the size of everything", 10, 10
            db "    object             bytes", 10, 0
st_cell:    db "    cell          ", 0
st_layer:   db "    layer         ", 0
st_model:   db "    model         ", 0
st_data:    db "    dataset       ", 0
st_agent:   db "    agent         ", 0
st_live:    db "    live agent    ", 0
st_swarm:   db "    swarm         ", 0
st_stack:   db "    thread stack  ", 0
st_io:      db "    io buffer     ", 0
st_exe:     db "    xn binary     ", 0
st_cpus:    db "    cpus          ", 0
st_billion: db "    1e9 agents    ", 0
n_cell:     db "    64-bit weight + 8-bit threshold", 10, 0
n_layer:    db "    64 cells + a 16-byte header", 10, 0
n_model:    db "    8 layers + a 112-byte header · fixed, never allocated", 10, 0
n_data:     db "    every truth table (<= 64 rows) is one register", 10, 0
n_agent:    db "    an agent is its seed; nothing else is kept", 10, 0
n_live:     db "    peak stack while an agent lives, per thread", 10, 0
n_swarm:    db "    the shared tournament ledger", 10, 0
n_stack:    db "    per spawned worker, mmap'd, freed on join", 10, 0
n_io:       db "    the only process-global state", 10, 0
n_exe:      db "    this executable, static, no libc", 10, 0
n_cpus:     db "    threads the swarm may use", 10, 0
n_billion:  db "    stored · streamed, judged, forgotten", 10, 0
st_this:    db 10, "    ── this model · ", 0
st_this2:   db " ──", 10, 0
st_cells:   db "    cells         ", 0
st_wbits:   db "    weight bits   ", 0
st_tbits:   db "    thresh bits   ", 0
st_genome:  db "    genome bits   ", 0
st_wired:   db "    wired bytes   ", 0
st_rows:    db "    rows          ", 0
n_cells:    db "    XOR cells across every layer", 10, 0
n_wbits:    db "    one per cell input", 10, 0
n_tbits:    db "    enough to count 0..fan-in", 10, 0
n_genome:   db "    search space 2^", 0
n_wired:    db "    of the model struct in use", 10, 0
n_rows:     db "    truth word ", 0

%macro STAT 3                       ; label, value already in rdi, note
    push    rdi
    lea     rdi, [%1]
    call    xn_io_puts
    pop     rdi
    mov     esi, 10
    call    put_dec_w
    lea     rdi, [%3]
    call    xn_io_puts
%endmacro

section .text

; put_dec_w(rdi value, esi width)   right-aligned decimal
put_dec_w:
    push    rdi
    mov     rax, rdi
    xor     ecx, ecx
    mov     r8d, 10
.count:
    xor     edx, edx
    div     r8
    inc     ecx
    test    rax, rax
    jnz     .count
    mov     edi, esi
    sub     edi, ecx
    call    xn_io_spaces
    pop     rdi
    jmp     xn_io_dec

; put_hundredths(rdi value·100)   "12.05"
put_hundredths:
    push    rbx
    mov     rax, rdi
    xor     edx, edx
    mov     ecx, 100
    div     rcx
    mov     rbx, rdx
    mov     rdi, rax
    call    xn_io_dec
    mov     edi, '.'
    call    xn_io_putc
    cmp     rbx, 10
    jae     .two
    mov     edi, '0'
    call    xn_io_putc
.two:
    mov     rdi, rbx
    call    xn_io_dec
    pop     rbx
    ret

; exe_size() -> rax   bytes of /proc/self/exe (0 if unknown)
exe_size:
    push    rbx
    sub     rsp, STAT_SIZE
    lea     rdi, [proc_exe]
    mov     esi, O_RDONLY
    xor     edx, edx
    mov     eax, SYS_open
    syscall
    test    rax, rax
    js      .none
    mov     rbx, rax
    mov     edi, ebx
    mov     rsi, rsp
    mov     eax, SYS_fstat
    syscall
    push    rax
    mov     edi, ebx
    mov     eax, SYS_close
    syscall
    pop     rax
    test    rax, rax
    js      .none
    mov     rax, [rsp + STAT_ST_SIZE]
    jmp     .out
.none:
    xor     eax, eax
.out:
    add     rsp, STAT_SIZE
    pop     rbx
    ret

; swarm_report()   what the swarm did, after xn_swarm_run
swarm_report:
    push    rbx
    push    r12
    push    r13
    lea     rbx, [the_swarm]
    lea     r12, [the_model]
    PUTS    w_head
    mov     rdi, [rbx + swarm.agents]
    call    xn_io_dec
    PUTS    w_agents
    mov     edi, [rbx + swarm.threads]
    call    xn_io_dec
    PUTS    w_threads
    mov     rdi, [r12 + model.epochs]
    call    xn_io_dec
    PUTS    w_life

    PUTS    w_contact
    mov     rdi, [rbx + swarm.contact]
    call    xn_io_dec
    PUTS    w_open
    mov     rax, [rbx + swarm.contact]
    mov     ecx, 10000
    mul     rcx
    div     qword [rbx + swarm.agents]
    mov     rdi, rax
    call    put_hundredths
    PUTS    w_pct

    PUTS    w_champ
    mov     rdi, [rbx + swarm.best_id]
    call    xn_io_dec
    PUTS    w_dot
    mov     rdi, [rbx + swarm.best_steps]
    call    xn_io_dec
    PUTS    w_muts
    mov     rdi, [rbx + swarm.best_loss]
    call    xn_io_dec
    PUTS    w_seed
    mov     rdi, [r12 + model.seed]
    mov     rsi, [rbx + swarm.best_id]
    call    xn_agent_seed
    mov     rdi, rax
    call    xn_io_hex
    PUTS    s_nl

    PUTS    w_pace
    mov     rax, [rbx + swarm.ns]
    xor     edx, edx
    mov     ecx, 10000
    div     rcx                     ; hundredths of a millisecond
    mov     rdi, rax
    call    put_hundredths
    PUTS    w_ms
    mov     r13, [rbx + swarm.ns]
    test    r13, r13
    jnz     .timed
    inc     r13
.timed:
    mov     rax, [rbx + swarm.agents]
    mov     ecx, 1000000000
    mul     rcx
    div     r13
    mov     rdi, rax
    call    xn_io_dec
    PUTS    w_aps
    mov     rdi, [rbx + swarm.steps]
    call    xn_io_dec
    PUTS    w_total
    pop     r13
    pop     r12
    pop     rbx
    ret

; stats(rdi *model | 0)   the size of every object in the system
stats:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    mov     rbx, rdi
    PUTS    st_head
    mov     edi, 9
    STAT    st_cell, 0, n_cell
    mov     edi, XN_LAYER_SIZE
    STAT    st_layer, 0, n_layer
    call    xn_model_size
    mov     edi, eax
    STAT    st_model, 0, n_model
    mov     edi, 8
    STAT    st_data, 0, n_data
    mov     edi, 8
    STAT    st_agent, 0, n_agent
    call    xn_agent_footprint
    mov     edi, eax
    STAT    st_live, 0, n_live
    call    xn_swarm_size
    mov     edi, eax
    STAT    st_swarm, 0, n_swarm
    call    xn_swarm_stack
    mov     edi, eax
    STAT    st_stack, 0, n_stack
    call    xn_io_capacity
    mov     edi, eax
    STAT    st_io, 0, n_io
    call    exe_size
    mov     rdi, rax
    STAT    st_exe, 0, n_exe
    call    xn_cpus
    mov     edi, eax
    STAT    st_cpus, 0, n_cpus
    xor     edi, edi
    STAT    st_billion, 0, n_billion

    test    rbx, rbx
    jz      .done
    PUTS    st_this
    lea     rdi, [rbx + model.name]
    call    xn_io_puts
    PUTS    st_this2
    xor     r12d, r12d              ; cells
    xor     r13d, r13d              ; weight bits
    xor     r14d, r14d              ; threshold bits
    lea     r15, [rbx + model.layers]
    mov     ecx, [rbx + model.nlayers]
.layer:
    test    ecx, ecx
    jz      .summed
    mov     eax, [r15 + layer.neurons]
    add     r12, rax
    mov     edx, [r15 + layer.fan_in]
    imul    edx, eax
    add     r13, rdx
    bsr     edx, [r15 + layer.fan_in]
    inc     edx                     ; bits to hold 0..fan_in
    imul    edx, eax
    add     r14, rdx
    add     r15, XN_LAYER_SIZE
    dec     ecx
    jmp     .layer
.summed:
    mov     rdi, r12
    STAT    st_cells, 0, n_cells
    mov     rdi, r13
    STAT    st_wbits, 0, n_wbits
    mov     rdi, r14
    STAT    st_tbits, 0, n_tbits
    lea     rdi, [r13 + r14]
    STAT    st_genome, 0, n_genome
    lea     rdi, [r13 + r14]
    call    xn_io_dec
    PUTS    s_nl
    imul    edi, [rbx + model.nlayers], XN_LAYER_SIZE
    add     edi, model.layers
    STAT    st_wired, 0, n_wired
    mov     ecx, [rbx + model.inputs]
    mov     edi, 1
    shl     edi, cl
    STAT    st_rows, 0, n_rows
    mov     rdi, [rbx + model.truth]
    call    xn_io_hex
    PUTS    s_nl
.done:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

NOEXECSTACK
