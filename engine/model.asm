; ═══════════════════════════════════════════════════════════════════════════
;  model.asm ─ the organism: birth, the .xn reader, validation, transmission.
;
;  .xn is a sectioned key = value text:
;
;      [model]    name, inputs (1..6), target (xor|xnor|and|or|nand|nor)
;                 or truth (an explicit truth word, bit x = f(x))
;      [layer]    neurons (1..64), optional weights + thresholds lists
;      [options]  seed, epochs, restarts, verbose
;
;  '#' and ';' open comments.  Numbers are decimal, 0x hex or 0b binary.
;  Layers stack in file order; the last must hold exactly one neuron.
;  If every layer carries weights + thresholds, the model is born trained.
; ═══════════════════════════════════════════════════════════════════════════
%include "xn.inc"
default rel

extern xn_option_set, xn_text_eq, xn_text_u64
extern xn_target_truth, xn_rowmask, xn_fanmask, xn_loss
extern xn_io_fd, xn_io_flush, xn_io_puts, xn_io_dec, xn_io_hex

global xn_model_init, xn_parse, xn_load, xn_emit
global xn_strerror, xn_version, xn_model_size, xn_abi_version

section .rodata
s_model:      db "model", 0
s_layer:      db "layer", 0
s_options:    db "options", 0
s_name:       db "name", 0
s_inputs:     db "inputs", 0
s_target:     db "target", 0
s_truth:      db "truth", 0
s_neurons:    db "neurons", 0
s_weights:    db "weights", 0
s_thresholds: db "thresholds", 0
t_xor:        db "xor", 0
t_xnor:       db "xnor", 0
t_and:        db "and", 0
t_or:         db "or", 0
t_nand:       db "nand", 0
t_nor:        db "nor", 0

version_str:  db "xn 0.1.0 ", 0xE2, 0x8A, 0x95, " first-contact", 0

errstrs:
    db "ok", 0
    db "cannot read the .xn file", 0
    db "syntax error", 0
    db "unknown section, or a key outside any section", 0
    db "unknown key", 0
    db "malformed value", 0
    db "value outside the cortex limits", 0
    db "model has no [layer]", 0
    db "the last layer must hold exactly 1 neuron", 0
    db "unknown target (xor xnor and or nand nor)", 0
    db "[model] inputs must be 1..6", 0
    db "weights and thresholds must list one value per neuron", 0
    db "bad command line", 0
    db "unknown error", 0

e_head:     db "# xn ", 0xE2, 0x8A, 0x95, " transmission ", 0xC2, 0xB7, " trained cortex", 10, "# loss ", 0
e_model:    db 10, 10, "[model]", 10, "name     = ", 0
e_inputs:   db 10, "inputs   = ", 0
e_truth:    db 10, "truth    = ", 0
e_layer:    db 10, 10, "[layer]", 10, "neurons    = ", 0
e_weights:  db 10, "weights    = ", 0
e_thresh:   db 10, "thresholds = ", 0
e_options:  db 10, 10, "[options]", 10, "seed     = ", 0
e_epochs:   db 10, "epochs   = ", 0
e_restarts: db 10, "restarts = ", 0
e_verbose:  db 10, "verbose  = ", 0
e_sep:      db ", ", 0
e_nl:       db 10, 0

; parse frame
P_KEY       equ 0
P_KEYLEN    equ 8
P_VAL       equ 16
P_VALLEN    equ 24
P_LIST      equ 40
P_FRAME     equ 40 + 8 * XN_MAX_NEURONS     ; 552: rsp stays 16-aligned

%macro KEYIS 2
    mov     rdi, [rsp + P_KEY]
    mov     rsi, [rsp + P_KEYLEN]
    lea     rdx, [%1]
    call    xn_text_eq
    test    eax, eax
    jnz     %2
%endmacro

%macro VALIS 2
    mov     rdi, [rsp + P_VAL]
    mov     rsi, [rsp + P_VALLEN]
    lea     rdx, [%1]
    call    xn_text_eq
    test    eax, eax
    jnz     %2
%endmacro

%macro VALU64 0                     ; rax = value, or bail out with XN_EVALUE
    mov     rdi, [rsp + P_VAL]
    mov     rsi, [rsp + P_VALLEN]
    call    xn_text_u64
    test    edx, edx
    jnz     .value
%endmacro

%macro PUTS 1
    lea     rdi, [%1]
    call    xn_io_puts
%endmacro

section .text

xn_model_size:
    mov     eax, XN_MODEL_SIZE
    ret

xn_abi_version:
    mov     eax, XN_ABI_VERSION
    ret

xn_version:
    lea     rax, [version_str]
    ret

; xn_strerror(edi code) -> rax cstr
xn_strerror:
    cmp     edi, XN_ECOUNT
    jbe     .known
    mov     edi, XN_ECOUNT
.known:
    lea     rax, [errstrs]
.skip:
    test    edi, edi
    jz      .done
.chr:
    inc     rax
    cmp     byte [rax - 1], 0
    jne     .chr
    dec     edi
    jmp     .skip
.done:
    ret

; xn_model_init(rdi *model)   a blank organism with default instincts
xn_model_init:
    push    rdi
    mov     ecx, XN_MODEL_SIZE
    xor     eax, eax
    rep     stosb
    pop     rdi
    mov     dword [rdi + model.magic], XN_MAGIC
    mov     dword [rdi + model.version], XN_ABI_VERSION
    mov     qword [rdi + model.seed], 0x5EED
    mov     qword [rdi + model.epochs], 20000
    mov     dword [rdi + model.restarts], 32
    mov     dword [rdi + model.verbose], 1
    mov     rax, 'unnamed'
    mov     [rdi + model.name], rax
    ret

; ── little scanners (leaf, rdi = p, rsi = end, rax = new p) ─────────────────
skip_ws:
    cmp     rdi, rsi
    jae     .e
    movzx   eax, byte [rdi]
    cmp     al, ' '
    je      .s
    cmp     al, 9
    je      .s
    cmp     al, 13
    jne     .e
.s:
    inc     rdi
    jmp     skip_ws
.e:
    mov     rax, rdi
    ret

to_eol:
    cmp     rdi, rsi
    jae     .e
    cmp     byte [rdi], 10
    je      .e
    inc     rdi
    jmp     to_eol
.e:
    mov     rax, rdi
    ret

ident_end:
    cmp     rdi, rsi
    jae     .e
    movzx   eax, byte [rdi]
    cmp     al, '_'
    je      .ok
    cmp     al, '-'
    je      .ok
    lea     ecx, [rax - '0']
    cmp     ecx, 9
    jbe     .ok
    or      eax, 0x20
    sub     eax, 'a'
    cmp     eax, 25
    ja      .e
.ok:
    inc     rdi
    jmp     ident_end
.e:
    mov     rax, rdi
    ret

; parse_list(rdi ptr, rsi len, rdx *out, ecx max) -> rax count | -1 bad | -2 too many
parse_list:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    mov     rbx, rdi
    lea     r12, [rdi + rsi]
    mov     r13, rdx
    mov     r14d, ecx
    xor     r15d, r15d
.sep:
    cmp     rbx, r12
    jae     .done
    mov     al, [rbx]
    cmp     al, ' '
    je      .skip
    cmp     al, 9
    je      .skip
    cmp     al, ','
    jne     .tok
.skip:
    inc     rbx
    jmp     .sep
.tok:
    mov     rdi, rbx
.tl:
    cmp     rbx, r12
    jae     .te
    mov     al, [rbx]
    cmp     al, ' '
    je      .te
    cmp     al, 9
    je      .te
    cmp     al, ','
    je      .te
    inc     rbx
    jmp     .tl
.te:
    cmp     r15d, r14d
    jae     .many
    mov     rsi, rbx
    sub     rsi, rdi
    call    xn_text_u64
    test    edx, edx
    jnz     .bad
    mov     [r13 + r15 * 8], rax
    inc     r15d
    jmp     .sep
.done:
    mov     eax, r15d
    jmp     .out
.bad:
    mov     rax, -1
    jmp     .out
.many:
    mov     rax, -2
.out:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; xn_parse(rdi *model, rsi text, rdx len) -> eax XN_*
;   on failure model.err_line holds the 1-based line (0 = whole-model problem)
xn_parse:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, P_FRAME
    mov     rbx, rdi
    mov     r12, rsi                ; p
    lea     r13, [rsi + rdx]        ; end
    call    xn_model_init
    xor     r14d, r14d              ; line
    xor     r15d, r15d              ; section: 0 none, 1 model, 2 layer, 3 options
    xor     ebp, ebp                ; current layer

.line:
    inc     r14d
.ws:
    mov     rdi, r12
    mov     rsi, r13
    call    skip_ws
    mov     r12, rax
    cmp     r12, r13
    jae     .eof
    movzx   eax, byte [r12]
    cmp     al, 10
    je      .newline
    cmp     al, '#'
    je      .comment
    cmp     al, ';'
    je      .comment
    cmp     al, '['
    je      .section
    jmp     .keyval
.newline:
    inc     r12
    jmp     .line
.comment:
    mov     rdi, r12
    mov     rsi, r13
    call    to_eol
    mov     r12, rax
    jmp     .ws

; ── [section] ──
.section:
    inc     r12
    mov     rdi, r12
    mov     rsi, r13
    call    ident_end
    cmp     rax, r13
    jae     .syntax
    cmp     byte [rax], ']'
    jne     .syntax
    mov     [rsp + P_KEY], r12
    lea     r12, [rax + 1]
    sub     rax, [rsp + P_KEY]
    mov     [rsp + P_KEYLEN], rax
    KEYIS   s_model, .sec_model
    KEYIS   s_layer, .sec_layer
    KEYIS   s_options, .sec_options
    mov     eax, XN_ESECTION
    jmp     .fail
.sec_model:
    mov     r15d, 1
    jmp     .tail
.sec_layer:
    mov     eax, [rbx + model.nlayers]
    cmp     eax, XN_MAX_LAYERS
    jae     .limit
    imul    ecx, eax, XN_LAYER_SIZE
    lea     rbp, [rbx + model.layers + rcx]
    inc     eax
    mov     [rbx + model.nlayers], eax
    mov     r15d, 2
    jmp     .tail
.sec_options:
    mov     r15d, 3
.tail:                              ; nothing but a comment may follow
    mov     rdi, r12
    mov     rsi, r13
    call    skip_ws
    mov     r12, rax
    cmp     r12, r13
    jae     .eof
    mov     al, [r12]
    cmp     al, 10
    je      .newline
    cmp     al, '#'
    je      .comment
    cmp     al, ';'
    je      .comment
    jmp     .syntax

; ── key = value ──
.keyval:
    mov     rdi, r12
    mov     rsi, r13
    call    ident_end
    cmp     rax, r12
    je      .syntax
    mov     [rsp + P_KEY], r12
    mov     rcx, rax
    sub     rcx, r12
    mov     [rsp + P_KEYLEN], rcx
    mov     rdi, rax
    mov     rsi, r13
    call    skip_ws
    mov     r12, rax
    cmp     r12, r13
    jae     .syntax
    cmp     byte [r12], '='
    jne     .syntax
    inc     r12
    mov     rdi, r12
    mov     rsi, r13
    call    skip_ws
    mov     r12, rax
    mov     [rsp + P_VAL], rax
.vscan:
    cmp     rax, r13
    jae     .vend
    mov     cl, [rax]
    cmp     cl, 10
    je      .vend
    cmp     cl, '#'
    je      .vend
    inc     rax
    jmp     .vscan
.vend:
    mov     r12, rax
.trim:
    cmp     rax, [rsp + P_VAL]
    jbe     .trimmed
    mov     cl, [rax - 1]
    cmp     cl, ' '
    je      .trim1
    cmp     cl, 9
    je      .trim1
    cmp     cl, 13
    jne     .trimmed
.trim1:
    dec     rax
    jmp     .trim
.trimmed:
    sub     rax, [rsp + P_VAL]
    mov     [rsp + P_VALLEN], rax
    jz      .value
    cmp     r15d, 1
    je      .mkey
    cmp     r15d, 2
    je      .lkey
    cmp     r15d, 3
    je      .okey
    mov     eax, XN_ESECTION
    jmp     .fail

; ── [options] ──
.okey:
    mov     rdi, rbx
    mov     rsi, [rsp + P_KEY]
    mov     rdx, [rsp + P_KEYLEN]
    mov     rcx, [rsp + P_VAL]
    mov     r8, [rsp + P_VALLEN]
    call    xn_option_set
    test    eax, eax
    jnz     .fail
    jmp     .ws

; ── [model] ──
.mkey:
    KEYIS   s_name, .m_name
    KEYIS   s_inputs, .m_inputs
    KEYIS   s_target, .m_target
    KEYIS   s_truth, .m_truth
    jmp     .key
.m_name:
    lea     rdi, [rbx + model.name]
    xor     eax, eax
    mov     ecx, 32
    rep     stosb
    mov     rcx, [rsp + P_VALLEN]
    cmp     rcx, 31
    jbe     .name_fits
    mov     ecx, 31
.name_fits:
    mov     rsi, [rsp + P_VAL]
    lea     rdi, [rbx + model.name]
    rep     movsb
    jmp     .ws
.m_inputs:
    VALU64
    lea     rcx, [rax - 1]
    cmp     rcx, XN_MAX_INPUTS - 1
    ja      .inputs
    mov     [rbx + model.inputs], eax
    jmp     .ws
.m_target:
    mov     r8d, XN_T_XOR
    VALIS   t_xor, .m_target_set
    mov     r8d, XN_T_XNOR
    VALIS   t_xnor, .m_target_set
    mov     r8d, XN_T_AND
    VALIS   t_and, .m_target_set
    mov     r8d, XN_T_OR
    VALIS   t_or, .m_target_set
    mov     r8d, XN_T_NAND
    VALIS   t_nand, .m_target_set
    mov     r8d, XN_T_NOR
    VALIS   t_nor, .m_target_set
    mov     eax, XN_ETARGET
    jmp     .fail
.m_target_set:
    mov     [rbx + model.target], r8d
    jmp     .ws
.m_truth:
    VALU64
    mov     [rbx + model.truth], rax
    mov     dword [rbx + model.target], XN_T_TRUTH
    jmp     .ws

; ── [layer] ──
.lkey:
    KEYIS   s_neurons, .l_neurons
    KEYIS   s_weights, .l_weights
    KEYIS   s_thresholds, .l_thresholds
    jmp     .key
.l_neurons:
    VALU64
    lea     rcx, [rax - 1]
    cmp     rcx, XN_MAX_NEURONS - 1
    ja      .limit
    mov     [rbp + layer.neurons], eax
    jmp     .ws
.l_weights:
    call    .list
    mov     [rbp + layer.nw], eax
    xor     ecx, ecx
.wcopy:
    cmp     ecx, eax
    jae     .ws
    mov     rdx, [rsp + P_LIST + rcx * 8]
    mov     [rbp + layer.w + rcx * 8], rdx
    inc     ecx
    jmp     .wcopy
.l_thresholds:
    call    .list
    mov     [rbp + layer.nt], eax
    xor     ecx, ecx
.tcopy:
    cmp     ecx, eax
    jae     .ws
    mov     rdx, [rsp + P_LIST + rcx * 8]
    cmp     rdx, 64
    ja      .value
    mov     [rbp + layer.t + rcx], dl
    inc     ecx
    jmp     .tcopy

; local subroutine: parse the value as a list into P_LIST, eax = count
; (called with one extra return address on the stack, hence the +8)
.list:
    mov     rdi, [rsp + 8 + P_VAL]
    mov     rsi, [rsp + 8 + P_VALLEN]
    lea     rdx, [rsp + 8 + P_LIST]
    mov     ecx, XN_MAX_NEURONS
    call    parse_list
    test    rax, rax
    js      .list_err
    ret
.list_err:
    add     rsp, 8                  ; abandon the subroutine frame
    cmp     rax, -2
    je      .limit
    jmp     .value

; ── endings ──
.eof:
    mov     rdi, rbx
    call    finalize
    jmp     .out
.syntax:
    mov     eax, XN_ESYNTAX
    jmp     .fail
.key:
    mov     eax, XN_EKEY
    jmp     .fail
.value:
    mov     eax, XN_EVALUE
    jmp     .fail
.limit:
    mov     eax, XN_ELIMIT
    jmp     .fail
.inputs:
    mov     eax, XN_EINPUTS
.fail:
    mov     [rbx + model.err_line], r14d
.out:
    add     rsp, P_FRAME
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; finalize(rdi *model) -> eax   wire the layers together and check the anatomy
finalize:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    mov     rbx, rdi
    mov     dword [rbx + model.err_line], 0
    mov     eax, [rbx + model.inputs]
    dec     eax
    cmp     eax, XN_MAX_INPUTS - 1
    ja      .einputs
    cmp     dword [rbx + model.nlayers], 0
    je      .enolayers
    mov     r12d, [rbx + model.inputs]          ; fan into the next layer
    lea     r13, [rbx + model.layers]
    xor     r14d, r14d
    mov     r15d, 1                             ; every layer weighted?
.layer:
    cmp     r14d, [rbx + model.nlayers]
    jae     .wired
    mov     eax, [r13 + layer.neurons]
    test    eax, eax
    jz      .elimit
    mov     [r13 + layer.fan_in], r12d
    mov     ecx, [r13 + layer.nw]
    or      ecx, [r13 + layer.nt]
    jnz     .weighted
    xor     r15d, r15d
    jmp     .next
.weighted:
    cmp     [r13 + layer.nw], eax
    jne     .eweights
    cmp     [r13 + layer.nt], eax
    jne     .eweights
    mov     edi, r12d
    call    xn_fanmask
    xor     ecx, ecx
.mask:
    cmp     ecx, [r13 + layer.neurons]
    jae     .next
    and     [r13 + layer.w + rcx * 8], rax
    movzx   edx, byte [r13 + layer.t + rcx]
    cmp     edx, r12d
    ja      .evalue
    inc     ecx
    jmp     .mask
.next:
    mov     r12d, [r13 + layer.neurons]
    add     r13, XN_LAYER_SIZE
    inc     r14d
    jmp     .layer
.wired:
    cmp     dword [r13 - XN_LAYER_SIZE + layer.neurons], 1
    jne     .eoutput
    mov     edi, [rbx + model.target]
    cmp     edi, XN_T_TRUTH
    je      .explicit
    test    edi, edi
    jnz     .named
    mov     edi, XN_T_XOR                       ; an xor net wants xor by default
    mov     [rbx + model.target], edi
.named:
    mov     esi, [rbx + model.inputs]
    call    xn_target_truth
    mov     [rbx + model.truth], rax
    jmp     .targeted
.explicit:
    mov     edi, [rbx + model.inputs]
    call    xn_rowmask
    and     [rbx + model.truth], rax
.targeted:
    test    r15d, r15d
    jz      .scored
    or      dword [rbx + model.flags], XN_F_WEIGHTS
.scored:
    mov     rdi, rbx
    call    xn_loss
    mov     [rbx + model.loss], rax
    xor     eax, eax
    jmp     .out
.einputs:
    mov     eax, XN_EINPUTS
    jmp     .out
.enolayers:
    mov     eax, XN_ENOLAYERS
    jmp     .out
.elimit:
    mov     eax, XN_ELIMIT
    jmp     .out
.eweights:
    mov     eax, XN_EWEIGHTS
    jmp     .out
.evalue:
    mov     eax, XN_EVALUE
    jmp     .out
.eoutput:
    mov     eax, XN_EOUTPUT
.out:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; xn_load(rdi *model, rsi path) -> eax XN_*    mmap the file, parse in place
xn_load:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, STAT_SIZE
    mov     rbx, rdi
    mov     r12, rsi
    call    xn_model_init
    mov     rdi, r12
    mov     esi, O_RDONLY
    xor     edx, edx
    mov     eax, SYS_open
    syscall
    test    rax, rax
    js      .eio
    mov     r13, rax                ; fd
    mov     edi, r13d
    mov     rsi, rsp
    mov     eax, SYS_fstat
    syscall
    test    rax, rax
    js      .eio_close
    mov     r14, [rsp + STAT_ST_SIZE]
    test    r14, r14
    jz      .empty
    xor     edi, edi
    mov     rsi, r14
    mov     edx, PROT_READ
    mov     r10d, MAP_PRIVATE
    mov     r8, r13
    xor     r9d, r9d
    mov     eax, SYS_mmap
    syscall
    cmp     rax, -4096
    ja      .eio_close
    mov     r15, rax                ; mapping
    mov     edi, r13d
    mov     eax, SYS_close
    syscall
    mov     rdi, rbx
    mov     rsi, r15
    mov     rdx, r14
    call    xn_parse
    mov     r13d, eax
    mov     rdi, r15
    mov     rsi, r14
    mov     eax, SYS_munmap
    syscall
    mov     eax, r13d
    jmp     .out
.empty:
    mov     edi, r13d
    mov     eax, SYS_close
    syscall
    mov     rdi, rbx
    xor     esi, esi
    xor     edx, edx
    call    xn_parse
    jmp     .out
.eio_close:
    mov     edi, r13d
    mov     eax, SYS_close
    syscall
.eio:
    mov     dword [rbx + model.err_line], 0
    mov     eax, XN_EIO
.out:
    add     rsp, STAT_SIZE
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; xn_emit(rdi *model, esi fd) -> eax XN_*    write the organism back out as .xn
xn_emit:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    mov     rbx, rdi
    mov     r12d, esi
    call    xn_io_flush
    mov     r13d, [xn_io_fd]
    mov     [xn_io_fd], r12d

    PUTS    e_head
    mov     rdi, [rbx + model.loss]
    call    xn_io_dec
    PUTS    e_model
    lea     rdi, [rbx + model.name]
    call    xn_io_puts
    PUTS    e_inputs
    mov     edi, [rbx + model.inputs]
    call    xn_io_dec
    PUTS    e_truth
    mov     rdi, [rbx + model.truth]
    call    xn_io_hex

    lea     r15, [rbx + model.layers]
    xor     r14d, r14d
.layer:
    cmp     r14d, [rbx + model.nlayers]
    jae     .options
    PUTS    e_layer
    mov     edi, [r15 + layer.neurons]
    call    xn_io_dec
    PUTS    e_weights
    xor     r12d, r12d
.w:
    cmp     r12d, [r15 + layer.neurons]
    jae     .wdone
    test    r12d, r12d
    jz      .w1
    PUTS    e_sep
.w1:
    mov     rdi, [r15 + layer.w + r12 * 8]
    call    xn_io_hex
    inc     r12d
    jmp     .w
.wdone:
    PUTS    e_thresh
    xor     r12d, r12d
.t:
    cmp     r12d, [r15 + layer.neurons]
    jae     .tdone
    test    r12d, r12d
    jz      .t1
    PUTS    e_sep
.t1:
    movzx   edi, byte [r15 + layer.t + r12]
    call    xn_io_dec
    inc     r12d
    jmp     .t
.tdone:
    add     r15, XN_LAYER_SIZE
    inc     r14d
    jmp     .layer

.options:
    PUTS    e_options
    mov     rdi, [rbx + model.seed]
    call    xn_io_hex
    PUTS    e_epochs
    mov     rdi, [rbx + model.epochs]
    call    xn_io_dec
    PUTS    e_restarts
    mov     edi, [rbx + model.restarts]
    call    xn_io_dec
    PUTS    e_verbose
    mov     edi, [rbx + model.verbose]
    call    xn_io_dec
    PUTS    e_nl
    call    xn_io_flush
    mov     [xn_io_fd], r13d
    test    rax, rax
    mov     eax, XN_OK
    jns     .out
    mov     eax, XN_EIO
.out:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

NOEXECSTACK
