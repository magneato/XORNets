; ═══════════════════════════════════════════════════════════════════════════
;  train.asm ─ learning without gradients.
;
;  A 1-bit cortex has no slope to descend, so xn learns by stochastic bit-flip
;  search: mutate one cell, keep the change if the loss did not grow, undo it
;  otherwise.  When a run stalls, the cortex is re-rolled (a restart) and the
;  best organism ever seen is kept.  The snapshot lives on the stack, so
;  independent models may be trained concurrently.
; ═══════════════════════════════════════════════════════════════════════════
%include "xn.inc"
default rel

extern xn_layer_randomize, xn_layer_mutate, xn_layer_restore
extern xn_loss, xn_rng_below

global xn_train

FRAME_RNG   equ 0
FRAME_SNAP  equ 16
FRAME_SIZE  equ XN_LAYERS_BYTES + 24        ; keeps rsp 16-aligned after 6 pushes

section .text

; xn_train(rdi *model) -> rax best loss
xn_train:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, FRAME_SIZE
    mov     rbx, rdi

    mov     rax, [rbx + model.seed]
    mov     rdx, 0x9E3779B97F4A7C15
    xor     rax, rdx
    mov     [rsp + FRAME_RNG], rax
    mov     qword [rbx + model.steps], 0
    mov     dword [rbx + model.used], 0
    mov     r15, -1                 ; best loss
    xor     r12d, r12d              ; restart

.restart:
    mov     eax, [rbx + model.restarts]
    cmp     eax, 1
    adc     eax, 0                  ; at least one attempt
    cmp     r12d, eax
    jae     .finish

    ; ── re-roll every layer ──
    xor     r13d, r13d
.roll:
    cmp     r13d, [rbx + model.nlayers]
    jae     .rolled
    imul    edi, r13d, XN_LAYER_SIZE
    lea     rdi, [rbx + model.layers + rdi]
    lea     rsi, [rsp + FRAME_RNG]
    call    xn_layer_randomize
    inc     r13d
    jmp     .roll
.rolled:
    mov     rdi, rbx
    call    xn_loss
    mov     r14, rax                ; current loss
    xor     r13d, r13d              ; epoch

.step:
    test    r14, r14
    jz      .judge
    cmp     r13, [rbx + model.epochs]
    jae     .judge
    lea     rdi, [rsp + FRAME_RNG]
    mov     esi, [rbx + model.nlayers]
    call    xn_rng_below
    imul    eax, eax, XN_LAYER_SIZE
    lea     rbp, [rbx + model.layers + rax]     ; victim layer
    mov     rdi, rbp
    lea     rsi, [rsp + FRAME_RNG]
    call    xn_layer_mutate
    mov     [rsp + 8], rdx          ; old weight
    mov     r8, rax                 ; undo code
    push    r8
    sub     rsp, 8
    mov     rdi, rbx
    call    xn_loss
    add     rsp, 8
    pop     r8
    cmp     rax, r14
    ja      .reject
    mov     r14, rax                ; sideways moves are allowed: drift helps
    jmp     .stepped
.reject:
    mov     rdi, rbp
    mov     esi, r8d
    mov     rdx, [rsp + 8]
    call    xn_layer_restore
.stepped:
    inc     r13
    inc     qword [rbx + model.steps]
    jmp     .step

.judge:
    inc     r12d
    cmp     r14, r15
    jae     .worse
    mov     r15, r14
    mov     [rbx + model.used], r12d
    lea     rsi, [rbx + model.layers]
    lea     rdi, [rsp + FRAME_SNAP]
    mov     ecx, XN_LAYERS_BYTES
    rep     movsb
.worse:
    test    r15, r15
    jnz     .restart

.finish:
    lea     rsi, [rsp + FRAME_SNAP]
    lea     rdi, [rbx + model.layers]
    mov     ecx, XN_LAYERS_BYTES
    rep     movsb
    mov     [rbx + model.loss], r15
    or      dword [rbx + model.flags], XN_F_TRAINED
    mov     rax, r15
    add     rsp, FRAME_SIZE
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

NOEXECSTACK
