; ═══════════════════════════════════════════════════════════════════════════
;  inference.asm ─ listening.  A row goes in as bits, one bit comes out.
; ═══════════════════════════════════════════════════════════════════════════
%include "xn.inc"
default rel

extern xn_layer_eval, xn_rowmask

global xn_forward, xn_truth, xn_loss

section .text

; xn_forward(rdi *model, rsi x) -> rax 0|1
xn_forward:
    push    rbx
    push    r12
    push    r13
    lea     rbx, [rdi + model.layers]
    mov     r12d, [rdi + model.nlayers]
    mov     rax, rsi
.layer:
    test    r12d, r12d
    jz      .done
    mov     rdi, rbx
    mov     rsi, rax
    call    xn_layer_eval
    add     rbx, XN_LAYER_SIZE
    dec     r12d
    jmp     .layer
.done:
    and     eax, 1
    pop     r13
    pop     r12
    pop     rbx
    ret

; xn_truth(rdi *model) -> rax  the cortex's own truth word, one bit per row
xn_truth:
    push    rbx
    push    r12
    push    r13
    push    r14
    sub     rsp, 8
    mov     rbx, rdi
    mov     ecx, [rbx + model.inputs]
    mov     r12d, 1
    shl     r12d, cl                ; rows
    xor     r13d, r13d              ; x
    xor     r14d, r14d              ; truth
.row:
    cmp     r13d, r12d
    jae     .done
    mov     rdi, rbx
    mov     esi, r13d
    call    xn_forward
    test    eax, eax
    jz      .next
    bts     r14, r13
.next:
    inc     r13d
    jmp     .row
.done:
    mov     rax, r14
    add     rsp, 8
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; xn_loss(rdi *model) -> rax  rows where cortex ⊕ target = 1
xn_loss:
    push    rbx
    mov     rbx, rdi
    call    xn_truth
    xor     rax, [rbx + model.truth]
    push    rax
    mov     edi, [rbx + model.inputs]
    call    xn_rowmask
    pop     rdx
    and     rax, rdx
    popcnt  rax, rax
    pop     rbx
    ret

NOEXECSTACK
