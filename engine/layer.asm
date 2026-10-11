; ═══════════════════════════════════════════════════════════════════════════
;  layer.asm ─ a rank of XOR cells sharing one input vector.
;
;  A layer reads a bit-vector of width fan_in and emits a bit-vector of width
;  neurons: bit n of the output is cell n.  Vectors never leave a register.
; ═══════════════════════════════════════════════════════════════════════════
%include "xn.inc"
default rel

extern xn_fanmask, xn_rng_next, xn_rng_below

global xn_layer_eval, xn_layer_randomize, xn_layer_mutate, xn_layer_restore

section .text

; xn_layer_eval(rdi *layer, rsi v) -> rax output vector
xn_layer_eval:
    mov     ecx, [rdi + layer.fan_in]
    mov     rdx, -1
    cmp     ecx, 64
    jae     .wide
    mov     edx, 1
    shl     rdx, cl
    dec     rdx                     ; fan mask
.wide:
    mov     r8d, [rdi + layer.neurons]
    xor     eax, eax
    xor     r9d, r9d
.cell:
    cmp     r9d, r8d
    jae     .done
    mov     r10, rsi
    xor     r10, [rdi + layer.w + r9 * 8]
    and     r10, rdx
    popcnt  r10, r10
    movzx   r11d, byte [rdi + layer.t + r9]
    cmp     r10, r11
    jb      .quiet
    bts     rax, r9
.quiet:
    inc     r9d
    jmp     .cell
.done:
    ret

; xn_layer_randomize(rdi *layer, rsi *rng)   fresh weights and thresholds
xn_layer_randomize:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    mov     rbx, rdi
    mov     rbp, rsi
    mov     edi, [rbx + layer.fan_in]
    call    xn_fanmask
    mov     r12, rax                ; fan mask
    xor     r13d, r13d              ; n
.cell:
    cmp     r13d, [rbx + layer.neurons]
    jae     .done
    mov     rdi, rbp
    call    xn_rng_next
    and     rax, r12
    mov     [rbx + layer.w + r13 * 8], rax
    mov     rdi, rbp
    mov     esi, [rbx + layer.fan_in]
    inc     esi
    call    xn_rng_below
    mov     [rbx + layer.t + r13], al
    inc     r13d
    jmp     .cell
.done:
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; xn_layer_mutate(rdi *layer, rsi *rng) -> rax undo code, rdx old weight
;   ¾ of the time flip one weight bit, otherwise re-roll one threshold.
;   undo code = neuron | old_threshold << 8
xn_layer_mutate:
    push    rbx
    push    rbp
    push    r12
    push    r13
    mov     rbx, rdi
    mov     rbp, rsi
    mov     rdi, rbp
    mov     esi, [rbx + layer.neurons]
    call    xn_rng_below
    mov     r12d, eax               ; neuron
    mov     r13, [rbx + layer.w + r12 * 8]      ; old weight
    movzx   eax, byte [rbx + layer.t + r12]
    shl     eax, 8
    or      r12d, eax               ; undo code
    mov     rdi, rbp
    call    xn_rng_next
    test    eax, 0x300
    jz      .threshold
    mov     rdi, rbp
    mov     esi, [rbx + layer.fan_in]
    call    xn_rng_below
    movzx   ecx, r12b
    btc     [rbx + layer.w + rcx * 8], rax
    jmp     .out
.threshold:
    mov     rdi, rbp
    mov     esi, [rbx + layer.fan_in]
    inc     esi
    call    xn_rng_below
    movzx   ecx, r12b
    mov     [rbx + layer.t + rcx], al
.out:
    mov     eax, r12d
    mov     rdx, r13
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; xn_layer_restore(rdi *layer, esi undo code, rdx old weight)
xn_layer_restore:
    movzx   ecx, sil
    mov     [rdi + layer.w + rcx * 8], rdx
    shr     esi, 8
    mov     [rdi + layer.t + rcx], sil
    ret

NOEXECSTACK
