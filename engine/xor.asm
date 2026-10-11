; ═══════════════════════════════════════════════════════════════════════════
;  xor.asm ─ the primitive.  One gate, emulating itself.
;
;      cell(x, w, fan, t)  =  popcnt((x ⊕ w) & fan)  >=  t
;
;  A cell XORs its input against its weight, counts the disagreements, and
;  fires once enough of them pile up.  Everything else in xn is bookkeeping.
; ═══════════════════════════════════════════════════════════════════════════
%include "xn.inc"
default rel

global xn_cell, xn_parity, xn_hamming, xn_fanmask, xn_rowmask
global xn_rng_next, xn_rng_below, xn_target_truth

section .text

; xn_cell(rdi x, rsi w, rdx fan, ecx t) -> rax 0|1
xn_cell:
    xor     rdi, rsi
    and     rdi, rdx
    popcnt  rdi, rdi
    xor     eax, eax
    cmp     rdi, rcx
    setae   al
    ret

; xn_parity(rdi x) -> rax  ⊕-fold of every bit
xn_parity:
    popcnt  rax, rdi
    and     eax, 1
    ret

; xn_hamming(rdi a, rsi b) -> rax  popcnt(a ⊕ b)
xn_hamming:
    xor     rdi, rsi
    popcnt  rax, rdi
    ret

; xn_fanmask(edi k) -> rax  low k bits set (k >= 64 → all)
xn_fanmask:
    mov     rax, -1
    cmp     edi, 64
    jae     .all
    mov     ecx, edi
    mov     eax, 1
    shl     rax, cl
    dec     rax
.all:
    ret

; xn_rowmask(edi inputs) -> rax  one bit per truth-table row
xn_rowmask:
    mov     ecx, edi
    mov     edi, 1
    shl     edi, cl                 ; rows = 2^inputs   (inputs <= 6)
    jmp     xn_fanmask

; xn_rng_next(rdi *state) -> rax   xorshift64*: the stochastic heartbeat
xn_rng_next:
    mov     rax, [rdi]
    test    rax, rax
    jnz     .live
    mov     rax, 0x9E3779B97F4A7C15 ; a dead state is reborn golden
.live:
    mov     rdx, rax
    shr     rdx, 12
    xor     rax, rdx
    mov     rdx, rax
    shl     rdx, 25
    xor     rax, rdx
    mov     rdx, rax
    shr     rdx, 27
    xor     rax, rdx
    mov     [rdi], rax
    mov     rdx, 0x2545F4914F6CDD1D
    imul    rax, rdx
    ret

; xn_rng_below(rdi *state, rsi n) -> rax in [0, n)   (n > 0)
xn_rng_below:
    push    rsi
    call    xn_rng_next
    pop     rcx
    shr     rax, 11                 ; drop the weaker low bits
    xor     edx, edx
    div     rcx
    mov     rax, rdx
    ret

; xn_target_truth(edi kind, esi inputs) -> rax  truth word for a named target
;   bit x of the result is the desired output for input row x
xn_target_truth:
    mov     r8d, edi                ; kind
    mov     ecx, esi
    mov     r9d, 1
    shl     r9d, cl                 ; rows
    lea     r10d, [r9d - 1]         ; the all-ones row
    xor     eax, eax                ; truth
    xor     r11d, r11d              ; x
.row:
    cmp     r11d, r9d
    jae     .done
    popcnt  edx, r11d
    and     edx, 1                  ; parity(x)
    cmp     r8d, XN_T_XOR
    je      .set_if
    cmp     r8d, XN_T_XNOR
    je      .flip
    xor     edx, edx
    cmp     r11d, r10d
    sete    dl                      ; all(x)
    cmp     r8d, XN_T_AND
    je      .set_if
    cmp     r8d, XN_T_NAND
    je      .flip
    xor     edx, edx
    test    r11d, r11d
    setnz   dl                      ; any(x)
    cmp     r8d, XN_T_OR
    je      .set_if
    cmp     r8d, XN_T_NOR
    je      .flip
    xor     edx, edx                ; unknown kind: silence
    jmp     .set_if
.flip:
    xor     edx, 1
.set_if:
    test    edx, edx
    jz      .next
    bts     rax, r11
.next:
    inc     r11d
    jmp     .row
.done:
    ret

NOEXECSTACK
