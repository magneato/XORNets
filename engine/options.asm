; ═══════════════════════════════════════════════════════════════════════════
;  options.asm ─ knobs, and the little text scanners everyone shares.
;
;  The same xn_option_set serves the [options] section of a .xn file and the
;  --seed / --epochs / --restarts / --verbose overrides on the command line.
; ═══════════════════════════════════════════════════════════════════════════
%include "xn.inc"
default rel

global xn_option_set, xn_text_eq, xn_text_u64

section .rodata
k_seed:     db "seed", 0
k_epochs:   db "epochs", 0
k_restarts: db "restarts", 0
k_verbose:  db "verbose", 0

section .text

; xn_text_eq(rdi ptr, rsi len, rdx cstr) -> eax 1 if the slice equals cstr
xn_text_eq:
    xor     ecx, ecx
.cmp:
    movzx   eax, byte [rdx + rcx]
    cmp     rcx, rsi
    je      .end_of_slice
    test    al, al
    jz      .no
    cmp     al, [rdi + rcx]
    jne     .no
    inc     rcx
    jmp     .cmp
.end_of_slice:
    test    al, al
    jnz     .no
    mov     eax, 1
    ret
.no:
    xor     eax, eax
    ret

; xn_text_u64(rdi ptr, rsi len) -> rax value, edx 0 ok | 1 bad
;   decimal, 0x hex or 0b binary; '_' may separate digit groups
xn_text_u64:
    xor     eax, eax
    mov     r8d, 10
    cmp     rsi, 2
    jb      .digits
    cmp     byte [rdi], '0'
    jne     .digits
    movzx   ecx, byte [rdi + 1]
    or      ecx, 0x20
    cmp     ecx, 'x'
    jne     .bin
    mov     r8d, 16
    jmp     .prefixed
.bin:
    cmp     ecx, 'b'
    jne     .digits
    mov     r8d, 2
.prefixed:
    add     rdi, 2
    sub     rsi, 2
.digits:
    xor     r9d, r9d                ; digits seen
.next:
    test    rsi, rsi
    jz      .end
    movzx   ecx, byte [rdi]
    inc     rdi
    dec     rsi
    cmp     ecx, '_'
    je      .next
    lea     r11d, [rcx - '0']
    cmp     r11d, 9
    jbe     .have
    or      ecx, 0x20
    lea     r11d, [rcx - 'a' + 10]
    cmp     ecx, 'a'
    jb      .bad
    cmp     ecx, 'f'
    ja      .bad
.have:
    cmp     r11d, r8d
    jae     .bad
    mul     r8
    jc      .bad
    add     rax, r11
    jc      .bad
    inc     r9d
    jmp     .next
.end:
    test    r9d, r9d
    jz      .bad
    xor     edx, edx
    ret
.bad:
    mov     edx, 1
    ret

; xn_option_set(rdi *model, rsi key, rdx keylen, rcx val, r8 vallen) -> eax XN_*
xn_option_set:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    mov     rbx, rdi
    mov     r12, rsi
    mov     r13, rdx
    mov     rdi, rcx
    mov     rsi, r8
    call    xn_text_u64
    mov     r14, rax                ; value
    mov     r15d, edx               ; parse failure

    mov     rdi, r12
    mov     rsi, r13
    lea     rdx, [k_seed]
    call    xn_text_eq
    test    eax, eax
    jnz     .seed
    mov     rdi, r12
    mov     rsi, r13
    lea     rdx, [k_epochs]
    call    xn_text_eq
    test    eax, eax
    jnz     .epochs
    mov     rdi, r12
    mov     rsi, r13
    lea     rdx, [k_restarts]
    call    xn_text_eq
    test    eax, eax
    jnz     .restarts
    mov     rdi, r12
    mov     rsi, r13
    lea     rdx, [k_verbose]
    call    xn_text_eq
    test    eax, eax
    jnz     .verbose
    mov     eax, XN_EKEY
    jmp     .out

.seed:
    test    r15d, r15d
    jnz     .bad
    mov     [rbx + model.seed], r14
    jmp     .ok
.epochs:
    test    r15d, r15d
    jnz     .bad
    test    r14, r14
    jz      .limit
    mov     [rbx + model.epochs], r14
    jmp     .ok
.restarts:
    test    r15d, r15d
    jnz     .bad
    test    r14, r14
    jz      .limit
    cmp     r14, 1000000
    ja      .limit
    mov     [rbx + model.restarts], r14d
    jmp     .ok
.verbose:
    test    r15d, r15d
    jnz     .bad
    cmp     r14, 9
    ja      .limit
    mov     [rbx + model.verbose], r14d
.ok:
    xor     eax, eax
    jmp     .out
.limit:
    mov     eax, XN_ELIMIT
    jmp     .out
.bad:
    mov     eax, XN_EVALUE
.out:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

NOEXECSTACK
