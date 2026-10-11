; ═══════════════════════════════════════════════════════════════════════════
;  io.asm ─ the transmitter.  Buffered output straight onto syscalls, no libc.
;
;  Every routine clobbers only SysV caller-saved registers.
;  The buffer is process-global; the cortex is single-voiced by design.
; ═══════════════════════════════════════════════════════════════════════════
%include "xn.inc"
default rel

IOBUF_SIZE equ 65536

global xn_io_fd, xn_io_capacity
global xn_io_write, xn_io_flush, xn_io_putn, xn_io_puts, xn_io_putc
global xn_io_dec, xn_io_hex, xn_io_bits, xn_io_glyphs, xn_io_spaces, xn_io_strlen

section .data
xn_io_fd:   dd 1

section .rodata
hexdig:     db "0123456789abcdef"

section .bss
alignb 64
iobuf:      resb IOBUF_SIZE
iopos:      resq 1

section .text

xn_io_capacity:
    mov     eax, IOBUF_SIZE
    ret

; xn_io_write(edi fd, rsi buf, rdx len) -> rax 0 | -errno
xn_io_write:
.again:
    test    rdx, rdx
    jz      .ok
    mov     eax, SYS_write
    syscall
    test    rax, rax
    js      .fail
    add     rsi, rax
    sub     rdx, rax
    jmp     .again
.ok:
    xor     eax, eax
.fail:
    ret

; xn_io_flush() -> rax   (drains into [xn_io_fd])
xn_io_flush:
    mov     rdx, [iopos]
    mov     qword [iopos], 0
    lea     rsi, [iobuf]
    mov     edi, [xn_io_fd]
    jmp     xn_io_write

; xn_io_strlen(rdi cstr) -> rax
xn_io_strlen:
    mov     rax, rdi
.scan:
    cmp     byte [rax], 0
    je      .end
    inc     rax
    jmp     .scan
.end:
    sub     rax, rdi
    ret

; xn_io_putn(rdi ptr, rsi len)
xn_io_putn:
    mov     rcx, [iopos]
    lea     rax, [rcx + rsi]
    cmp     rax, IOBUF_SIZE
    jbe     .fit
    push    rdi
    push    rsi
    call    xn_io_flush
    pop     rsi
    pop     rdi
    xor     ecx, ecx
    cmp     rsi, IOBUF_SIZE
    ja      .direct
.fit:
    mov     rdx, rsi                ; len
    mov     rsi, rdi                ; src
    lea     rdi, [iobuf]
    add     rdi, rcx                ; dst
    add     rcx, rdx
    mov     [iopos], rcx
    mov     rcx, rdx
    rep     movsb
    ret
.direct:
    mov     rdx, rsi
    mov     rsi, rdi
    mov     edi, [xn_io_fd]
    jmp     xn_io_write

; xn_io_puts(rdi cstr)
xn_io_puts:
    push    rdi
    call    xn_io_strlen
    pop     rdi
    mov     rsi, rax
    jmp     xn_io_putn

; xn_io_putc(dil byte)
xn_io_putc:
    sub     rsp, 8
    mov     [rsp], dil
    mov     rdi, rsp
    mov     esi, 1
    call    xn_io_putn
    add     rsp, 8
    ret

; xn_io_spaces(edi n)
xn_io_spaces:
    push    rbx
    mov     ebx, edi
.more:
    test    ebx, ebx
    jle     .done
    mov     edi, ' '
    call    xn_io_putc
    dec     ebx
    jmp     .more
.done:
    pop     rbx
    ret

; xn_io_dec(rdi u64)
xn_io_dec:
    sub     rsp, 40
    lea     r8, [rsp + 32]
    mov     r9, r8
    mov     rax, rdi
    mov     ecx, 10
.digit:
    xor     edx, edx
    div     rcx
    add     dl, '0'
    dec     r9
    mov     [r9], dl
    test    rax, rax
    jnz     .digit
    mov     rdi, r9
    mov     rsi, r8
    sub     rsi, r9
    call    xn_io_putn
    add     rsp, 40
    ret

; xn_io_hex(rdi u64)  ->  "0x…" lowercase, minimal digits
xn_io_hex:
    sub     rsp, 40
    lea     r8, [rsp + 32]
    mov     r9, r8
    mov     rax, rdi
    lea     r10, [hexdig]
.nib:
    mov     edx, eax
    and     edx, 15
    mov     dl, [r10 + rdx]
    dec     r9
    mov     [r9], dl
    shr     rax, 4
    jnz     .nib
    dec     r9
    mov     byte [r9], 'x'
    dec     r9
    mov     byte [r9], '0'
    mov     rdi, r9
    mov     rsi, r8
    sub     rsi, r9
    call    xn_io_putn
    add     rsp, 40
    ret

; xn_io_bits(rdi value, esi n)  ->  n chars of 0/1, most significant first
xn_io_bits:
    sub     rsp, 72
    mov     ecx, esi
    xor     edx, edx
.bit:
    test    ecx, ecx
    jz      .emit
    dec     ecx
    bt      rdi, rcx
    setc    al
    add     al, '0'
    mov     [rsp + rdx], al
    inc     edx
    jmp     .bit
.emit:
    mov     rdi, rsp
    mov     esi, edx
    call    xn_io_putn
    add     rsp, 72
    ret

; xn_io_glyphs(rdi value, esi n)  ->  ▮ for 1, ▯ for 0, most significant first
xn_io_glyphs:
    sub     rsp, 200
    mov     ecx, esi
    xor     edx, edx
.g:
    test    ecx, ecx
    jz      .emit
    dec     ecx
    mov     byte [rsp + rdx], 0xE2
    mov     byte [rsp + rdx + 1], 0x96
    bt      rdi, rcx
    setnc   al
    add     al, 0xAE                ; U+25AE ▮ / U+25AF ▯
    mov     [rsp + rdx + 2], al
    add     edx, 3
    jmp     .g
.emit:
    mov     rdi, rsp
    mov     esi, edx
    call    xn_io_putn
    add     rsp, 200
    ret

NOEXECSTACK
