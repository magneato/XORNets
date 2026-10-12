; ═══════════════════════════════════════════════════════════════════════════
;  mnist.asm ─ MNIST Digit Classification on the XN 1-Bit Platform
;  Layer Stacking Geometry: 4 Bit Planes + Inter-Stack XOR Compute (7 Planks)
;
;  Input:  784 bytes (28x28 grayscale image, [0..255])
;  Stack:  12x12 spatial pooling (144 bits per plank)
;          Planks 0..3: Bit planes (>32, >96, >160, >224)
;          Planks 4..6: Inter-stack XOR (P0^P1, P1^P2, P2^P3)
;          Total 7 planks = 1008 stacked bit features (21 x 64-bit words)
;  Hidden: 64 neurons, 16-bit accumulator, clamped Q8 activation
;  Output: 10 logits (argmax = predicted digit 0..9)
; ═══════════════════════════════════════════════════════════════════════════
%include "xn.inc"
default rel

global xn_mnist_preprocess
global xn_mnist_infer
global xn_mnist_infer_image
global xn_mnist_test_sample
global xn_mnist_test_count

%include "mnist_weights.inc"
%include "mnist_sample_data.inc"

section .text

; ── xn_mnist_test_count() -> rax
xn_mnist_test_count:
    mov     rax, 50
    ret

; ── xn_mnist_test_sample(rdi idx, rsi *out_label) -> rax (const uint8_t *img784)
xn_mnist_test_sample:
    cmp     rdi, 50
    jae     .invalid
    test    rsi, rsi
    jz      .no_label
    lea     rax, [xn_mnist_test_labels]
    movzx   eax, byte [rax + rdi]
    mov     byte [rsi], al
.no_label:
    imul    rax, rdi, 784
    lea     rdx, [xn_mnist_test_images]
    add     rax, rdx
    ret
.invalid:
    xor     eax, eax
    ret

; ── xn_mnist_preprocess(rdi const uint8_t *img784, rsi xn_mnist_stack *out)
; Preprocesses 28x28 image into 7 stacked planks (21 uint64_t words).
; Each plank has 144 bits (stored across 3 uint64 words: w0 [0..63], w1 [64..127], w2 [128..143]).
; Total: 7 planks * 3 words = 21 words.
xn_mnist_preprocess:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    push    rbp
    sub     rsp, 160            ; local buffer for 144 pooled pixels

    mov     rbx, rdi            ; img784
    mov     r12, rsi            ; out_stack

    ; Clear local buffer (144 bytes)
    lea     rdi, [rsp]
    xor     eax, eax
    mov     ecx, 144
    rep     stosb

    ; Compute 12x12 pooling from 24x24 crop (rows 2..25, cols 2..25)
    ; Pixel (r, c) in 12x12 = average of 4 pixels in 28x28 at:
    ;   row0 = 2 + r*2, col0 = 2 + c*2
    xor     r13d, r13d          ; r in 0..11
.row_loop:
    cmp     r13d, 12
    jae     .pooling_done
    xor     r14d, r14d          ; c in 0..11
.col_loop:
    cmp     r14d, 12
    jae     .row_next

    ; Base index in img784:
    ; r0 = 2 + r*2, c0 = 2 + c*2
    lea     eax, [r13d * 2 + 2] ; r0
    imul    eax, eax, 28
    lea     ecx, [r14d * 2 + 2] ; c0
    add     eax, ecx            ; offset of (r0, c0)

    movzx   edx, byte [rbx + rax]
    movzx   ecx, byte [rbx + rax + 1]
    add     edx, ecx
    movzx   ecx, byte [rbx + rax + 28]
    add     edx, ecx
    movzx   ecx, byte [rbx + rax + 29]
    add     edx, ecx
    shr     edx, 2              ; mean = sum / 4

    ; Store in local buffer at r13*12 + r14
    imul    eax, r13d, 12
    add     eax, r14d
    mov     [rsp + rax], dl

    inc     r14d
    jmp     .col_loop
.row_next:
    inc     r13d
    jmp     .row_loop

.pooling_done:
    ; Zero out the 21 destination words
    mov     rdi, r12
    xor     eax, eax
    mov     ecx, 21
    rep     stosq

    ; Construct base planks 0..3: thresholds (32, 96, 160, 224)
    ; Plank p words start at r12 + p * 24 bytes
    xor     ebp, ebp            ; plank p in 0..3
.plank_loop:
    cmp     ebp, 4
    jae     .base_done

    ; Set threshold for this plank
    cmp     ebp, 0
    jne     .check_p1
    mov     r15d, 32
    jmp     .thresh_ready
.check_p1:
    cmp     ebp, 1
    jne     .check_p2
    mov     r15d, 96
    jmp     .thresh_ready
.check_p2:
    cmp     ebp, 2
    jne     .check_p3
    mov     r15d, 160
    jmp     .thresh_ready
.check_p3:
    mov     r15d, 224
.thresh_ready:

    ; Scan 144 cells
    xor     ecx, ecx            ; bit cell in 0..143
.cell_loop:
    cmp     ecx, 144
    jae     .plank_next
    movzx   eax, byte [rsp + rcx]
    cmp     eax, r15d
    jbe     .bit_off

    ; Set bit ecx in plank ebp
    ; Word index = ecx / 64, bit index = ecx % 64
    mov     eax, ecx
    shr     eax, 6              ; 0, 1, or 2
    lea     edx, [rbp + rbp * 2] ; ebp * 3
    add     edx, eax
    mov     r8d, ecx
    and     r8d, 63             ; bit shift
    bts     qword [r12 + rdx * 8], r8

.bit_off:
    inc     ecx
    jmp     .cell_loop

.plank_next:
    inc     ebp
    jmp     .plank_loop

.base_done:
    ; Inter-stack XOR compute:
    ; Plank 4 = Plank 0 ^ Plank 1
    ; Plank 5 = Plank 1 ^ Plank 2
    ; Plank 6 = Plank 2 ^ Plank 3
    ; Each plank is 3 qwords.
    xor     ecx, ecx
.xor_loop:
    cmp     ecx, 3
    jae     .finish_preprocess

    ; Plank 4: words 12..14
    mov     rax, [r12 + rcx * 8]            ; Plank 0
    xor     rax, [r12 + 24 + rcx * 8]       ; ^ Plank 1
    mov     [r12 + 96 + rcx * 8], rax       ; -> Plank 4

    ; Plank 5: words 15..17
    mov     rax, [r12 + 24 + rcx * 8]       ; Plank 1
    xor     rax, [r12 + 48 + rcx * 8]       ; ^ Plank 2
    mov     [r12 + 120 + rcx * 8], rax      ; -> Plank 5

    ; Plank 6: words 18..20
    mov     rax, [r12 + 48 + rcx * 8]       ; Plank 2
    xor     rax, [r12 + 72 + rcx * 8]       ; ^ Plank 3
    mov     [r12 + 144 + rcx * 8], rax      ; -> Plank 6

    inc     ecx
    jmp     .xor_loop

.finish_preprocess:
    add     rsp, 160
    pop     rbp
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret


; ── xn_mnist_infer(rdi const xn_mnist_stack *stack, rsi int32_t *out_logits) -> rax (predicted digit 0..9)
xn_mnist_infer:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    push    rbp
    sub     rsp, 208            ; 128 bytes for acc1 (64 x int16), 64 bytes for h (64 x uint8)
    ; Aligned properly: 6 pushes = 48 bytes + 8 (ret) + 208 = 264 -> 16-aligned

    mov     rbx, rdi            ; stack (21 qwords = 1008 bits)
    mov     r12, rsi            ; out_logits (can be NULL)

    ; Initialize acc1 with biases b1 (64 int16 words)
    lea     rsi, [xn_mnist_b1]
    lea     rdi, [rsp]
    mov     ecx, 16             ; 16 qwords = 128 bytes = 64 words
    rep     movsq

    ; Accumulate weights for every active bit across the 1008 bits
    ; Iterate 21 qwords
    xor     r13d, r13d          ; word index 0..20
.word_loop:
    cmp     r13d, 21
    jae     .acc_done

    mov     r14, [rbx + r13 * 8] ; load 64 bits
.bit_scan:
    test    r14, r14
    jz      .next_word

    ; Find lowest set bit
    tzcnt   r8, r14
    btr     r14, r8             ; clear this bit
    ; r8 = bit within current 64-bit word (0..63)

    ; Map r13 (word in 0..20) to plank base bit offset:
    ; Each plank is 3 words:
    ;   plank = r13 / 3
    ;   subword = r13 % 3
    ; base_bit = plank * 144 + subword * 64
    ; Let us calculate:
    mov     eax, r13d
    xor     edx, edx
    mov     ecx, 3
    div     ecx                 ; eax = plank (0..6), edx = subword (0..2)
    imul    eax, eax, 144       ; plank * 144
    shl     edx, 6              ; subword * 64
    add     eax, edx
    add     rax, r8             ; bit_index = base_bit + r8

    ; Add W1[bit_index, 0..63] (64 int8 values) to acc1[0..63] (64 int16 values)
    ; W1 row offset = bit_index * 64
    lea     rdx, [xn_mnist_w1]
    shl     rax, 6
    add     rdx, rax            ; rdx = &W1[bit_index][0]


    ; Add 64 int8s to 64 int16s (unrolled by 8 elements)
    xor     ecx, ecx
.add_weights:
    cmp     ecx, 64
    jae     .bit_scan

    movsx   eax, byte [rdx + rcx]
    add     word [rsp + rcx * 2], ax

    movsx   eax, byte [rdx + rcx + 1]
    add     word [rsp + rcx * 2 + 2], ax

    movsx   eax, byte [rdx + rcx + 2]
    add     word [rsp + rcx * 2 + 4], ax

    movsx   eax, byte [rdx + rcx + 3]
    add     word [rsp + rcx * 2 + 6], ax

    add     ecx, 4
    jmp     .add_weights

.next_word:
    inc     r13d
    jmp     .word_loop

.acc_done:
    ; Compute hidden activations: h[j] = clamp(acc1[j] >> K1, 0, 255)
    ; K1 = 4
    xor     ecx, ecx
.clamp_loop:
    cmp     ecx, 64
    jae     .layer2
    movsx   eax, word [rsp + rcx * 2]
    sar     eax, MNIST_K1
    test    eax, eax
    jns     .check_max
    xor     eax, eax
    jmp     .store_h
.check_max:
    cmp     eax, 255
    jle     .store_h
    mov     eax, 255
.store_h:
    mov     [rsp + 128 + rcx], al
    inc     ecx
    jmp     .clamp_loop

.layer2:
    ; Compute 10 output logits:
    ; logit[k] = b2[k] + sum_{j=0..63} h[j] * W2[j, k]
    ; Track best logit and best digit
    mov     r15d, 0x80000000    ; best_score = INT32_MIN
    xor     ebp, ebp            ; best_digit = 0
    xor     r13d, r13d          ; digit k in 0..9

.digit_loop:
    cmp     r13d, 10
    jae     .done_logits

    ; Load bias b2[k]
    lea     rax, [xn_mnist_b2]
    mov     eax, [rax + r13 * 4] ; acc2 = b2[k]

    ; Multiply-accumulate over 64 hidden units
    lea     rdx, [xn_mnist_w2]
    ; W2 is shaped [64, 10]
    xor     ecx, ecx
.mac_loop:
    cmp     ecx, 64
    jae     .digit_scored

    ; h[ecx]
    movzx   r8d, byte [rsp + 128 + rcx]
    test    r8d, r8d
    jz      .mac_skip

    ; W2[ecx, r13] offset = ecx * 10 + r13
    imul    r9d, ecx, 10
    add     r9d, r13d
    movsx   r10d, byte [rdx + r9]
    imul    r10d, r8d
    add     eax, r10d

.mac_skip:
    inc     ecx
    jmp     .mac_loop

.digit_scored:
    ; Save logit if out_logits != NULL
    test    r12, r12
    jz      .check_best
    mov     [r12 + r13 * 4], eax

.check_best:
    cmp     eax, r15d
    jle     .digit_next
    mov     r15d, eax
    mov     ebp, r13d

.digit_next:
    inc     r13d
    jmp     .digit_loop

.done_logits:
    mov     eax, ebp            ; return best digit (0..9)
    add     rsp, 208
    pop     rbp
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret


; ── xn_mnist_infer_image(rdi const uint8_t *img784, rsi int32_t *out_logits) -> rax digit
xn_mnist_infer_image:
    push    rbx
    push    r12
    sub     rsp, 168            ; 21 * 8 bytes = 168 bytes for xn_mnist_stack
    mov     rbx, rdi            ; img784
    mov     r12, rsi            ; out_logits

    lea     rsi, [rsp]
    call    xn_mnist_preprocess

    lea     rdi, [rsp]
    mov     rsi, r12
    call    xn_mnist_infer

    add     rsp, 168
    pop     r12
    pop     rbx
    ret

NOEXECSTACK
