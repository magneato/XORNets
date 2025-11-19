;---------------------------------------------------------
; WDR1_KERNEL.ASM — Single Bit Stochastic Processor
; "A bit is not a state, it is a probability frozen in time."
;---------------------------------------------------------
global wdr1_process

wdr1_process:
 ; RDI = Input Stream (64 bits of history)
 ; RSI = Weights
 ; RDX = Threshold
 
 ; 1. Apply Weights (XNOR - correlate bits)
 mov   rax, rdi
 xnor  rax, rsi      ; Custom macro for NOT XOR
 
 ; 2. Population Count (Integrate over time)
 ; Collapses the 64-bit time dimension into a spatial scalar
 popcnt rax, rax
 
 ; 3. Activation (Step function)
 cmp   rax, rdx
 jg   .fire
 xor   rax, rax      ; Output 0
 ret
 
.fire:
 mov   rax, 1       ; Output 1
 ret
