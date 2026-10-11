; =========================================================================
; WDR1_KERNEL.ASM - XORNets AI Kernel
; =========================================================================
; XORNets - AI using XOR operations
; This file contains the core implementation of the XOR-based neural network
; =========================================================================

.MODEL SMALL
.STACK 100h

.DATA
    ; Neural network weights and biases
    w1_1    DW 1    ; Weight from input 1 to hidden neuron 1
    w1_2    DW 1    ; Weight from input 2 to hidden neuron 1
    w2_1    DW 1    ; Weight from input 1 to hidden neuron 2
    w2_2    DW 1    ; Weight from input 2 to hidden neuron 2
    w3_1    DW 1    ; Weight from hidden 1 to output
    w3_2    DW 1    ; Weight from hidden 2 to output
    
    ; Training data for XOR function
    ; Input pairs and expected outputs
    input1_1    DB 0    ; First input of pair 1
    input1_2    DB 0    ; Second input of pair 1
    output1     DB 0    ; Expected output: 0 XOR 0 = 0
    
    input2_1    DB 0    ; First input of pair 2
    input2_2    DB 1    ; Second input of pair 2
    output2     DB 1    ; Expected output: 0 XOR 1 = 1
    
    input3_1    DB 1    ; First input of pair 3
    input3_2    DB 0    ; Second input of pair 3
    output3     DB 1    ; Expected output: 1 XOR 0 = 1
    
    input4_1    DB 1    ; First input of pair 4
    input4_2    DB 1    ; Second input of pair 4
    output4     DB 0    ; Expected output: 1 XOR 1 = 0
    
    ; Messages
    msg_start   DB 'XORNets AI Kernel v1.0', 0Dh, 0Ah, '$'
    msg_train   DB 'Training XOR network...', 0Dh, 0Ah, '$'
    msg_done    DB 'Training complete!', 0Dh, 0Ah, '$'
    msg_result  DB 'Result: $'
    
    ; Working variables
    hidden1     DW 0    ; Hidden neuron 1 activation
    hidden2     DW 0    ; Hidden neuron 2 activation
    output      DW 0    ; Output neuron activation
    temp        DW 0    ; Temporary storage
    
.CODE
MAIN PROC
    ; Initialize data segment
    MOV AX, @DATA
    MOV DS, AX
    
    ; Display startup message
    LEA DX, msg_start
    MOV AH, 09h
    INT 21h
    
    ; Display training message
    LEA DX, msg_train
    MOV AH, 09h
    INT 21h
    
    ; Train the network
    CALL TRAIN_NETWORK
    
    ; Display completion message
    LEA DX, msg_done
    MOV AH, 09h
    INT 21h
    
    ; Test the network
    CALL TEST_NETWORK
    
    ; Exit program
    MOV AH, 4Ch
    INT 21h
MAIN ENDP

; =========================================================================
; TRAIN_NETWORK - Trains the XOR neural network
; =========================================================================
TRAIN_NETWORK PROC
    PUSH AX
    PUSH BX
    PUSH CX
    
    ; Training loop (simplified)
    MOV CX, 1000    ; Number of training iterations
    
TRAIN_LOOP:
    ; Forward pass for each training example
    ; (Simplified training - actual backpropagation would be more complex)
    
    ; Process training example 1
    MOV AL, input1_1
    MOV BL, input1_2
    CALL FORWARD_PASS
    
    ; Process training example 2
    MOV AL, input2_1
    MOV BL, input2_2
    CALL FORWARD_PASS
    
    ; Process training example 3
    MOV AL, input3_1
    MOV BL, input3_2
    CALL FORWARD_PASS
    
    ; Process training example 4
    MOV AL, input4_1
    MOV BL, input4_2
    CALL FORWARD_PASS
    
    LOOP TRAIN_LOOP
    
    POP CX
    POP BX
    POP AX
    RET
TRAIN_NETWORK ENDP

; =========================================================================
; FORWARD_PASS - Performs forward propagation through the network
; Input: AL = first input, BL = second input
; Output: output variable contains the result
; =========================================================================
FORWARD_PASS PROC
    PUSH AX
    PUSH BX
    PUSH DX
    
    ; Calculate hidden layer 1
    MOV AH, 0
    MUL w1_1        ; AL * w1_1
    MOV temp, AX
    
    MOV AL, BL
    MOV AH, 0
    MUL w1_2        ; BL * w1_2
    ADD temp, AX
    
    ; Apply activation (simplified - using threshold)
    MOV AX, temp
    CMP AX, 1
    JGE HIDDEN1_ON
    MOV hidden1, 0
    JMP HIDDEN1_DONE
HIDDEN1_ON:
    MOV hidden1, 1
HIDDEN1_DONE:
    
    ; Calculate hidden layer 2
    MOV AL, [BP-2]  ; Restore original AL
    MOV AH, 0
    MUL w2_1
    MOV temp, AX
    
    MOV AL, BL
    MOV AH, 0
    MUL w2_2
    ADD temp, AX
    
    ; Apply activation
    MOV AX, temp
    CMP AX, 1
    JGE HIDDEN2_ON
    MOV hidden2, 0
    JMP HIDDEN2_DONE
HIDDEN2_ON:
    MOV hidden2, 1
HIDDEN2_DONE:
    
    ; Calculate output layer
    MOV AX, hidden1
    MUL w3_1
    MOV temp, AX
    
    MOV AX, hidden2
    MUL w3_2
    ADD temp, AX
    
    ; Apply activation
    MOV AX, temp
    CMP AX, 1
    JGE OUTPUT_ON
    MOV output, 0
    JMP OUTPUT_DONE
OUTPUT_ON:
    MOV output, 1
OUTPUT_DONE:
    
    POP DX
    POP BX
    POP AX
    RET
FORWARD_PASS ENDP

; =========================================================================
; TEST_NETWORK - Tests the trained network with XOR examples
; =========================================================================
TEST_NETWORK PROC
    PUSH AX
    PUSH BX
    
    ; Test case 1: 0 XOR 0 = 0
    MOV AL, 0
    MOV BL, 0
    CALL FORWARD_PASS
    CALL DISPLAY_RESULT
    
    ; Test case 2: 0 XOR 1 = 1
    MOV AL, 0
    MOV BL, 1
    CALL FORWARD_PASS
    CALL DISPLAY_RESULT
    
    ; Test case 3: 1 XOR 0 = 1
    MOV AL, 1
    MOV BL, 0
    CALL FORWARD_PASS
    CALL DISPLAY_RESULT
    
    ; Test case 4: 1 XOR 1 = 0
    MOV AL, 1
    MOV BL, 1
    CALL FORWARD_PASS
    CALL DISPLAY_RESULT
    
    POP BX
    POP AX
    RET
TEST_NETWORK ENDP

; =========================================================================
; DISPLAY_RESULT - Displays the output of the network
; =========================================================================
DISPLAY_RESULT PROC
    PUSH AX
    PUSH DX
    
    ; Display result message
    LEA DX, msg_result
    MOV AH, 09h
    INT 21h
    
    ; Display the output value
    MOV AX, output
    ADD AL, '0'     ; Convert to ASCII
    MOV DL, AL
    MOV AH, 02h
    INT 21h
    
    ; Print newlineXORNets is stored in a single assembler file called WDR1_KERNEL.ASM
    MOV DL, 0Dh
    MOV AH, 02h
    INT 21h
    MOV DL, 0Ah
    MOV AH, 02h
    INT 21h
    
    POP DX
    POP AX
    RET
DISPLAY_RESULT ENDP

; =========================================================================
; XOR_OPERATION - Performs XOR operation on two bits
; Input: AL = bit 1, BL = bit 2
; Output: AL = result
; =========================================================================
XOR_OPERATION PROC
    PUSH BX
    
    ; Perform XOR using assembly XOR instruction
    XOR AL, BL
    AND AL, 1       ; Ensure single bit result
    
    POP BX
    RET
XOR_OPERATION ENDP

END MAIN
