.section .text.start,"ax",@progbits
.global _start
_start:
    la sp, __stack_top
    li t0, 0x6000
    csrs mstatus, t0
    call main
1:
    j 1b
