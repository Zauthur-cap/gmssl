#include <stdio.h>

typedef unsigned long long gm_bn_t;

static void gm_i_bn_add_x(gm_bn_t *r, gm_bn_t *a, gm_bn_t *b, int count) {
    int i;
    
    for (i = 0; i < count; i++) {
        gm_bn_t carry = 0;
        gm_bn_t temp_a, temp_b;
        
        __asm__(
            "ldr %x[temp_a], [%[a]]\n"
            "add %x[a], %x[a], #8\n"
            "ldr %x[temp_b], [%[b]]\n"
            "add %x[b], %x[b], #8\n"
            "ldr %x[r], [%[r]]\n"
            "adds %x[r], %x[r], %x[temp_a]\n"
            "adc %x[r], %x[r], %x[temp_b]\n"
            : [a] "+r" (a), [b] "+r" (b), [r] "+r" (r[i]), [temp_a] "=&r" (temp_a), [temp_b] "=&r" (temp_b)
            :
            : "memory"
        );
    }
}

int main() {
    gm_bn_t a[] = {1234567890, 9876543210, 1357924680};
    gm_bn_t b[] = {9876543210, 1234567890, 2468135790};
    int count = sizeof(a) / sizeof(gm_bn_t);
    gm_bn_t r[count];

    gm_i_bn_add_x(r, &a[0], &b[0], count);

    // Print the result
    for (int i = 0; i < count; i++) {
        printf("%llu\n", r[i]);
    }

    return 0;
}
