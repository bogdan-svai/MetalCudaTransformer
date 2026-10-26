// Original synthetic corpus for host-loop analysis. No GPU transformations yet.
void map(unsigned* out, const unsigned* in, unsigned n) {
    // interval UTF-8: șir
    for (unsigned i = 0; i < n; ++i) out[i] = in[i] * 3u + 1u;
}
void combine(unsigned* out, const unsigned* a, const unsigned* b, unsigned n) {
    for (unsigned i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
void overlap(unsigned* a, unsigned n) {
    unsigned* out = a + 1;
    for (unsigned i = 0; i < n; ++i) out[i] = a[i];
}
void prefix(unsigned* a, unsigned n) {
    for (unsigned i = 1; i < n; ++i) a[i] += a[i - 1];
}
extern void observe(unsigned);
void effects(unsigned n) {
    for (unsigned i = 0; i < n; ++i) observe(i);
}
void wrap(unsigned* out, unsigned limit) {
    for (unsigned i = 0; i <= limit; ++i) out[0] = i;
}
void empty(unsigned* out) {
    for (unsigned i = 0; i < 0; ++i) out[i] = 1u;
}
#define LOOP(n) for (unsigned i = 0; i < n; ++i)
void macro_loop(unsigned* out, unsigned n) {
    LOOP(n) out[i] = i;
}
