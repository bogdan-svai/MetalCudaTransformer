void cudaLaunch(void(*kernel)(int), int blocks, int threads, int arg);

void ruleazaPeGPU(int x) {}

void codCpp() {
    cudaLaunch(ruleazaPeGPU, 1, 32, 5);
}
