extern float pregatesteVectori(float factorScalare, int elemente);
extern int ruleazaKerneluriGpu(int elemente, int laturaMatrice, float factor);

int ruleazaPipelineComplex(int elemente, int laturaMatrice) {
    float factorNormalizare = pregatesteVectori(0.5f, elemente);
    return ruleazaKerneluriGpu(elemente, laturaMatrice, factorNormalizare);
}
