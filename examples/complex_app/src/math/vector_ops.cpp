float calculeazaBiasVector(int elemente) {
    return elemente > 1024 ? 2.0f : 1.0f;
}

float pregatesteVectori(float factorScalare, int elemente) {
    float bias = calculeazaBiasVector(elemente);
    return factorScalare * bias;
}
