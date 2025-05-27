#include "mymath.h"

double inner(double* x, double* y, int len){
    /*
        Produto interno entre x e y.
    */
    double out = 0;
    for (int k=0; k<len; k++){
        out += x[k]*y[k];
    }
    return out;
}

double* zero(int len){
    double* out = new double[len];
    for (int s=0; s<len; s++)
        out[s] = 0;
    return out;
}

void iadd(double* x, double* y, int len){
    /*
        Adição inplace sobre x.
    */
    for (int k=0; k<len; k++)
        x[k] += y[k];
}

void imul(double*x, double p, int len){
    /*
        Adição inplace sobre x.
    */
    for (int k=0; k<len; k++)
        x[k] *= p;
}