#include "grafo.h"
#include <cmath>
#include "mymath.h"
#include <fstream>
#include <iostream>

double costBOW(graph& G){
    double out = 0;
    //Itera sobre os nós
    for (int i=0; i<G.size(); i++){
        //Vetor para armazenar a soma dos vizinhos
        double* neighborSum = zero(G.dim);
        //Itera sobre vizinhos
        for (int j=0; j<G.nd[i]->neighbors.size(); j++){
            out += inner(G.nd[i]->vec,G.nd[i]->neighbors[j]->vec,G.dim);
            for (int s=0; s<G.dim; s++){
                neighborSum[s] += G.nd[i]->neighbors[j]->vec[s];
            }
        }
        //Itera sobre todos os nós para calcular fator de normalização
        double Z = 0;
        for (int p=0; p<G.nd.size(); p++){
            Z += exp(inner(G.nd[p]->vec,neighborSum,G.dim));
        }
        delete[] neighborSum;
        out -= log(Z);
    }
    return -out;
}

vector<double*> gradBOW(graph& G, ifstream& F){
    vector<double*> out;
    for (int i=0; i<G.size(); i++){
        double* aux = zero(G.dim);
        out.push_back(aux);
    }
    while(!F.eof()){
        string S;
        getline(F,S,' ');
        string V;
        getline(F,V,' ');
        string T;
        getline(F,T);

        //Indices
        int pV = G.where(V);
        int pS = G.where(S);
        int pT = G.where(T);

        //Gradiente para a resposta (verbo)
        double* prob = G.probGivenContext(S,T);
        double *grad_V = zero(G.dim);
        for (int s=0; s<G.dim; s++){
            grad_V[s] = (1-prob[pV])*G.nd[pS]->vec[s]+G.nd[pT]->vec[s];
        }
        iadd(out[pV],grad_V,G.dim);
        delete[] grad_V;

        //Gradiente para o primeiro substantivo
        double* avgS = G.vecAverage(prob);
        imul(avgS,-1,G.dim);
        iadd(out[pS],G.nd[pV]->vec,G.dim);
        iadd(out[pS],avgS,G.dim);
        delete[] avgS;
        double *auxS = zero(G.dim);
        copy(G.nd[pS]->vec,G.nd[pS]->vec+(G.dim-1),auxS);
        imul(auxS,-prob[pS],G.dim);
        iadd(out[pS],auxS,G.dim);
        delete[] auxS;

        //Gradiente para o segundo substantivo
        double* avgT = G.vecAverage(prob);
        imul(avgT,-1,G.dim);
        iadd(out[pT],G.nd[pV]->vec,G.dim);
        iadd(out[pT],avgT,G.dim);
        delete[] avgT;
        double *auxT = zero(G.dim);
        copy(G.nd[pS]->vec,G.nd[pS]->vec+(G.dim-1),auxT);
        imul(auxT,-prob[pS],G.dim);
        iadd(out[pS],auxT,G.dim);
        delete[] auxT;
    }

    return out;
}

static int findNodeIndex(graph* G, node* n) {
    for (int idx = 0; idx < G->size(); ++idx) {
        if (G->nd[idx] == n) return idx;
    }
    return -1;
}

void trainBOW(graph* G, std::ifstream& /*in*/, double lr, int epochs) {
    const int dim = G->dim;
    const int V = G->size();

    for (int e = 0; e < epochs; ++e) {
        // pra cada nó i no grafo
        for (int i = 0; i < V; ++i) {
            // obter dois vizinhos do contexto
            auto& neigh = G->nd[i]->neighbors;
            if (neigh.size() != 2) continue;  // pula se não tiver exatamente 2
            int iS = findNodeIndex(G, neigh[0]);
            int iT = findNodeIndex(G, neigh[1]);
            if (iS < 0 || iT < 0) continue;    // segurança

            //soma dos vetores de contexto
            std::vector<double> ctxSum(dim, 0.0);
            for (int k = 0; k < dim; ++k) {
                ctxSum[k] = neigh[0]->vec[k] + neigh[1]->vec[k];
            }

            //forward: calcula scores e softmax
            std::vector<double> scores(V);
            double Z = 0.0;
            for (int j = 0; j < V; ++j) {
                double dot = inner(G->nd[j]->vec, ctxSum.data(), dim);
                scores[j] = std::exp(dot);
                Z += scores[j];
            }
            for (int j = 0; j < V; ++j) scores[j] /= Z;

            //backward e atualização
            //output embeddings (target == i)
            for (int j = 0; j < V; ++j) {
                double coeff = scores[j] - (j == i ? 1.0 : 0.0);
                for (int k = 0; k < dim; ++k) {
                    G->nd[j]->vec[k] -= lr * coeff * ctxSum[k];
                }
            }

            //input embeddings (dois vizinhos)
            for (int idx : {iS, iT}) {
                for (int k = 0; k < dim; ++k) {
                    double grad_in = (1.0 - scores[i]) * G->nd[i]->vec[k];
                    for (int j = 0; j < V; ++j) {
                        if (j != i) grad_in -= scores[j] * G->nd[j]->vec[k];
                    }
                    G->nd[idx]->vec[k] -= lr * grad_in;
                }
            }
        }

        //custos a cada épocas
        if ((e + 1) % 1 == 0) {
            double c = costBOW(*G);
            std::cout << "Epoch " << (e + 1) << ": cost = " << c << std::endl;
        }
    }
}


