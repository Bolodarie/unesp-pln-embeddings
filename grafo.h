#include "nodes.h"

#ifndef GRAFO_H
#define GRAFO_H

class graph{
public:
    int dim;
    vector<node*> nd;

    graph(int dim){
        this->dim = dim;
    }

    int size(){
        return this->nd.size();
    }

    bool isIn(string S); //Checa se string S já foi armazenada em algum nó
    void append(string S); //Adiciona S no final de nd
    int where(string S);


    void load(ifstream& F); //Carrega base de dados
    void printNodes(); // Imprime nós
    void printRelations(string S);
    void connect(string S, string T); //Conecta duas strings

    void writeVecs(ofstream& F);

    double* probGivenContext(string S, string T);
    double* vecAverage(double* p);
};

#endif