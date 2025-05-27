#include "grafo.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include "mymath.h"

bool graph::isIn(string S){
    for (int i=0; i<this->size(); i++){
        if ((*this->nd[i])==S)
            return true;
    }
    return false;
}

void graph::append(string S){
    if (!this->isIn(S)){
        node* newNode = new node(S,this->dim);
        this->nd.push_back(newNode);
    }
}

int graph::where(string S){
    /*
        Retorna o indice da string k entre os nós.
    */
    for (int i=0; i<this->size(); i++){
        if ((*this->nd[i])==S)
            return i;
    }
    return -1;
}

void graph::load(ifstream& F){
    /*
        Carrega uma base de dados para um grafo.
    */
    while (!F.eof()){
        string S;
        getline(F,S,' ');
        string V;
        getline(F,V,' ');
        string T;
        getline(F,T);
        this->append(S);
        this->append(V);
        this->append(T);
        this->connect(V,S);
        this->connect(V,T);
    }
}

void graph::printNodes(){
    /*
        Imprime lista de nós na tela..
    */
    for (int i=0; i<this->size(); i++){
        cout << this->nd[i]->word << endl;
    }
}

void graph::connect(string S, string T){
    int k, p;
    for (int i=0; i<this->size(); i++){
        if ((*this->nd[i])==S)
            k = i;
        if ((*this->nd[i])==T)
            p = i;
    }
    this->nd[k]->connect(this->nd[p]);
}

void graph::printRelations(string S){
    int p;
    for (int i=0; i<this->size(); i++){
        if((*this->nd[i])==S)
            p = i;
    }
    cout << "Relações para " << S << ": \n";
    for (int k=0; k<this->nd[p]->neighbors.size(); k++){
        cout << this->nd[p]->neighbors[k]->word << endl;
    }
}

void graph::writeVecs(ofstream& F){
    /*
        Escreve representação vetorial num arquivo.
    */
    for (int i=0; i<this->size(); i++){
        //F << this->nd[i]->word << ": ";
        for (int s=0; s<this->dim; s++){
            F << this->nd[i]->vec[s] << " ";
        }
        F << endl;
    }
}

double* graph::probGivenContext(string S, string T){
    if (!this->isIn(S) || !this->isIn(T)){
        return nullptr;
    }
    double* prob = new double[this->size()]; //variável de saída
    double* aux = new double[this->dim]; //para armazenar soma dos contextos
    for (int s=0; s<this->dim; s++)
        aux[s] = 0;
    double z = 0; //constante de normalização
    //Calcula soma dos contextos
    for (int i=0; i<this->size(); i++){
        if ((*this->nd[i])==S || (*this->nd[i])==T){
            for (int s=0; s<this->dim; s++)
                aux[s] += this->nd[i]->vec[s];
        }
    }
    //Calcula fator exponencial e z
    for (int i=0; i<this->size(); i++){
        prob[i] = exp(inner(this->nd[i]->vec,aux,this->dim));
        z += prob[i];
    }
    //Normalização
    for (int i=0; i<this->size(); i++){
        prob[i] /= z;
    }
    delete[] aux;
    return prob;
}

double* graph::vecAverage(double* p){
    /*
        Média dos vetores ponderados por p.
    */
    double* out = new double[this->dim];
    for (int s=0; s<this->dim; s++){
        out[s] = 0;
    }
    for (int i=0; i<this->size(); i++){
        for (int s=0; s<this->dim; s++){
            out[s] += this->nd[i]->vec[s]*p[i];
        }
    }
    return out;
}