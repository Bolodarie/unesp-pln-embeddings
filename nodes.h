#include <vector>
#include <string>

#ifndef NODES_H
#define NODES_H

using namespace std;

class node{
public:
    string word; //palavra armazenada
    int dim; //dimensão do espaço de embarcação
    double* vec; //vetor no espaço de embarcação
    vector<node*> neighbors; //vizinhos

    node(string word, int dim);
    ~node();

    //Sobrecarga do operador [] para acessar vizinhos
    node* operator[](int k);
    //Operator == para checar palavras iguais
    bool operator==(string S);
    
    void connect(node* n);
    void normalize();
};

#endif