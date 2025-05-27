#include <vector>
#include <string>

#ifndef NODES_H
#define NODES_H

// Usando using namespace std; conforme o original, embora seja preferível qualificar no header.
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
    // Função de normalização (corrigida para L2, mas não usada ativamente no treino principal)
    void normalize();
};

#endif