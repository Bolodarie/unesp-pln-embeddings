#include "nodes.h"

node::node(string word, int dim){
    /*
        Inicializa nó.
    */

    this->dim = dim;
    this->vec = new double[dim];
    for (int s=0; s<dim; s++)
        this->vec[s] = 0.01;
    this->word = word;
}

node::~node(){
    delete[] this->vec;
}

node* node::operator[](int k){
    if (k<this->neighbors.size())
        return this->neighbors[k];
    else
        return nullptr;
}

bool node::operator==(string S){
    if (S.size()!=this->word.size())
        return false;
    for (int k=0; k<this->word.size(); k++){
        if (S[k]!=this->word[k])
            return false;
    }
    return true;
}

void node::connect(node* n){
    for (int k=0; k<this->neighbors.size(); k++){
        if (this->neighbors[k]==n)
            return;
    }
    this->neighbors.push_back(n);
}

void node::normalize(){
    double z = 0;
    for (int s=0; s<this->dim; s++){
        z += this->vec[s]*this->vec[s];
    }
    for (int s=0; s<this->dim; s++){
        this->vec[s] /= z;
    }
}