#include "nodes.h"
#include <string>
#include <vector>
#include <random>  // Para std::mt19937, std::uniform_real_distribution
#include <ctime>   // Para std::time
#include <cmath>   // Para std::sqrt

// Helper para inicializar o gerador de números aleatórios uma vez.
static std::mt19937& get_random_engine() {
    // Semeia o gerador com o tempo atual para obter sequências diferentes a cada execução.
    // static garante que seja inicializado apenas uma vez.
    static std::mt19937 random_engine(static_cast<unsigned int>(std::time(nullptr)));
    return random_engine;
}

node::node(std::string word_val, int dim_val) { // Parâmetros renomeados para evitar sombreamento
    this->dim = dim_val;
    this->vec = new double[dim_val];
    this->word = word_val; // Atribui a palavra

    // Inicialização com pequenos valores aleatórios para quebrar a simetria
    // Distribuição uniforme, por exemplo, entre -0.05 e 0.05 (ou outro range pequeno)
    std::uniform_real_distribution<double> distribution(-0.05, 0.05);

    for (int s = 0; s < this->dim; s++) {
        this->vec[s] = distribution(get_random_engine());
    }
}

node::~node() {
    delete[] this->vec;
}

node* node::operator[](int k) {
    // Adicionada checagem para k >= 0
    if (k >= 0 && static_cast<size_t>(k) < this->neighbors.size())
        return this->neighbors[k];
    else
        return nullptr;
}

bool node::operator==(std::string S) {
    return this->word == S; // Comparação direta de strings
}

void node::connect(node* n) {
    if (n == nullptr || n == this) return; // Não conectar a si mesmo ou a nullptr

    for (size_t k = 0; k < this->neighbors.size(); k++) {
        if (this->neighbors[k] == n)
            return; // Já conectado
    }
    this->neighbors.push_back(n);
}

// Função de normalização L2 (corrigida). Não é chamada no fluxo de treino principal.
void node::normalize() {
    double norm_sq = 0;
    for (int s = 0; s < this->dim; s++) {
        norm_sq += this->vec[s] * this->vec[s];
    }

    if (norm_sq > 1e-12) { // Evitar divisão por zero ou por número muito pequeno
        double norm_val = std::sqrt(norm_sq);
        for (int s = 0; s < this->dim; s++) {
            this->vec[s] /= norm_val;
        }
    }
}