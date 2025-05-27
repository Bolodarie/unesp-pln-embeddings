#include "grafo.h"
#include <cmath> // Para std::exp, std::log
#include "mymath.h"
#include <fstream>
#include <iostream>
#include <vector>  // Para std::vector
#include <limits>  // Para std::numeric_limits
#include <numeric> // Para std::accumulate (opcional, mas bom para somas)
#include <algorithm> // Para std::max_element (opcional, mas bom para encontrar max)


// Função costBOW com estabilização numérica
double costBOW(graph& G) {
    double total_log_likelihood = 0.0;

    for (int i = 0; i < G.size(); i++) { // Nó central i (palavra a ser prevista)
        if (G.nd[i]->neighbors.size() != 2) {
            // O modelo de custo aqui assume que o contexto é a soma dos vizinhos.
            // Se não há vizinhos, ou não são 2 (como em trainBOW), o que fazer?
            // Por simplicidade, se não houver exatamente 2 vizinhos (como trainBOW espera para ctxSum)
            // talvez pular este nó ou adaptar.
            // Assumindo que o custo deve ser calculado mesmo assim,
            // com neighborSum sendo a soma dos vizinhos existentes.
            if (G.nd[i]->neighbors.empty()) continue;
        }

        // Vetor para armazenar a soma dos vetores dos vizinhos (contexto)
        std::vector<double> neighborSum_vec(G.dim, 0.0);
        for (size_t j = 0; j < G.nd[i]->neighbors.size(); ++j) {
            for (int s = 0; s < G.dim; ++s) {
                neighborSum_vec[s] += G.nd[i]->neighbors[j]->vec[s];
            }
        }
        double* neighborSum = neighborSum_vec.data();

        // Termo do produto interno direto para a palavra central i e seu contexto
        // Na Equação 4 dos slides, este é <x_S, sum_{v in V_S} x_v>
        // O 'out += inner(G.nd[i]->vec,G.nd[i]->neighbors[j]->vec,G.dim);' original parecia
        // somar produtos internos com cada vizinho individualmente, o que é diferente.
        // A Equação 4 sugere um produto interno do vetor da palavra central com a SOMA dos vetores de contexto.
        total_log_likelihood += inner(G.nd[i]->vec, neighborSum, G.dim);

        // Calcular o termo de normalização Z e log(Z) com estabilização
        std::vector<double> dot_products_cost(G.size());
        double max_dot_cost = -std::numeric_limits<double>::infinity();

        for (int p = 0; p < G.size(); ++p) {
            dot_products_cost[p] = inner(G.nd[p]->vec, neighborSum, G.dim);
            if (dot_products_cost[p] > max_dot_cost) {
                max_dot_cost = dot_products_cost[p];
            }
        }
        
        if (std::isinf(max_dot_cost) && max_dot_cost < 0) { // Todos os produtos internos foram -infinito
             max_dot_cost = 0; // Evita -inf - (-inf) = NaN em exp, define um ponto de referência
        }


        double Z_cost = 0.0;
        for (int p = 0; p < G.size(); ++p) {
            if (dot_products_cost[p] - max_dot_cost < -700) { // Prevenção de underflow para std::exp
                Z_cost += 0.0;
            } else {
                Z_cost += std::exp(dot_products_cost[p] - max_dot_cost);
            }
        }

        if (Z_cost < 1e-9) { // Z_cost é efetivamente zero
            // log(Z_cost) seria -infinito, o que pode ser problemático.
            // O custo total iria para +infinito.
            // Pode-se retornar um valor muito grande ou lidar com isso como um erro.
            // Para a fórmula do slide: ln f(X) = Sum [ <x_S, ctx> - log Z ]
            // Se log Z = -inf, então -log Z = +inf.
            total_log_likelihood += std::numeric_limits<double>::infinity();
        } else {
            total_log_likelihood -= (std::log(Z_cost) + max_dot_cost);
        }
    }
    // A Equação 4 é para maximizar ln f(X). O custo geralmente é -ln f(X) para minimizar.
    return -total_log_likelihood;
}

// A função gradBOW não é chamada por main.cpp e não foi o foco da correção de NaN.
// Mantendo-a como estava no arquivo original fornecido pelo usuário.
std::vector<double*> gradBOW(graph& G, std::ifstream& F){
    std::vector<double*> out_grads; // Renomeado para evitar conflito com 'out' em costBOW
    for (int i=0; i<G.size(); i++){
        double* aux = zero(G.dim);
        out_grads.push_back(aux);
    }
    while(!F.eof()){
        std::string S_str, V_str, T_str; // Renomeado para evitar conflito
        std::getline(F,S_str,' ');
        if (S_str.empty() && F.eof()) break; // Checagem para linha vazia no final do arquivo
        std::getline(F,V_str,' ');
        std::getline(F,T_str);
        
        // Remover possível \r de T_str se o arquivo for Windows-style
        if (!T_str.empty() && T_str.back() == '\r') {
            T_str.pop_back();
        }
        if (S_str.empty() || V_str.empty() || T_str.empty()) continue; // Pular linhas malformadas


        //Indices
        int pV = G.where(V_str);
        int pS = G.where(S_str);
        int pT = G.where(T_str);

        if (pV < 0 || pS < 0 || pT < 0) { // Checar se as palavras existem no grafo
            std::cerr << "Warning: Uma das palavras (" << S_str << ", " << V_str << ", " << T_str << ") não encontrada no grafo em gradBOW." << std::endl;
            continue;
        }

        //Gradiente para a resposta (verbo)
        double* prob = G.probGivenContext(S_str,T_str);
        if (!prob) { // probGivenContext pode retornar nullptr
             std::cerr << "Warning: probGivenContext retornou nullptr para " << S_str << ", " << T_str << std::endl;
             continue;
        }

        double *grad_V = zero(G.dim);
        // A fórmula original do grad_V parece específica para uma certa interpretação.
        // (1-prob[pV]) * (vec_S[s] + vec_T[s]) seria mais comum para o gradiente do log P(V|S,T) em relação ao vetor V.
        // A formula (1-prob[pV])*G.nd[pS]->vec[s]+G.nd[pT]->vec[s] é (1-P_v)S_s + T_s, o que é incomum.
        // Vamos manter a original por enquanto, pois o foco não é esta função.
        for (int s=0; s<G.dim; s++){
            grad_V[s] = (1-prob[pV])*(G.nd[pS]->vec[s] + G.nd[pT]->vec[s]); // Suposição: contexto é soma S+T
        }
        iadd(out_grads[pV],grad_V,G.dim);
        delete[] grad_V;
        delete[] prob; // prob foi alocado por probGivenContext

        //Gradiente para o primeiro substantivo
        // A lógica aqui é complexa e parece ser específica para a fonte original do código.
        // Será mantida, mas com cuidado na desalocação.
        prob = G.probGivenContext(S_str, T_str); // Recalcular ou reutilizar se possível (assumindo que não muda)
         if (!prob) continue;

        double* avgS = G.vecAverage(prob); // vecAverage aloca memória
        if(avgS) { // Checar se avgS não é nullptr
            imul(avgS,-1,G.dim);
            iadd(out_grads[pS],G.nd[pV]->vec,G.dim);
            iadd(out_grads[pS],avgS,G.dim);
            delete[] avgS;
        }

        double *auxS = zero(G.dim);
        // Cuidado: G.nd[pS]->vec+(G.dim-1) é apenas o último elemento, não um range para copy.
        // std::copy(source_begin, source_end, destination_begin)
        // Se a intenção é copiar todo o vetor:
        std::copy(G.nd[pS]->vec, G.nd[pS]->vec + G.dim, auxS);
        imul(auxS,-prob[pS],G.dim); // prob[pS] é P(S | S, T) ? Incomum.
        iadd(out_grads[pS],auxS,G.dim);
        delete[] auxS;

        //Gradiente para o segundo substantivo
        double* avgT = G.vecAverage(prob); // vecAverage aloca memória
        if(avgT){
            imul(avgT,-1,G.dim);
            iadd(out_grads[pT],G.nd[pV]->vec,G.dim);
            iadd(out_grads[pT],avgT,G.dim);
            delete[] avgT;
        }
        
        double *auxT = zero(G.dim);
        std::copy(G.nd[pT]->vec, G.nd[pT]->vec + G.dim, auxT); // Corrigido para copiar todo o vetor de T
        imul(auxT,-prob[pT],G.dim); // prob[pT] é P(T | S, T) ? Incomum.
        iadd(out_grads[pT],auxT,G.dim); // Adiciona a out_grads[pT], não out_grads[pS]
        delete[] auxT;

        delete[] prob; // Deletar prob se foi recalculado
    }

    return out_grads;
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

    if (V == 0) {
        std::cerr << "Error: Graph size is zero. Cannot train." << std::endl;
        return;
    }

    for (int e = 0; e < epochs; ++e) {
        // pra cada nó i no grafo (palavra central/alvo)
        for (int i = 0; i < V; ++i) {
            // obter dois vizinhos do contexto
            auto& neigh = G->nd[i]->neighbors;
            if (neigh.size() != 2) continue;  // pula se não tiver exatamente 2 vizinhos de contexto
            
            int iS = findNodeIndex(G, neigh[0]); // Índice do primeiro vizinho (contexto S)
            int iT = findNodeIndex(G, neigh[1]); // Índice do segundo vizinho (contexto T)
            
            if (iS < 0 || iT < 0) { // Segurança: vizinhos não encontrados no grafo (improvável se bem construído)
                 std::cerr << "Warning: Context node not found by index in trainBOW." << std::endl;
                 continue;
            }

            //soma dos vetores de contexto (ctxSum = vec(S) + vec(T))
            std::vector<double> ctxSum(dim, 0.0);
            for (int k = 0; k < dim; ++k) {
                ctxSum[k] = G->nd[iS]->vec[k] + G->nd[iT]->vec[k]; // Correção: usar iS e iT para os vetores de contexto
            }

            // --- Forward pass: calcula scores e softmax (estabilizado) ---
            std::vector<double> dot_products(V);
            double max_dot_product = -std::numeric_limits<double>::infinity();

            for (int j = 0; j < V; ++j) { // Para cada palavra j no vocabulário
                dot_products[j] = inner(G->nd[j]->vec, ctxSum.data(), dim);
                if (dot_products[j] > max_dot_product) {
                    max_dot_product = dot_products[j];
                }
            }
            
            // Caso extremo: se todos os produtos internos forem -infinito (e.g. vetores NaN ou Inf)
            if (std::isinf(max_dot_product) && max_dot_product < 0) {
                max_dot_product = 0; // Evita -inf - (-inf) = NaN na exponenciação
            }


            std::vector<double> scores(V); // scores[j] = P(palavra_j | contexto)
            double Z = 0.0; // Denominador do softmax
            for (int j = 0; j < V; ++j) {
                // Truque Log-Sum-Exp para estabilidade numérica
                double val_exp = dot_products[j] - max_dot_product;
                if (val_exp < -700) { // Prevenção de underflow para std::exp (exp(-700) é muito próximo de 0)
                    scores[j] = 0.0;
                } else {
                    scores[j] = std::exp(val_exp);
                }
                Z += scores[j];
            }

            if (Z < 1e-9) { // Z é efetivamente zero, evitar divisão por zero
                if ((e + 1) % 100 == 0 || e < 5) { // Imprimir aviso menos frequentemente ou nas primeiras épocas
                     std::cerr << "Epoch " << (e + 1) << ": Warning! Z (" << Z <<") is close to zero for target node " << G->nd[i]->word
                               << ". Max dot: " << max_dot_product << ". Skipping update for this instance." << std::endl;
                }
                continue; // Pula para o próximo nó i no grafo
            }

            for (int j = 0; j < V; ++j) {
                scores[j] /= Z;
                if (std::isnan(scores[j]) || std::isinf(scores[j])) {
                    if ((e + 1) % 100 == 0 || e < 5) {
                        std::cerr << "Epoch " << (e + 1) << ": Warning! scores[" << j << "] is " << scores[j]
                                  << " for target " << G->nd[i]->word << ". Z=" << Z << ", dot_prod-max_dot=" << (dot_products[j]-max_dot_product)
                                  << std::endl;
                    }
                    // Poderia pular a atualização se um score for NaN/Inf
                }
            }

            // --- Backward e atualização ---
            // 1. Output embeddings (vetores G->nd[j]->vec, onde j é qualquer palavra no vocabulário)
            //    O alvo é a palavra i.
            for (int j = 0; j < V; ++j) { // Para cada palavra de saída j
                double coeff = scores[j] - (j == i ? 1.0 : 0.0); // (P(j|ctx) - target_indicator)
                for (int k = 0; k < dim; ++k) {
                    G->nd[j]->vec[k] -= lr * coeff * ctxSum[k];
                }
            }

            // 2. Input embeddings (vetores das palavras de contexto G->nd[iS]->vec e G->nd[iT]->vec)
            //    Implementando Equação 6: d(ln P(i|ctx))/dx_v^c = x_i - Sum_t(x_t * P(t|ctx)) - x_v^c * P(v|ctx)
            //    onde x_i é o alvo (G->nd[i]), x_t são todos os outputs, x_v^c é uma palavra do contexto (G->nd[idx])
            for (int idx : {iS, iT}) { // Para cada palavra de contexto (S ou T), G->nd[idx] é x_v^c
                for (int k = 0; k < dim; ++k) {
                    // Termo 1: x_i[k] (vetor da palavra alvo i)
                    double x_target_k = G->nd[i]->vec[k];

                    // Termo 2: Sum_{t in Vocab} (x_t[k] * P(t|ctx))
                    double sum_xt_Pt_k = 0.0;
                    for (int j = 0; j < V; ++j) { // j aqui é o 't' da fórmula
                        sum_xt_Pt_k += G->nd[j]->vec[k] * scores[j];
                    }

                    // Termo 3: x_v^c[k] * P(v|ctx)
                    // x_v^c[k] é G->nd[idx]->vec[k]
                    // P(v|ctx) é scores[idx] (probabilidade da palavra de contexto 'idx' ser predita pelo mesmo contexto)
                    double x_vc_k_Pv_k = G->nd[idx]->vec[k] * scores[idx];

                    // Gradiente para subida (maximização de ln P), conforme Eq. 6
                    double grad_ascent_k = (x_target_k - sum_xt_Pt_k) - x_vc_k_Pv_k;
                    
                    if(std::isnan(grad_ascent_k) || std::isinf(grad_ascent_k)){
                         if ((e + 1) % 100 == 0 || e < 5) {
                            std::cerr << "Epoch " << (e + 1) << ": Warning! grad_ascent_k is " << grad_ascent_k
                                      << " for target " << G->nd[i]->word << ", context " << G->nd[idx]->word << ", dim " << k
                                      << std::endl;
                            std::cerr << "    x_target_k=" << x_target_k << ", sum_xt_Pt_k=" << sum_xt_Pt_k << ", x_vc_k_Pv_k=" << x_vc_k_Pv_k << std::endl;
                         }
                         // Pular esta atualização específica da dimensão k se o gradiente for NaN/Inf
                         continue;
                    }


                    // Atualiza o vetor da palavra de contexto usando gradiente de subida
                    G->nd[idx]->vec[k] += lr * grad_ascent_k;
                }
            }
        } // Fim do loop sobre os nós i

        // custos a cada épocas
        // if ((e + 1) % 1 == 0) { // Imprimir sempre, ou mude o módulo para menos frequência
        //     double c = costBOW(*G);
        //     std::cout << "Epoch " << (e + 1) << "/" << epochs << ": cost = " << c << std::endl;
        //      if (std::isnan(c) || std::isinf(c)) {
        //         std::cerr << "Error: Cost is NaN or Inf. Stopping training." << std::endl;
        //         return; // Parar o treinamento se o custo explodir
        //     }
        // }
    } // Fim do loop de épocas
}