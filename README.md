# unesp-pln-embeddings
Implementação em C++ do modelo CBOW para aprendizado de representações vetoriais de palavras a partir de uma estrutura de grafo relacional. Inclui visualização dos vetores 2D gerados.

# GraphCBOW-Embeddings

Implementação em C++ do modelo CBOW (Continuous Bag-of-Words) para aprendizado de representações vetoriais de palavras (word embeddings) a partir de uma estrutura de grafo relacional. Este projeto foi desenvolvido com base nos conceitos apresentados nas aulas de Processamento de Linguagem Natural[cite: 1], explorando como as relações semânticas extraídas de um corpus e representadas em um grafo podem ser usadas para gerar embeddings significativos.

## Visão Geral

O sistema funciona da seguinte maneira:
1.  **Carregamento de Dados e Construção do Grafo:** Um corpus textual (fornecido em `data.txt`) contendo relações entre palavras (tipicamente no formato Sujeito-Verbo-Objeto) é lido. As palavras são representadas como nós e suas interconexões como arestas em um grafo. Especificamente, a palavra central da relação (ex: verbo) é conectada às suas palavras adjacentes.
2.  **Treinamento do Modelo CBOW:** Um modelo Continuous Bag-of-Words é treinado sobre a estrutura do grafo. Para cada nó central no grafo, o modelo tenta prever este nó a partir da soma vetorial de seus nós de contexto (vizinhos diretos). O projeto considera contextos de exatamente duas palavras vizinhas. Os vetores de palavras (embeddings) são ajustados iterativamente para minimizar uma função de perda baseada na predição correta da palavra central. As fórmulas de gradiente para as variáveis de resposta e de contexto são inspiradas nas discussões em aula.
3.  **Geração e Visualização de Embeddings:** Após o treinamento, os vetores de palavras aprendidos são salvos em um arquivo (`vecs.dat`). Um script Python (`plotVec.py`) é fornecido para visualizar esses embeddings em um espaço 2D, permitindo uma análise qualitativa das relações aprendidas.

## Estrutura do Código

* **`main.cpp`**: Ponto de entrada do programa. Orquestra o carregamento dos dados, treinamento e salvamento dos vetores.
* **`grafo.h`, `grafo.cpp`**: Define e implementa a estrutura do grafo e suas operações (carregar dados, adicionar nós/arestas, etc.).
* **`nodes.h`, `nodes.cpp`**: Define e implementa a estrutura do nó (palavra), que armazena o vetor de embedding e seus vizinhos.
* **`train.h`, `train.cpp`**: Contém a lógica de treinamento do modelo CBOW, incluindo a função de custo e o cálculo/aplicação dos gradientes.
* **`mymath.h`, `mymath.cpp`**: Funções matemáticas auxiliares para operações vetoriais.
* **`data.txt`**: Arquivo de exemplo com o corpus textual.
* **`vecs.dat`**: Arquivo de saída onde os vetores de palavras são armazenados (gerado após a execução).
* **`plotVec.py`**: Script Python para visualização 2D dos vetores em `vecs.dat`.

## Como Compilar e Executar

### Pré-requisitos
* Um compilador C++ (compatível com C++11 ou superior, ex: g++)
* Python 3.x
* Bibliotecas Python: `numpy`, `matplotlib` (para `plotVec.py`)

### Compilação
Recomenda-se o uso de um Makefile para facilitar a compilação. Um exemplo simples de comando de compilação com g++ seria:
```bash
g++ -std=c++11 -g -O0     main.cpp     train.cpp     grafo.cpp     mymath.cpp     nodes.cpp   -o main
./main
