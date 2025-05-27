#include "grafo.h"
#include "train.h"
#include <fstream>

int main(){

    ifstream F;
    F.open("data.txt");

    graph G(2);
    G.load(F);
    F.close();

    G.printRelations("é");
    ofstream V; 
    V.open("vecs.dat");

    F.open("data.txt");
    trainBOW(&G,F,0.05,1000);
    G.writeVecs(V);

    F.close();

    return 0;
}