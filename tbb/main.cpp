#include <iostream>
#include "seq.cpp"
#include "par_tbb.cpp"

std::vector<std::vector<std::vector<int>>> generate_matrices(int size){
    std::vector<std::vector<std::vector<int>>> matrices;
    for(int i = 0; i < 4; i++){
        std::vector<std::vector<int>> matrix;
        for(int j = 0; j < size; j++){
            std::vector<int> row;
            for(int k = 0; k < size; k++){
                row.push_back(rand() % 64);
            }
            matrix.push_back(row);
        }
        matrices.push_back(matrix);
    }
    return matrices;
}

std::pair<std::vector<std::vector<int>>, std::vector<std::pair<int, int>>> generate_lattice_order(int size){
    std::vector<std::vector<int>> lattice;
    for(int i = 0; i < size; i++){
        std::vector<int> v;
        for(int j = 0; j < size; j++){
            if(rand() % 2){
                v.push_back(1);
            } else{
                v.push_back(-1);
            }
        }
        lattice.push_back(v);
    }
    std::vector<std::pair<int, int>> order;
    for(int i = 0; i < size * size / 4; i++){
        order.push_back(std::pair<int, int>{rand() % size, rand() % size});
    }
    return std::pair<std::vector<std::vector<int>>, std::vector<std::pair<int, int>>>{lattice, order};
}

int main(){
    std::vector<std::vector<std::vector<int>>> m = generate_matrices(8);
    int alpha = rand() % 64;
    int beta = rand() % 64;
    std::pair<std::vector<std::vector<int>>, std::vector<std::pair<int, int>>> l = generate_lattice_order(4);
    //1073741824
    //67108864
    std::pair<std::vector<size_t>, uint64_t> collatzseq = collatz_seq(32);
    // for(int i = 1; i <= 4; i++){
    //     std::cout << collatzseq.first[i] << " ";
    // }
    std::cout << std::endl << "SEQUENTIAL\ncollatz time = " << collatzseq.second << std::endl;

    std::pair<std::vector<std::vector<int>>, uint64_t> twommseq = two_mm_seq(alpha, beta, m[0], m[1], m[2], m[3]);
    std::cout << "2mm time = " << twommseq.second << std::endl;

    std::pair<std::vector<std::vector<int>>, uint64_t> isingseq = ising_seq(l.first, l.second);
    std::cout << "ising time = " << isingseq.second << std::endl << std::endl;
    //67108864
    std::pair<std::vector<size_t>, uint64_t> collatztbb = collatz_tbb(32);
    for(int i = 0; i < 32; i++){
        if(collatzseq.first[i] != collatztbb.first[i]){
            std::cout << "WRONG COLLATZ\n";
            std::cout << i + 1 << ": " << collatzseq.first[i] << " vs " << collatztbb.first[i] << std::endl;
            break;
        }
    }
    std::cout << std::endl << "TBB\ncollatz time = " << collatztbb.second << std::endl;

    std::pair<std::vector<std::vector<int>>, uint64_t> twommtbb = two_mm_tbb(alpha, beta, m[0], m[1], m[2], m[3]);
    std::cout << "2mm time = " << twommtbb.second << std::endl;
    if(twommseq.first != twommtbb.first){
        std::cout << "WRONG 2MM\n";
    }

    std::pair<std::vector<std::vector<int>>, uint64_t> isingtbb = ising_tbb(l.first, l.second);
    std::cout << "ising time = " << isingtbb.second << std::endl << std::endl;
    return 0;
}