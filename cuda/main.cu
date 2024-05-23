#include <iostream>
#include "seq.cu"
#include "par_cuda.cu"

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
    int s = 8;
    std::vector<std::vector<std::vector<int>>> m = generate_matrices(s);
    int alpha = rand() % 64;
    int beta = rand() % 64;
    std::pair<std::vector<std::vector<int>>, std::vector<std::pair<int, int>>> l = generate_lattice_order(4);
    
    std::pair<std::vector<size_t>, uint64_t> collatzseq = collatz_seq(32);
    std::cout << std::endl << "SEQUENTIAL\ncollatz time = " << collatzseq.second << std::endl;

    std::pair<std::vector<std::vector<int>>, uint64_t> twommseq = two_mm_seq(alpha, beta, m[0], m[1], m[2], m[3]);
    std::cout << "2mm time = " << twommseq.second << std::endl;

    std::pair<std::vector<std::vector<int>>, uint64_t> isingseq = ising_seq(l.first, l.second);
    std::cout << "ising time = " << isingseq.second << std::endl << std::endl;

    std::pair<std::vector<size_t>, uint64_t> collatzcuda = collatz_cuda(32);
    std::pair<std::vector<std::vector<int>>, uint64_t> twommcuda = two_mm_cuda(alpha, beta, m[0], m[1], m[2], m[3]);
    std::pair<std::vector<std::vector<int>>, uint64_t> isingcuda = ising_cuda(l.first, l.second);
    for(int i = 0; i < 32; i++){
        // std::cout << i << ": " << collatzseq.first[i] << " vs " << collatzcuda.first[i] << std::endl;
        if(collatzseq.first[i] != collatzcuda.first[i]){
            std::cout << "WRONG COLLATZ" << std::endl;
            std::cout << i << ": " << collatzseq.first[i] << " vs " << collatzcuda.first[i] << std::endl;
            break;
        }
    }
    bool br = false;
    for(int i = 0; i < s; i++){
        for(int j = 0; j < s; j++){
            if(twommcuda.first[i][j] != twommseq.first[i][j]){
                std::cout << "WRONG 2MM\n";
                std::cout << i << " " << j << " " << twommseq.first[i][j] << " " << twommcuda.first[i][j] << std::endl;
                br = true;
                break;
            }
        }
        if(br){
            break;
        }
    }

    std::cout << std::endl << "CUDA\ncollatz time = " << collatzcuda.second << std::endl;
    std::cout << "2mm time = " << twommcuda.second << std::endl;
    std::cout << "ising time = " << isingcuda.second << std::endl << std::endl;
}