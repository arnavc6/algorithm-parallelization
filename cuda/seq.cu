#include <vector>
#include <math.h>
#include <chrono>

std::pair<std::vector<size_t>, uint64_t> collatz_seq(size_t n){
    std::vector<size_t> cmap(n);
    uint64_t start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for(size_t i = 1; i <= n; i++){
        size_t j = i;
        int steps = 0;
        while(j != 1){
            steps++;
            if(j % 2 == 0){
                j /= 2;
            } else{
                j *= 3;
                j++;
            }
        }
        cmap[i - 1] = steps;
    }
    uint64_t end_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    return std::pair<std::vector<size_t>, uint64_t>{cmap, end_time - start_time};
}

std::pair<std::vector<std::vector<int>>, uint64_t> two_mm_seq(int alpha, int beta, std::vector<std::vector<int>> a, std::vector<std::vector<int>> b, std::vector<std::vector<int>> c, std::vector<std::vector<int>> d){
    int s = a.size();
    std::vector<std::vector<int>> bc(s, std::vector<int>(s, 0));
    std::vector<std::vector<int>> abc(s, std::vector<int>(s, 0));
    uint64_t start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for(int i = 0; i < s; i++){
        for(int j = 0; j < s; j++){
            a[i][j] *= alpha;
        }
    }
    for(int i = 0; i < s; i++){
        for(int j = 0; j < s; j++){
            for(int k = 0; k < s; k++){
                bc[i][j] += b[i][k] * c[k][j];
            }
        }
    }
    for(int i = 0; i < s; i++){
        for(int j = 0; j < s; j++){
            for(int k = 0; k < s; k++){
                abc[i][j] += a[i][k] * bc[k][j];
            }
        }
    }
    for(int i = 0; i < s; i++){
        for(int j = 0; j < s; j++){
            d[i][j] *= beta;
        }
    }
    for(int i = 0; i < s; i++){
        for(int j = 0; j < s; j++){
            abc[i][j] += d[i][j];
        }
    }
    uint64_t end_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    return std::pair<std::vector<std::vector<int>>, uint64_t>{abc, end_time - start_time};
}

std::pair<std::vector<std::vector<int>>, uint64_t> ising_seq(std::vector<std::vector<int>> lattice, std::vector<std::pair<int, int>> order){
    int s = lattice.size();
    uint64_t start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for(int i = 0; i < order.size(); i++){
        int r = order[i].first;
        int c = order[i].second;
        int cost = 2 * lattice[r][c] * (lattice[r][((c-1)%s+s)%s] + lattice[r][((c+1)%s+s)%s] + lattice[((r-1)%s+s)%s][c] + lattice[((r+1)%s+s)%s][c]);
        if (cost < 0){
            lattice[r][c] *= -1;
        }
    }
    uint64_t end_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    return std::pair<std::vector<std::vector<int>>, uint64_t>{lattice, end_time - start_time};
}