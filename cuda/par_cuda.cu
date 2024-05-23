#include <vector>
#include <math.h>
#include <chrono>

__global__ void collatz_kernel(size_t n, size_t* vals){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index; i < n; i += stride){
        size_t j = i + 1;
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
        vals[i] = steps;
    }
}

std::pair<std::vector<size_t>, uint64_t> collatz_cuda(size_t n){
    std::vector<size_t> cmap(n);
    size_t* vals;
    cudaMallocManaged(&vals, n*sizeof(size_t));
    for(int i = 0; i < n; i++){
        vals[i] = 0;
    }
    uint64_t start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    collatz_kernel<<<1024, 1024>>>(n, vals);
    cudaDeviceSynchronize();
    uint64_t end_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // cudaDeviceSynchronize();
    for(int i = 0; i < n; i++){
        cmap[i] = vals[i];
    }
    cudaFree(vals);
    return std::pair<std::vector<size_t>, uint64_t>{cmap, end_time - start_time};
}

__global__ void scalar_kernel(int factor, int* m, int s){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < s * s; i += stride){
        m[i] *= factor;
    }
}

__global__ void transpose_kernel(int* m, int s){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < s * s - 1; i += stride * (s + 1)){
        int c = i + 1;
        for(int r = i + s; r < s * s; r += s){
            int temp = m[r];
            m[r] = m[c];
            m[c] = temp;
            c++;
        }
    }
}

__global__ void matrix_kernel(int* product, int* m1, int* m2, int s){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < s * s; i += stride){
        product[i] = 0;
        int init_r = s * (i / s);
        int init_c = s * (i % s);
        int r = init_r;
        int c = init_c;
        while(r < init_r + s && c < init_c + s){
            product[i] += m1[r] * m2[c];
            r++;
            c++;
        }
    }
}

__global__ void add_kernel(int* m1, int* m2, int s){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < s * s; i += stride){
        m1[i] += m2[i];
    }
}

std::pair<std::vector<std::vector<int>>, uint64_t> two_mm_cuda(int alpha, int beta, std::vector<std::vector<int>> a, std::vector<std::vector<int>> b, std::vector<std::vector<int>> c, std::vector<std::vector<int>> d){
    int s = a.size();
    int* aPtr;
    int* bPtr;
    int* cPtr;
    int* dPtr;
    int* bc;
    int* abc;
    cudaMallocManaged(&aPtr, s*s*sizeof(int));
    cudaMallocManaged(&bPtr, s*s*sizeof(int));
    cudaMallocManaged(&cPtr, s*s*sizeof(int));
    cudaMallocManaged(&dPtr, s*s*sizeof(int));
    cudaMallocManaged(&bc, s*s*sizeof(int));
    cudaMallocManaged(&abc, s*s*sizeof(int));
    int col = 0;
    int row = 0;
    for(int i = 0; i < s * s; i++){
        aPtr[i] = a[row][col];
        bPtr[i] = b[row][col];
        cPtr[i] = c[row][col];
        dPtr[i] = d[row][col];
        col++;
        if(col >= s){
            col = 0;
            row++;
        }
    }
    uint64_t start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    transpose_kernel<<<1, 1>>>(cPtr, s);
    scalar_kernel<<<1024, 1024>>>(alpha, aPtr, s);
    cudaDeviceSynchronize();
    for(int i = 0; i < s * s; i++){
        bc[i] = 0;
        int init_r = s * (i / s);
        int init_c = s * (i % s);
        row = init_r;
        col = init_c;
        while(row < init_r + s && col < init_c + s){
            bc[i] += bPtr[row] * cPtr[col];
            row++;
            col++;
        }
    }
    // matrix_kernel<<<1024, 1>>>(bc, bPtr, cPtr, s);
    // cudaDeviceSynchronize();
    transpose_kernel<<<1, 1>>>(bc, s);
    scalar_kernel<<<1024, 1024>>>(beta, dPtr, s);
    cudaDeviceSynchronize();
    for(int i = 0; i < s * s; i++){
        abc[i] = 0;
        int init_r = s * (i / s);
        int init_c = s * (i % s);
        row = init_r;
        col = init_c;
        while(row < init_r + s && col < init_c + s){
            abc[i] += aPtr[row] * bc[col];
            row++;
            col++;
        }
    }
    // matrix_kernel<<<1024, 1024>>>(abc, aPtr, bc, s);
    // cudaDeviceSynchronize();
    add_kernel<<<1024, 1024>>>(abc, dPtr, s);
    cudaDeviceSynchronize();
    uint64_t end_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::vector<std::vector<int>> abcVec(s, std::vector<int>(s));
    col = 0;
    row = 0;
    for(int i = 0; i < s * s; i++){
        abcVec[row][col] = abc[i];
        col++;
        if(col >= s){
            col = 0;
            row++;
        }
    }
    cudaFree(aPtr);
    cudaFree(bPtr);
    cudaFree(cPtr);
    cudaFree(dPtr);
    cudaFree(bc);
    cudaFree(abc);
    return std::pair<std::vector<std::vector<int>>, uint64_t>{abcVec, end_time - start_time};
}

__device__ void mutex_lock(int *mutex) {
    while (atomicCAS(mutex, 0, 1) == 1);
}

__device__ void mutex_unlock(int *mutex) {
    atomicExch(mutex, 0);
}

__global__ void ising_kernel(int* lattice, int* lockTable, int* roworder, int* colorder, int s, int t){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // int up, left, target, right, down;
    int attempts;
    bool backoff = false;;
    for(int i = index; i < s; i += stride){
        // printf("%d\n", i); 
        int r = roworder[i];
        int c = colorder[i];
        if(r == 0){
            if(c == 0){
                attempts = 0;
                // printf("1\n");
                while(atomicCAS(&lockTable[0], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("2\n");
                while(atomicCAS(&lockTable[1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("3\n");
                while(atomicCAS(&lockTable[t - 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("4\n");
                while(atomicCAS(&lockTable[t], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("5\n");
                while(atomicCAS(&lockTable[t * (t - 1)], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                int cost = 2 * lattice[0] * (lattice[1] + lattice[t - 1] + lattice[t] + lattice[t * (t - 1)]);
                if (cost < 0){
                    lattice[0] *= -1;
                }
                atomicExch(&lockTable[0], 0);
                atomicExch(&lockTable[1], 0);
                atomicExch(&lockTable[t - 1], 0);
                atomicExch(&lockTable[t], 0);
                atomicExch(&lockTable[t * (t - 1)], 0);
            } else if(c == t - 1){
                // printf("6\n");
                while(atomicCAS(&lockTable[0], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("7\n");
                while(atomicCAS(&lockTable[t - 2], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("8\n");
                while(atomicCAS(&lockTable[t - 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("9\n");
                while(atomicCAS(&lockTable[2 * t - 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("10\n");
                while(atomicCAS(&lockTable[t * t - 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                int cost = 2 * lattice[0] * (lattice[t - 2] + lattice[t - 1] + lattice[2 * t - 1] + lattice[t * t - 1]);
                if (cost < 0){
                    lattice[t - 1] *= -1;
                }
                atomicExch(&lockTable[0], 0);
                atomicExch(&lockTable[t - 2], 0);
                atomicExch(&lockTable[t - 1], 0);
                atomicExch(&lockTable[2 * t - 1], 0);
                atomicExch(&lockTable[t * t - 1], 0);
            } else{
                // printf("11\n");
                while(atomicCAS(&lockTable[c - 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("12\n");
                while(atomicCAS(&lockTable[c], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("13\n");
                while(atomicCAS(&lockTable[c + 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("14\n");
                while(atomicCAS(&lockTable[t + c], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("15\n");
                while(atomicCAS(&lockTable[t * (t - 1) + c], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                int cost = 2 * lattice[c] * (lattice[c - 1] + lattice[c + 1] + lattice[t + c] + lattice[t * (t - 1) + c]);
                if (cost < 0){
                    lattice[c] *= -1;
                }
                atomicExch(&lockTable[c - 1], 0);
                atomicExch(&lockTable[c], 0);
                atomicExch(&lockTable[c + 1], 0);
                atomicExch(&lockTable[t + c], 0);
                atomicExch(&lockTable[t * (t - 1) + c], 0);
            }
        } else if(r == t - 1){
            if(c == 0){
                // printf("16\n");
                while(atomicCAS(&lockTable[0], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("17\n");
                while(atomicCAS(&lockTable[t * (t - 2)], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("18\n");
                while(atomicCAS(&lockTable[t * (t - 1)], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("19\n");
                while(atomicCAS(&lockTable[t * (t - 1) + 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("20\n");
                while(atomicCAS(&lockTable[t * t - 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                int cost = 2 * lattice[t * (t - 1)] * (lattice[0] + lattice[t * (t - 2)] + lattice[t * (t - 1) + 1] + lattice[t * t - 1]);
                if (cost < 0){
                    lattice[t * (t - 1)] *= -1;
                }
                atomicExch(&lockTable[0], 0);
                atomicExch(&lockTable[t * (t - 2)], 0);
                atomicExch(&lockTable[t * (t - 1)], 0);
                atomicExch(&lockTable[t * (t - 1) + 1], 0);
                atomicExch(&lockTable[t * t - 1], 0);
            } else if(c == t - 1){
                // printf("21\n");
                while(atomicCAS(&lockTable[t - 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("22\n");
                while(atomicCAS(&lockTable[t * (t - 1) - 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("23\n");
                while(atomicCAS(&lockTable[t * (t - 1)], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("24\n");
                while(atomicCAS(&lockTable[t * (t - 1) + 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("25\n");
                while(atomicCAS(&lockTable[t * t - 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                int cost = 2 * lattice[t * t - 1] * (lattice[t - 1] + lattice[t * (t - 1) - 1] + lattice[t * (t - 1)] + lattice[t * (t - 1) + 1]);
                if (cost < 0){
                    lattice[t * t - 1] *= -1;
                }
                atomicExch(&lockTable[t - 1], 0);
                atomicExch(&lockTable[t * (t - 1) - 1], 0);
                atomicExch(&lockTable[t * (t - 1)], 0);
                atomicExch(&lockTable[t * (t - 1) + 1], 0);
                atomicExch(&lockTable[t * t - 1], 0);
            } else{
                // printf("26\n");
                while(atomicCAS(&lockTable[c], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("27\n");
                while(atomicCAS(&lockTable[t * (t - 2) + c], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("28\n");
                while(atomicCAS(&lockTable[t * (t - 1) + c - 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("29\n");
                while(atomicCAS(&lockTable[t * (t - 1) + c], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("30\n");
                while(atomicCAS(&lockTable[t * (t - 1) + c + 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                int cost = 2 * lattice[t * (t - 1) + c] * (lattice[c] + lattice[t * (t - 2) + c] + lattice[t * (t - 1) + c - 1] + lattice[t * (t - 1) + c + 1]);
                if (cost < 0){
                    lattice[t * (t - 1) + c] *= -1;
                }
                atomicExch(&lockTable[c], 0);
                atomicExch(&lockTable[t * (t - 2) + c], 0);
                atomicExch(&lockTable[t * (t - 1) + c - 1], 0);
                atomicExch(&lockTable[t * (t - 1) + c], 0);
                atomicExch(&lockTable[t * (t - 1) + c + 1], 0);
            }
        } else{
            if(c == 0){
                // printf("31\n");
                while(atomicCAS(&lockTable[(r - 1) * t - 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("32\n");
                while(atomicCAS(&lockTable[r * t], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("33\n");
                while(atomicCAS(&lockTable[r * t + 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("34\n");
                while(atomicCAS(&lockTable[(r + 1) * t - 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("35\n");
                while(atomicCAS(&lockTable[(r + 1) * t], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                int cost = 2 * lattice[r * t] * (lattice[(r - 1) * t - 1] + lattice[r * t + 1] + lattice[(r + 1) * t - 1] + lattice[(r + 1) * t]);
                if (cost < 0){
                    lattice[r * t] *= -1;
                }
                atomicExch(&lockTable[(r - 1) * t - 1], 0);
                atomicExch(&lockTable[r * t], 0);
                atomicExch(&lockTable[r * t + 1], 0);
                atomicExch(&lockTable[(r + 1) * t - 1], 0);
                atomicExch(&lockTable[(r + 1) * t], 0);
            } else if(c == t - 1){
                // printf("36\n");
                while(atomicCAS(&lockTable[r * t - 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("37\n");
                while(atomicCAS(&lockTable[r * t], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("38\n");
                while(atomicCAS(&lockTable[r * t + 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("39\n");
                while(atomicCAS(&lockTable[(r + 1) * t - 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("40\n");
                while(atomicCAS(&lockTable[(r + 2) * t - 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                int cost = 2 * lattice[r * (t + 1) - 1] * (lattice[r * t - 1] + lattice[r * t] + lattice[r * t + 1] + lattice[(r + 2) * t - 1]);
                if (cost < 0){
                    lattice[(r + 1) * t - 1] *= -1;
                }
                atomicExch(&lockTable[r * t - 1], 0);
                atomicExch(&lockTable[r * t], 0);
                atomicExch(&lockTable[r * t + 1], 0);
                atomicExch(&lockTable[(r + 1) * t - 1], 0);
                atomicExch(&lockTable[(r + 2) * t - 1], 0);
            } else{
                // printf("41\n");
                while(atomicCAS(&lockTable[(r - 1) * t + c], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("42\n");
                while(atomicCAS(&lockTable[r * t + c - 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("43\n");
                while(atomicCAS(&lockTable[r * t + c], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("44\n");
                while(atomicCAS(&lockTable[r * t + c + 1], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                // printf("45\n");
                while(atomicCAS(&lockTable[(r + 1) * t + c], 0, 1) == 1){
                    attempts++;
                    if(attempts > 65536){
                        backoff = true;
                        break;
                    }
                } 
                if(backoff){
                    continue;
                }
                int cost = 2 * lattice[r * t + c] * (lattice[(r - 1) * t + c] + lattice[r * t + c - 1] + lattice[r * t + c + 1] + lattice[(r + 1) * t + c]);
                if (cost < 0){
                    lattice[r * t + c] *= -1;
                }
                atomicExch(&lockTable[(r - 1) * t + c], 0);
                atomicExch(&lockTable[r * t + c - 1], 0);
                atomicExch(&lockTable[r * t + c], 0);
                atomicExch(&lockTable[r * t + c + 1], 0);
                atomicExch(&lockTable[(r + 1) * t + c], 0);
            }
        }
    }
}

std::pair<std::vector<std::vector<int>>, uint64_t> ising_cuda(std::vector<std::vector<int>> lattice, std::vector<std::pair<int, int>> order){
    int s = lattice.size();
    int* latticePtr;
    int* lockTable;
    int* roworder;
    int* colorder;
    cudaMallocManaged(&latticePtr, s*s*sizeof(int));
    cudaMallocManaged(&lockTable, s*s*sizeof(int));
    cudaMallocManaged(&roworder, order.size()*sizeof(int));
    cudaMallocManaged(&colorder, order.size()*sizeof(int));
    int col = 0;
    int row = 0;
    for(int i = 0; i < s * s; i++){
        latticePtr[i] = lattice[row][col];
        col++;
        if(col >= s){
            col = 0;
            row++;
        }
    }
    for(int i = 0; i < order.size(); i++){
        roworder[i] = order[i].first;
        colorder[i] = order[i].second;
    }
    uint64_t start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    ising_kernel<<<1, 1024>>>(latticePtr, lockTable, roworder, colorder, order.size(), s);
    cudaDeviceSynchronize();
    uint64_t end_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    col = 0;
    row = 0;
    for(int i = 0; i < s * s; i++){
        lattice[row][col] = latticePtr[i];
        col++;
        if(col >= s){
            col = 0;
            row++;
        }
    }
    return std::pair<std::vector<std::vector<int>>, uint64_t>{lattice, end_time - start_time};
}
