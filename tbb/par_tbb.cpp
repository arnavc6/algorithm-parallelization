#include <vector>
#include <unordered_map>
#include <math.h>
#include <chrono>
#include <mutex>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_invoke.h>

std::pair<std::vector<size_t>, uint64_t> collatz_tbb(size_t n){
    std::vector<size_t> cmap(n);
    std::vector<std::mutex*> locks;
    for(size_t i = 0; i < n; i++){
        locks.push_back(new std::mutex());
    }
    uint64_t start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    tbb::parallel_for(tbb::blocked_range<size_t>(1, n), [&](tbb::blocked_range<size_t> r){
        for(size_t i = r.begin(); i <= r.end(); i++){
            // std::cout << i << std::endl;
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
            locks[i - 1]->lock();
            cmap[i - 1] = steps;
            locks[i - 1]->unlock();
        }
    });
    uint64_t end_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    return std::pair<std::vector<size_t>, uint64_t>{cmap, end_time - start_time};
}

// std::pair<std::unordered_map<size_t, int>, uint64_t> collatz_tbb(size_t n){
//     std::unordered_map<size_t, int> cmap;
//     uint64_t start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//     tbb::parallel_for(tbb::blocked_range<size_t>(1, n), [&](tbb::blocked_range<size_t> r){
//         for(size_t i = r.begin(); i <= r.end(); i++){
//             int steps = 0;
//             size_t j = i;
//             while(j != 1){
//                 if(j % 2 == 0){
//                     j /= 2;
//                 } else{
//                     j *= 3;
//                 }
//             }
//         }
//     });
//     uint64_t end_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//     return std::pair<std::unordered_map<size_t, int>, uint64_t>{cmap, end_time - start_time};
// }

std::pair<std::vector<std::vector<int>>, uint64_t> two_mm_tbb(int alpha, int beta, std::vector<std::vector<int>> a, std::vector<std::vector<int>> b, std::vector<std::vector<int>> c, std::vector<std::vector<int>> d){
    int s = a.size();
    std::vector<std::vector<int>> bc(s, std::vector<int>(s, 0));
    std::vector<std::vector<int>> abc(s, std::vector<int>(s, 0));
    auto scalar_mult_1 = [&]{
        for(int i = 0; i < s; i++){
            for(int j = 0; j < s; j++){
                a[i][j] *= alpha;
            }
        }
        // tbb::parallel_for(tbb::blocked_range<int>(0, s), [&](tbb::blocked_range<int> r){
        //     for(int i = r.begin(); i < r.end(); i++){
        //         tbb::parallel_for(tbb::blocked_range<int>(0, s), [&](tbb::blocked_range<int> q){
        //             for(int j = q.begin(); j < q.end(); j++){
        //                 a[i][j] *= alpha;
        //             }
        //         });
        //     }
        // });
    };
    auto matrix_mult_1 = [&]{
        tbb::parallel_for(tbb::blocked_range<int>(0, s), [&](tbb::blocked_range<int> r){
            for(int i = r.begin(); i < r.end(); i++){
                tbb::parallel_for(tbb::blocked_range<int>(0, s), [&](tbb::blocked_range<int> q){
                    for(int j = q.begin(); j < q.end(); j++){
                        for(int k = 0; k < s; k++){
                            bc[i][j] += b[i][k] * c[k][j];
                        }
                    }
                });
            }
        });
    };
    auto scalar_mult_2 = [&]{
        for(int i = 0; i < s; i++){
            for(int j = 0; j < s; j++){
                d[i][j] *= beta;
            }
        }
        // tbb::parallel_for(tbb::blocked_range<int>(0, s), [&](tbb::blocked_range<int> r){
        //     for(int i = r.begin(); i < r.end(); i++){
        //         tbb::parallel_for(tbb::blocked_range<int>(0, s), [&](tbb::blocked_range<int> q){
        //             for(int j = q.begin(); j < q.end(); j++){
        //                 d[i][j] *= beta;
        //             }
        //         });
        //     }
        // });
    };
    auto matrix_mult_2 = [&]{
        tbb::parallel_for(tbb::blocked_range<int>(0, s), [&](tbb::blocked_range<int> r){
            for(int i = r.begin(); i < r.end(); i++){
                tbb::parallel_for(tbb::blocked_range<int>(0, s), [&](tbb::blocked_range<int> q){
                    for(int j = q.begin(); j < q.end(); j++){
                        for(int k = 0; k < s; k++){
                            abc[i][j] += a[i][k] * bc[k][j];
                        }
                    }
                });
            }
        });
    };
    auto final_add = [&]{
        for(int i = 0; i < s; i++){
            for(int j = 0; j < s; j++){
                abc[i][j] += d[i][j];
            }
        }
        // tbb::parallel_for(tbb::blocked_range<int>(0, s), [&](tbb::blocked_range<int> r){
        //     for(int i = r.begin(); i < r.end(); i++){
        //         tbb::parallel_for(tbb::blocked_range<int>(0, s), [&](tbb::blocked_range<int> q){
        //             for(int j = q.begin(); j < q.end(); j++){
        //                 abc[i][j] += d[i][j];
        //             }
        //         });
        //     }
        // });
    };
    uint64_t start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    tbb::parallel_invoke(scalar_mult_1, matrix_mult_1);
    tbb::parallel_invoke(scalar_mult_2, matrix_mult_2);
    final_add();
    uint64_t end_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    return std::pair<std::vector<std::vector<int>>, uint64_t>{abc, end_time - start_time};
}

std::pair<std::vector<std::vector<int>>, uint64_t> ising_tbb(std::vector<std::vector<int>> lattice, std::vector<std::pair<int, int>> order){
    const int s = lattice.size();
    std::vector<std::vector<std::mutex*>> lock_table;
    for(int i = 0; i < s; i++){
    	std::vector<std::mutex*> row;
    	for(int j = 0; j < s; j++){
		row.push_back(new std::mutex());
	}
	lock_table.push_back(row);
    }
    uint64_t start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    tbb::parallel_for(tbb::blocked_range<int>(0, order.size()), [&](tbb::blocked_range<int> r){
        for(int i = r.begin(); i < r.end(); i++){
            int r = order[i].first;
            int c = order[i].second;
            if(r == 0){
                if(c == 0){
                    while(!lock_table[0][0]->try_lock()); //center
                    while(!lock_table[0][1]->try_lock());
                    while(!lock_table[0][s-1]->try_lock());
                    while(!lock_table[1][0]->try_lock());
                    while(!lock_table[s-1][0]->try_lock());
                    int cost = 2 * lattice[0][0] * (lattice[0][1] + lattice[0][s-1] + lattice[1][0] + lattice[s-1][0]);
                    if (cost < 0){
                        lattice[0][0] *= -1;
                    }
                    lock_table[0][0]->unlock();
                    lock_table[0][1]->unlock();
                    lock_table[0][s-1]->unlock();
                    lock_table[1][0]->unlock();
                    lock_table[s-1][0]->unlock();
                } else if(c == s - 1){
                    while(!lock_table[0][0]->try_lock());
                    while(!lock_table[0][s-2]->try_lock());
                    while(!lock_table[0][s-1]->try_lock()); //center
                    while(!lock_table[1][s-1]->try_lock());
                    while(!lock_table[s-1][s-1]->try_lock());
                    int cost = 2 * lattice[0][s-1] * (lattice[0][0] + lattice[0][s-2] + lattice[1][s-1] + lattice[s-1][s-1]);
                    if (cost < 0){
                        lattice[0][s-1] *= -1;
                    }
                    lock_table[0][0]->unlock();
                    lock_table[0][s-2]->unlock();
                    lock_table[0][s-1]->unlock(); 
                    lock_table[1][s-1]->unlock();
                    lock_table[s-1][s-1]->unlock();
                } else{
                    while(!lock_table[0][c-1]->try_lock());
                    while(!lock_table[0][c]->try_lock()); //center
                    while(!lock_table[0][c+1]->try_lock());
                    while(!lock_table[1][c]->try_lock());
                    while(!lock_table[s-1][c]->try_lock());
                    int cost = 2 * lattice[0][c] * (lattice[0][c-1] + lattice[1][c+1] + lattice[1][c] + lattice[s-1][c]);
                    if (cost < 0){
                        lattice[0][c] *= -1;
                    }
                    lock_table[0][c-1]->unlock();
                    lock_table[0][c]->unlock(); 
                    lock_table[0][c+1]->unlock();
                    lock_table[1][c]->unlock();
                    lock_table[s-1][c]->unlock();
                }
            } else if(r == s - 1){
                if(c == 0){
                    while(!lock_table[0][0]->try_lock());
                    while(!lock_table[s-2][0]->try_lock());
                    while(!lock_table[s-1][0]->try_lock()); //center
                    while(!lock_table[s-1][1]->try_lock());
                    while(!lock_table[s-1][s-1]->try_lock());
                    int cost = 2 * lattice[s-1][0] * (lattice[0][0] + lattice[s-2][0] + lattice[s-1][1] + lattice[s-1][s-1]);
                    if (cost < 0){
                        lattice[s-1][0] *= -1;
                    }
                    lock_table[0][0]->unlock();
                    lock_table[s-2][0]->unlock();
                    lock_table[s-1][0]->unlock(); 
                    lock_table[s-1][1]->unlock();
                    lock_table[s-1][s-1]->unlock();
                } else if(c == s - 1){
                    while(!lock_table[0][s-1]->try_lock());
                    while(!lock_table[s-2][s-1]->try_lock());
                    while(!lock_table[s-1][0]->try_lock());
                    while(!lock_table[s-1][s-2]->try_lock());
                    while(!lock_table[s-1][s-1]->try_lock()); //center
                    int cost = 2 * lattice[s-1][s-1] * (lattice[0][s-1] + lattice[s-2][s-1] + lattice[s-1][0] + lattice[s-1][s-2]);
                    if (cost < 0){
                        lattice[s-1][s-1] *= -1;
                    }
                    lock_table[0][s-1]->unlock();
                    lock_table[s-2][s-1]->unlock();
                    lock_table[s-1][0]->unlock();
                    lock_table[s-1][s-2]->unlock();
                    lock_table[s-1][s-1]->unlock(); 
                } else{
                    while(!lock_table[0][c]->try_lock());
                    while(!lock_table[s-2][c]->try_lock());
                    while(!lock_table[s-1][c-1]->try_lock());
                    while(!lock_table[s-1][c]->try_lock()); //center
                    while(!lock_table[s-1][c+1]->try_lock());
                    int cost = 2 * lattice[s-1][c] * (lattice[0][c] + lattice[s-2][c] + lattice[s-1][c-1] + lattice[s-1][c+1]);
                    if (cost < 0){
                        lattice[s-1][c] *= -1;
                    }
                    lock_table[0][c]->unlock();
                    lock_table[s-2][c]->unlock();
                    lock_table[s-1][c-1]->unlock();
                    lock_table[s-1][c]->unlock(); 
                    lock_table[s-1][c+1]->unlock();
                }
            } else{
                if(c == 0){
                    while(!lock_table[r-1][0]->try_lock());
                    while(!lock_table[r][0]->try_lock()); //center
                    while(!lock_table[r][1]->try_lock());
                    while(!lock_table[r][s-1]->try_lock());
                    while(!lock_table[r+1][0]->try_lock());
                    int cost = 2 * lattice[r][0] * (lattice[r-1][0] + lattice[r][1] + lattice[r][s-1] + lattice[r+1][0]);
                    if (cost < 0){
                        lattice[r][0] *= -1;
                    }
                    lock_table[r-1][0]->unlock();
                    lock_table[r][0]->unlock(); 
                    lock_table[r][1]->unlock();
                    lock_table[r][s-1]->unlock();
                    lock_table[r+1][0]->unlock();
                } else if(c == s - 1){
                    while(!lock_table[r-1][s-1]->try_lock());
                    while(!lock_table[r][0]->try_lock());
                    while(!lock_table[r][s-2]->try_lock());
                    while(!lock_table[r][s-1]->try_lock()); //center
                    while(!lock_table[r+1][s-1]->try_lock());
                    int cost = 2 * lattice[r][s-1] * (lattice[r-1][s-1] + lattice[r][0] + lattice[r][s-2] + lattice[r+1][s-1]);
                    if (cost < 0){
                        lattice[r][s-1] *= -1;
                    }
                    lock_table[r-1][s-1]->unlock();
                    lock_table[r][0]->unlock();
                    lock_table[r][s-2]->unlock();
                    lock_table[r][s-1]->unlock(); 
                    lock_table[r+1][s-1]->unlock();
                } else{
                    while(!lock_table[r-1][c]->try_lock());
                    while(!lock_table[r][c-1]->try_lock());
                    while(!lock_table[r][c]->try_lock()); //center
                    while(!lock_table[r][c+1]->try_lock());
                    while(!lock_table[r+1][c]->try_lock());
                    int cost = 2 * lattice[r][c] * (lattice[r-1][c] + lattice[r][c-1] + lattice[r][c+1] + lattice[r+1][c]);
                    if (cost < 0){
                        lattice[r][c] *= -1;
                    }
                    lock_table[r-1][c]->unlock();
                    lock_table[r][c-1]->unlock();
                    lock_table[r][c]->unlock(); 
                    lock_table[r][c+1]->unlock();
                    lock_table[r+1][c]->unlock();
                }
            }
        }
    });
    // tbb::parallel_for(tbb::blocked_range<int>(0, order.size()), [&](tbb::blocked_range<int> r){
    //     for(int i = r.begin(); i < r.end(); i++){
    //         int r = order[i].first;
    //         int c = order[i].second;
    //         if(r == 0){
    //             if(c == 0){
    //                 lock_table[0][0]->lock(); //center
    //                 lock_table[0][1]->lock();
    //                 lock_table[0][s-1]->lock();
    //                 lock_table[1][0]->lock();
    //                 lock_table[s-1][0]->lock();
    //                 int cost = 2 * lattice[0][0] * (lattice[0][1] + lattice[0][s-1] + lattice[1][0] + lattice[s-1][0]);
    //                 if (cost < 0){
    //                     lattice[0][0] *= -1;
    //                 }
    //                 lock_table[0][0]->unlock();
    //                 lock_table[0][1]->unlock();
    //                 lock_table[0][s-1]->unlock();
    //                 lock_table[1][0]->unlock();
    //                 lock_table[s-1][0]->unlock();
    //             } else if(c == s - 1){
    //                 lock_table[0][0]->lock();
    //                 lock_table[0][s-2]->lock();
    //                 lock_table[0][s-1]->lock(); //center
    //                 lock_table[1][s-1]->lock();
    //                 lock_table[s-1][s-1]->lock();
    //                 int cost = 2 * lattice[0][s-1] * (lattice[0][0] + lattice[0][s-2] + lattice[1][s-1] + lattice[s-1][s-1]);
    //                 if (cost < 0){
    //                     lattice[0][s-1] *= -1;
    //                 }
    //                 lock_table[0][0]->unlock();
    //                 lock_table[0][s-2]->unlock();
    //                 lock_table[0][s-1]->unlock(); 
    //                 lock_table[1][s-1]->unlock();
    //                 lock_table[s-1][s-1]->unlock();
    //             } else{
    //                 lock_table[0][c-1]->lock();
    //                 lock_table[0][c]->lock(); //center
    //                 lock_table[0][c+1]->lock();
    //                 lock_table[1][c]->lock();
    //                 lock_table[s-1][c]->lock();
    //                 int cost = 2 * lattice[0][c] * (lattice[0][c-1] + lattice[1][c+1] + lattice[1][c] + lattice[s-1][c]);
    //                 if (cost < 0){
    //                     lattice[0][c] *= -1;
    //                 }
    //                 lock_table[0][c-1]->unlock();
    //                 lock_table[0][c]->unlock(); 
    //                 lock_table[0][c+1]->unlock();
    //                 lock_table[1][c]->unlock();
    //                 lock_table[s-1][c]->unlock();
    //             }
    //         } else if(r == s - 1){
    //             if(c == 0){
    //                 lock_table[0][0]->lock();
    //                 lock_table[s-2][0]->lock();
    //                 lock_table[s-1][0]->lock(); //center
    //                 lock_table[s-1][1]->lock();
    //                 lock_table[s-1][s-1]->lock();
    //                 int cost = 2 * lattice[s-1][0] * (lattice[0][0] + lattice[s-2][0] + lattice[s-1][1] + lattice[s-1][s-1]);
    //                 if (cost < 0){
    //                     lattice[s-1][0] *= -1;
    //                 }
    //                 lock_table[0][0]->unlock();
    //                 lock_table[s-2][0]->unlock();
    //                 lock_table[s-1][0]->unlock(); 
    //                 lock_table[s-1][1]->unlock();
    //                 lock_table[s-1][s-1]->unlock();
    //             } else if(c == s - 1){
    //                 lock_table[0][s-1]->lock();
    //                 lock_table[s-2][s-1]->lock();
    //                 lock_table[s-1][0]->lock();
    //                 lock_table[s-1][s-2]->lock();
    //                 lock_table[s-1][s-1]->lock(); //center
    //                 int cost = 2 * lattice[s-1][s-1] * (lattice[0][s-1] + lattice[s-2][s-1] + lattice[s-1][0] + lattice[s-1][s-2]);
    //                 if (cost < 0){
    //                     lattice[s-1][s-1] *= -1;
    //                 }
    //                 lock_table[0][s-1]->unlock();
    //                 lock_table[s-2][s-1]->unlock();
    //                 lock_table[s-1][0]->unlock();
    //                 lock_table[s-1][s-2]->unlock();
    //                 lock_table[s-1][s-1]->unlock(); 
    //             } else{
    //                 lock_table[0][c]->lock();
    //                 lock_table[s-2][c]->lock();
    //                 lock_table[s-1][c-1]->lock();
    //                 lock_table[s-1][c]->lock(); //center
    //                 lock_table[s-1][c+1]->lock();
    //                 int cost = 2 * lattice[s-1][c] * (lattice[0][c] + lattice[s-2][c] + lattice[s-1][c-1] + lattice[s-1][c+1]);
    //                 if (cost < 0){
    //                     lattice[s-1][c] *= -1;
    //                 }
    //                 lock_table[0][c]->unlock();
    //                 lock_table[s-2][c]->unlock();
    //                 lock_table[s-1][c-1]->unlock();
    //                 lock_table[s-1][c]->unlock(); 
    //                 lock_table[s-1][c+1]->unlock();
    //             }
    //         } else{
    //             if(c == 0){
    //                 lock_table[r-1][0]->lock();
    //                 lock_table[r][0]->lock(); //center
    //                 lock_table[r][1]->lock();
    //                 lock_table[r][s-1]->lock();
    //                 lock_table[r+1][0]->lock();
    //                 int cost = 2 * lattice[r][0] * (lattice[r-1][0] + lattice[r][1] + lattice[r][s-1] + lattice[r+1][0]);
    //                 if (cost < 0){
    //                     lattice[r][0] *= -1;
    //                 }
    //                 lock_table[r-1][0]->unlock();
    //                 lock_table[r][0]->unlock(); 
    //                 lock_table[r][1]->unlock();
    //                 lock_table[r][s-1]->unlock();
    //                 lock_table[r+1][0]->unlock();
    //             } else if(c == s - 1){
    //                 lock_table[r-1][s-1]->lock();
    //                 lock_table[r][0]->lock();
    //                 lock_table[r][s-2]->lock();
    //                 lock_table[r][s-1]->lock(); //center
    //                 lock_table[r+1][s-1]->lock();
    //                 int cost = 2 * lattice[r][s-1] * (lattice[r-1][s-1] + lattice[r][0] + lattice[r][s-2] + lattice[r+1][s-1]);
    //                 if (cost < 0){
    //                     lattice[r][s-1] *= -1;
    //                 }
    //                 lock_table[r-1][s-1]->unlock();
    //                 lock_table[r][0]->unlock();
    //                 lock_table[r][s-2]->unlock();
    //                 lock_table[r][s-1]->unlock(); 
    //                 lock_table[r+1][s-1]->unlock();
    //             } else{
    //                 lock_table[r-1][c]->lock();
    //                 lock_table[r][c-1]->lock();
    //                 lock_table[r][c]->lock(); //center
    //                 lock_table[r][c+1]->lock();
    //                 lock_table[r+1][c]->lock();
    //                 int cost = 2 * lattice[r][c] * (lattice[r-1][c] + lattice[r][c-1] + lattice[r][c+1] + lattice[r+1][c]);
    //                 if (cost < 0){
    //                     lattice[r][c] *= -1;
    //                 }
    //                 lock_table[r-1][c]->unlock();
    //                 lock_table[r][c-1]->unlock();
    //                 lock_table[r][c]->unlock(); 
    //                 lock_table[r][c+1]->unlock();
    //                 lock_table[r+1][c]->unlock();
    //             }
    //         }
    //     }
    // });
    uint64_t end_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // for(int i = 0; i < s; i++){
    //     for(int j = 0; j < s; j++){
    //         delete lock_table[i][j];
    //     }
    // }
    lock_table.clear();
    return std::pair<std::vector<std::vector<int>>, uint64_t>{lattice, end_time - start_time};
}
