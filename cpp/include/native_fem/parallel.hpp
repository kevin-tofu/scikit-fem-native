#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace native_fem {

int num_threads();
void set_num_threads(int count);

inline std::size_t effective_threads(std::size_t count, int requested = 0) {
    const int configured = requested > 0 ? requested : num_threads();
    return std::min<std::size_t>(count,static_cast<std::size_t>(configured));
}

template <class Function>
void parallel_for_workers(std::size_t count, int requested, Function function) {
    const auto workers=effective_threads(count,requested);
    const std::size_t threshold=requested>0?128:1024;
    if(workers<=1||count<threshold){function(0,0,count);return;}
    std::vector<std::thread> threads;
    threads.reserve(workers);
    std::exception_ptr failure;
    std::mutex failure_mutex;
    for(std::size_t worker=0;worker<workers;++worker){
        const auto begin=count*worker/workers;
        const auto end=count*(worker+1)/workers;
        threads.emplace_back([&,worker,begin,end]{
            try{function(worker,begin,end);}
            catch(...){
                std::lock_guard<std::mutex> lock(failure_mutex);
                if(!failure)failure=std::current_exception();
            }
        });
    }
    for(auto&thread:threads)thread.join();
    if(failure)std::rethrow_exception(failure);
}

template <class Function>
void parallel_for(std::size_t count, Function function) {
    parallel_for_workers(count,0,[&](std::size_t,std::size_t begin,
                                     std::size_t end){function(begin,end);});
}

}  // namespace native_fem
