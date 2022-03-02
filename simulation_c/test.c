#include "threaded_random.h"
#include <stdio.h>
#include <omp.h>


int main(int argc, char **argv) {
    threaded_rand_t *thread_seeds;

    thread_seeds = seed_threads();

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < 10; i++) {
            printf("%d (TID %d)\n",
                   pcg32_boundedrand_r(get_thread_state(thread_seeds), 20),
                   omp_get_thread_num());
        }
    }

    return 0;
}
