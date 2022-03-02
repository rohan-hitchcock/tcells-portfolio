#include "threaded_random.h"
#include <time.h>
#include <assert.h>
#include <stdlib.h>
#include <omp.h>

//the argument to srand(..)
#define SRAND_SEED (clock())

struct threaded_rand {
    pcg32_random_t *thread_states;
    int num_threads;
};

/******************************************************************************/
threaded_rand_t *seed_threads() {
    threaded_rand_t *new;

    new = malloc(sizeof(*new));
    assert(new);

    new->num_threads = omp_get_max_threads();
    new->thread_states = malloc(new->num_threads * sizeof(*new->thread_states));
    assert(new->thread_states);

    //this is a bit lazy and creating the pcg_seeds could be improved
    srand(SRAND_SEED);
    for (int i = 0; i < new->num_threads; i++) {
        pcg32_srandom_r(new->thread_states+i, rand(), rand());
    }

    return new;
}

/******************************************************************************/
pcg32_random_t *get_thread_state(threaded_rand_t *this) {
    return this->thread_states + omp_get_thread_num();
}

/******************************************************************************/
void free_threaded_rand(threaded_rand_t *this) {
    free(this->thread_states);
    free(this);
}
