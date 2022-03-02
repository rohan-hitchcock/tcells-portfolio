#include "pcg_basic.h"

typedef struct threaded_rand threaded_rand_t;

//creates a seed for a random number generator for each thread
threaded_rand_t *seed_threads();

//returns the state of the random number generator for the calling thread
pcg32_random_t *get_thread_state(threaded_rand_t *this);

void free_threaded_rand(threaded_rand_t *this);
