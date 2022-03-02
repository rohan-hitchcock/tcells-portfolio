/*functions for simulation of tcell movement in graph*/
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include <assert.h>
#include "sim.h"
#include "pcg_basic.h"

//number of iterations each tcell is moved before the start location is set
#define BURN_IN 0


typedef struct {
    gnode_t *locn;
    gnode_t *prev_locn;
} tcell_t;

struct gnode_data {
    omp_lock_t lock;
    bool visited;
};

/*****************************************************************************/
//intitialises the start locations of tcells in the graph, returning the number
//of nodes covered by this.
int init_tcell_locn(graph_t *graph, tcell_t *tcells, int n, pcg32_random_t *pcg32_state);

//moves a tcell to the next location. the node moved to is returned
gnode_t *tcell_move(tcell_t *tcell, pcg32_random_t *pcg32_state);

//these functions are swapped in and out to change how the tcell moves
gnode_t *choose_next_node_uniform(tcell_t *tcell, pcg32_random_t *pcg32_state);
gnode_t *choose_next_node_noreverse(tcell_t *tcell, pcg32_random_t *pcg32_state);

//returns the index of value in the array using the comparison function
int array_get_index(void *arr_v, int len, void *value, int (*cmp)(void*,void*));

//compares two pointers
int ptrcmp(void *p1, void *p2);

//generates a random integer in the set [0, max) \ ex_num
unsigned int gen_rand_exclude(pcg32_random_t *pcg32_state, int max, int ex_num);

/*****************************************************************************/
int sim_tcell_coverage(graph_t *graph, int n_tcells, int n_iter) {
    int i, j, coverage=0;
    pcg32_random_t pcg32_state;
    gnode_data_t *node_data;
    tcell_t tcells[n_tcells];

    #pragma omp parallel private(j, pcg32_state, node_data) reduction(+:coverage)
    {

        //seeding random number generator
        #pragma omp critical
            pcg32_srandom_r(&pcg32_state, rand(), rand());

        //intitalising tcell locations
        coverage += init_tcell_locn(graph, tcells, n_tcells, &pcg32_state);

        //simulating tcell movement
        #pragma omp for
        for (i=0; i < n_tcells; i++) {

            for (j=0; j < n_iter; j++) {
                node_data = graph_node_get_data(tcell_move(tcells+i, &pcg32_state));

                //if another thread owns the lock that means it has been visited
                if (omp_test_lock(&node_data->lock)) {
                    if (!node_data->visited) {
                        coverage += 1;
                        node_data->visited = true;
                    }
                    omp_unset_lock(&node_data->lock);
                }
            }
        }
    }
    return coverage;
}

/*****************************************************************************/
int init_tcell_locn(graph_t *graph, tcell_t *tcells, int n, pcg32_random_t *pcg32_state) {
    int i, j, coverage=0;
    gnode_data_t *node_data;

    //set each tcell to a random location in the graph
    #pragma omp for
    for (i=0; i < n; i++) {

        (tcells+i)->locn = graph->nodes[pcg32_boundedrand_r(pcg32_state, graph->n_nodes)];
        (tcells+i)->prev_locn = NULL;

        //'burn-in' the tcell location
        for (j=0; j < BURN_IN; j++) {
            tcell_move(tcells+i, pcg32_state);
        }

        //record final location after burn in
        node_data = graph_node_get_data((tcells+i)->locn);
        if (omp_test_lock(&node_data->lock)) {
            if (!node_data->visited) {
                coverage += 1;
                node_data->visited = true;
            }
            omp_unset_lock(&node_data->lock);
        }
        //remove previous location history
        (tcells+i)->prev_locn = NULL;
    }
    return coverage;
}

/*****************************************************************************/
void reset_graph(graph_t *graph) {
    int i;
    gnode_data_t *ndata;
    #pragma omp parallel for private(ndata)
        for (i=0; i < graph->n_nodes; i++) {
            ndata = graph_node_get_data(graph->nodes[i]);
            ndata->visited = false;
        }
}

/*****************************************************************************/
gnode_data_t *create_gnode_data(void) {
    gnode_data_t *new_data;

    new_data = malloc(sizeof(*new_data));
    assert(new_data);

    new_data->visited = false;
    omp_init_lock(&new_data->lock);

    return new_data;
}

/*****************************************************************************/
gnode_t *tcell_move(tcell_t *tcell, pcg32_random_t *pcg32_state) {
    gnode_t *next;
    next = choose_next_node_noreverse(tcell, pcg32_state);
    tcell->prev_locn = tcell->locn;
    tcell->locn = next;
    return tcell->locn;
}

/*Functions for changing movement pattern of tcells***************************/
gnode_t *choose_next_node_uniform(tcell_t *tcell, pcg32_random_t *pcg32_state) {
    int n_adj, i;
    gnode_t **adj_nodes;

    n_adj = graph_adj_nodes(tcell->locn, &adj_nodes);

    i = pcg32_boundedrand_r(pcg32_state, n_adj);

    return adj_nodes[i];
}

gnode_t *choose_next_node_noreverse(tcell_t *tcell, pcg32_random_t *pcg32_state) {
    int n_adj, i, prev_index;
    gnode_t **adj_nodes;

    n_adj = graph_adj_nodes(tcell->locn, &adj_nodes);
    prev_index = array_get_index(adj_nodes, n_adj, tcell->prev_locn, ptrcmp);

    //exclude previous node from choice
    if (prev_index >= 0) {
        i = gen_rand_exclude(pcg32_state, n_adj, prev_index);

    //use normal generation if the previous node is not adjacent (this should
    //only happen when the previous node does not exist and is NULL)
    } else {
        i = pcg32_boundedrand_r(pcg32_state, n_adj);
    }


    return adj_nodes[i];
}

/*****************************************************************************/
int array_get_index(void *arr_v, int len, void *value, int (*cmp)(void*,void*)) {
    void **arr = arr_v;
    int i;

    for (i=0; i < len; i++) {
        if (!cmp(arr[i], value)) {
            return i;
        }
    }
    return -1;
}

/*****************************************************************************/
int ptrcmp(void *p1, void *p2) {
    return !(p1 == p2);
}

/*****************************************************************************/
unsigned int gen_rand_exclude(pcg32_random_t *pcg32_state, int max, int ex_num) {
    unsigned int rand_num;
    rand_num = pcg32_boundedrand_r(pcg32_state, max-1);
    return (rand_num >= ex_num) ? rand_num + 1 : rand_num;
}
