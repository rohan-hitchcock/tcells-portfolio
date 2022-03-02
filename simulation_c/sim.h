/*functions for simulation of tcell movement in graph*/
#include "graph.h"

typedef struct gnode_data gnode_data_t;

/*return the number of nodes covered by n_tcells in n_iter iterations*/
int sim_tcell_coverage(graph_t *graph, int n_tcells, int n_iter);

/*reset all graph node data following a simulation*/
void reset_graph(graph_t *graph);

/*creates and intitialises node data*/
gnode_data_t *create_gnode_data(void);
