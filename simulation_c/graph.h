/*
Module containing graph strutures and functions to maniplulate graphs
*/
typedef struct gnode gnode_t;

/*edge_t defined so it may be expanded later to include other fields*/
typedef gnode_t* edge_t;

typedef struct {
    gnode_t **nodes;
    int size_nodes;
    int n_nodes;
} graph_t;

/*creates a new graph with no nodes*/
graph_t *graph_create_empty(int size_nodes_init);

void graph_free(graph_t *graph, void (*free_node_data)(void*));

/*adds a new node to the graph with no edges*/
gnode_t *graph_add_node(graph_t *graph, void *data);

/*adds a node into array position specified by id.
    does not check if id is within the bounds of the array, and does not
    ensure there are no gaps in the array
    intended to be used as an to add nodes to a graph in parallel when the
    number of nodes is known and the array is already a suitible size*/
gnode_t *graph_add_node_by_id(graph_t *graph, void *data, int id);

/*creates a directed edge from src to dest*/
void graph_direct_edge(gnode_t *src, gnode_t *dest);

/*creates a undirected edge between node1 and node2*/
void graph_undirect_edge(gnode_t *node1, gnode_t *node2);

/*returns the outdegree of the node*/
int graph_node_outdegree(gnode_t *node);

/*gets the adjacent nodes to the node given. a pointer to the array is stored
in *adj_array and the length of the array is returned*/
int graph_adj_nodes(gnode_t *node, gnode_t **adj_array[]);

/*replaces the data currently stored at the node with new_data. the original
data is returned*/
void *graph_node_change_data(gnode_t *node, void *new_data);

/*returns the data pointer for the node given*/
void *graph_node_get_data(gnode_t *node);

/*Applies action function to the data of every node*/
void graph_process(graph_t *graph, void (*action)(void*));

/*prints the graph as an adjacency list using the index of each node as its id*/
void graph_print_as_adj_list(graph_t *graph, void (*print_node_data)(void*));
