/*
Module containing graph strutures and functions to maniplulate graphs
*/
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "graph.h"

#define NODE_ARRAY_GROWTH 2
#define EDGE_ARRAY_GROWTH 2
#define EDGE_ARRAY_INITIAL_SIZE 2

struct gnode {
    edge_t *edges;
    int n_edges;
    int size_edges;

    void *data;
};

/*creates a new node with no incomming or outgoing edges*/
gnode_t *create_node(void *data);

/*frees a node, returning its data*/
void *free_node(gnode_t *node);

/*****************************************************************************/
graph_t *graph_create_empty(int size_nodes_init) {
    graph_t *new_graph;

    new_graph = malloc(sizeof(*new_graph));
    assert(new_graph);

    new_graph->nodes = malloc(sizeof(*new_graph->nodes)*size_nodes_init);
    assert(new_graph->nodes);

    new_graph->size_nodes = size_nodes_init;
    new_graph->n_nodes = 0;

    return new_graph;
}

/*****************************************************************************/
void graph_free(graph_t *graph, void (*free_node_data)(void*)) {
    int i;

    for (i=0; i < graph->n_nodes; i++) {
        free_node_data(free_node(graph->nodes[i]));
    }

    free(graph->nodes);
    free(graph);
}

/*****************************************************************************/
gnode_t *graph_add_node(graph_t *graph, void *data) {
    gnode_t *new_node;

    new_node = create_node(data);

    /*grows node array if nessesary*/
    if (graph->n_nodes == graph->size_nodes) {
        graph->size_nodes *= NODE_ARRAY_GROWTH;
        graph->nodes = realloc(graph->nodes, graph->size_nodes*sizeof(*graph->nodes));
        assert(graph->nodes);
    }

    graph->nodes[graph->n_nodes++] = new_node;

    return new_node;
}

/*****************************************************************************/
gnode_t *graph_add_node_by_id(graph_t *graph, void *data, int id) {
    gnode_t *new_node;

    new_node = create_node(data);

    graph->nodes[id] = new_node;

    return new_node;
}

/*****************************************************************************/
void graph_direct_edge(gnode_t *src, gnode_t *dest) {

    /*grow edge array if nessesary*/
    if (src->size_edges == src->n_edges) {
        src->size_edges *= EDGE_ARRAY_GROWTH;
        src->edges = realloc(src->edges, src->size_edges*sizeof(src->edges));
        assert(src->edges);
    }

    src->edges[src->n_edges++] = dest;
}

/*****************************************************************************/
void graph_undirect_edge(gnode_t *node1, gnode_t *node2) {
    graph_direct_edge(node1, node2);
    graph_direct_edge(node2, node1);
}
/*****************************************************************************/
int graph_node_outdegree(gnode_t *node) {
    return node->n_edges;
}
/*****************************************************************************/
int graph_adj_nodes(gnode_t *node, gnode_t **adj_array[]) {
    *adj_array = node->edges;
    return node->n_edges;
}

/*****************************************************************************/
void *graph_node_get_data(gnode_t *node) {
    return node->data;
}

/*****************************************************************************/
void *graph_node_change_data(gnode_t *node, void *new_data) {
    void *old_data;

    old_data = node->data;
    node->data = new_data;
    return old_data;
}

/*****************************************************************************/
void graph_process(graph_t *graph, void (*action)(void*)) {
    int i;

    for (i=0; i < graph->n_nodes; i++) {
        action(graph->nodes[i]->data);
    }
}

/*****************************************************************************/
void graph_print_as_adj_list(graph_t *graph, void (*print_node_data)(void*)) {
    int i, j;
    gnode_t *curr_node;

    for (i=0; i < graph->n_nodes; i++) {
        curr_node = graph->nodes[i];
        print_node_data(curr_node->data);
        for (j=0; j < curr_node->n_edges; j++) {
            printf(" ");
            print_node_data(curr_node->edges[j]->data);
        }
        printf("\n");
    }
}

/*Helper functions************************************************************/
gnode_t *create_node(void *data) {
    gnode_t *new_node;

    new_node = malloc(sizeof(*new_node));
    assert(new_node);

    new_node->size_edges = EDGE_ARRAY_INITIAL_SIZE;
    new_node->n_edges = 0;
    new_node->data = data;

    new_node->edges = malloc(new_node->size_edges*sizeof(*new_node->edges));
    assert(new_node->edges);

    return new_node;
}

/*****************************************************************************/
void *free_node(gnode_t *node) {
    void *ret_data = node->data;
    free(node->edges);
    free(node);
    return ret_data;
}
