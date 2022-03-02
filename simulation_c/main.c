/*
Entry point to the program, handles reading and writing data.

*/
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include "sim.h"

//number of minutes per iteration
//#define MINS_PER_ITER 0.0382  //calculated from Brownian motion
#define MINS_PER_ITER 0.1715   //calculated using displacement data

//length of simulation in minutes
#define SIM_LENGTH_MINS 2880

//number of iterations over which to mesure coverage time
#define N_ITER (SIM_LENGTH_MINS / MINS_PER_ITER)

//the argument to srand(..)
#define RAND_SEED (clock())

//macros relating to command line arguments
#define NUM_COMMAND_LINE_ARGS 5
#define COMMAND_LINE_FORMAT "Expected arguments: <graph.mxf> <start> <stop> <step> <output.csv>\n"

//reading mxf files
#define MXF_NUM_NODES " nodeNumber=\"%d\"\n"
#define MXF_NUM_NODES " nodeNumber=\"%d\"\n"
#define MXF_UNIT_CONVERSION " modelToMicrometer=\"%lf\"\n"
#define MXF_NODE_LINE "<NetworkNode id=\"%d\" xf=\"%lf\" yf=\"%lf\" zf=\"%lf\" radius=\"%lf\" type=\"%d\" />\n"
#define MXF_EDGE_LINE "<NetworkSegment start=\"%d\" end=\"%d\" />\n"
#define READ_ERROR_MESSAGE "Error reading file: Unexpected EOF\n"

//macros related to writing data
#define CSV_HEADER "graph size, number of tcells, coverage\n"
#define PRINT_FORMAT "%d, %d, %d\n"

/*****************************************************************************/
/*moves to the start of the next line of fp, returning the last charecter read.
return of new line charecter indicates sucess*/
int skip_line(FILE *fp);

//prints the value at num to stdout as an integer
void printf_int(void *num);

//returns a graph stored in the file at fp, which is in the .mxf format
graph_t *read_mxf(FILE *fp);

/*****************************************************************************/
int main(int argc, char *argv[]) {
    int n_tcells_start, n_tcells_stop, n_tcells_step, n_tcells, coverage;
    graph_t *tissue_graph;
    FILE *graph_fp, *out_fp;


    //reading command line arguments
    if (argc-1 != NUM_COMMAND_LINE_ARGS) {
        fprintf(stderr, "%d arguments expected, %d given\n",
                                                NUM_COMMAND_LINE_ARGS, argc-1);
        fprintf(stderr, COMMAND_LINE_FORMAT);
        return 0;
    }

    graph_fp = fopen(argv[1], "r");

    n_tcells_start = atoi(argv[2]);
    n_tcells_stop = atoi(argv[3]);
    n_tcells_step = atoi(argv[4]);

    tissue_graph = read_mxf(graph_fp);
    fclose(graph_fp);
    assert(tissue_graph);

    srand(RAND_SEED);

    out_fp = fopen(argv[5], "w");

    fprintf(out_fp, CSV_HEADER);
    for (n_tcells=n_tcells_start; n_tcells<=n_tcells_stop; n_tcells+=n_tcells_step) {
        coverage = sim_tcell_coverage(tissue_graph, n_tcells, N_ITER);
        fprintf(out_fp, PRINT_FORMAT, tissue_graph->n_nodes, n_tcells, coverage);
        reset_graph(tissue_graph);
    }

    fclose(out_fp);

    graph_free(tissue_graph, free);
    return 0;
}


/*****************************************************************************/
graph_t *read_mxf(FILE *fp) {
    int num_nodes, id, type, first_id, start_id, end_id, n;
    double unit_conversion_factor, x, y, z, radius;
    graph_t *graph;

    //skip header of file, stop once the number of nodes has been read
    while ((n=fscanf(fp, MXF_NUM_NODES, &num_nodes)) != 1) {
        if (skip_line(fp) != '\n') {
            fprintf(stderr, READ_ERROR_MESSAGE);
            return NULL;
        }
    }

    graph = graph_create_empty(num_nodes);

    //read conversion factor
    while (fscanf(fp, MXF_UNIT_CONVERSION, &unit_conversion_factor) != 1) {
        if (skip_line(fp) != '\n') {
            fprintf(stderr, READ_ERROR_MESSAGE);
            return NULL;
        }
    }
    //nothing is done with this value at the moment, captured for completeness

    //skip to node list
    while (fscanf(fp, MXF_NODE_LINE, &id, &x, &y, &z, &radius, &type) != 6) {
        if (skip_line(fp) != '\n') {
            fprintf(stderr, READ_ERROR_MESSAGE);
            return NULL;
        }
    }

    //save first node id to account for node id's starting from any number
    first_id = id;
    graph_add_node(graph, create_gnode_data());

    //read node list
    while ((fscanf(fp, MXF_NODE_LINE, &id, &x, &y, &z, &radius, &type) == 6)) {
        graph_add_node(graph, create_gnode_data());
        //nothing is done with node attributes, captured for completeness
    }

    //skip to edge list
    while ((fscanf(fp, MXF_EDGE_LINE, &start_id, &end_id) != 2)) {
        if (skip_line(fp) != '\n') {
            fprintf(stderr, READ_ERROR_MESSAGE);
            return NULL;
        }
    }

    start_id = start_id - first_id;
    end_id = end_id - first_id;
    graph_undirect_edge(graph->nodes[start_id], graph->nodes[end_id]);

    //read edge list
    while (fscanf(fp, MXF_EDGE_LINE, &start_id, &end_id) == 2) {
        start_id = start_id - first_id;
        end_id = end_id - first_id;
        graph_undirect_edge(graph->nodes[start_id], graph->nodes[end_id]);
    }

    return graph;
}

/*****************************************************************************/
int skip_line(FILE *fp) {
    char ch;
    while ((ch=getc(fp)) != EOF && ch != '\n');
    return ch;
}
