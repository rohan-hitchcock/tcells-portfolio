import os
import json

class SimReporter:

    def __init__(self, path, sim_name):
        """ Args:
                path: the path to directory to save results
                sim_name: the name of the simulation
        """

        self.path = path
        self.sim_name = sim_name

        #flags for different types of reporting
        self.report_coverage = False
        self.report_positions = False
        self.report_metadata = False
        
        #file pointers which may be opened
        self.positions_fp = None
        self.coverage_fp = None

        #other data required to report the requested data
        self.graph_size = None
        self.covered_nodes = None
        self.metadata = None

    def __enter__(self):

        os.makedirs(self.path, exist_ok=True)

        #write metadata upon entering the simulation contex
        if self.report_metadata:
            fname = os.path.join(self.path, self.sim_name + "_meta" + ".json")
            with open(fname, "w") as fp:
                json.dump(self.metadata, fp, indent=4)

        #open any required files here

        if self.report_positions:
            fname = os.path.join(self.path, self.sim_name + "_positions" + ".csv")
            self.positions_fp = open(fname, "w")
            self.positions_fp.write(f"iteration,cell_id,tail_id,head_id,edge_loc\n")
        
        if self.report_coverage:
            fname = os.path.join(self.path, self.sim_name + "_coverage" + ".csv")
            self.coverage_fp = open(fname, "w")
            self.coverage_fp.write("iteration,coverage\n")

        return self
            

    def __exit__(self, exc_type, exc_value, traceback):

        #close open files here 
        
        if self.report_positions:
            self.positions_fp.close()
        
        if self.report_coverage:
            self.coverage_fp.close()

    def set_report_coverage(self, graph_size):
        """ Call this method before the simulation to report cell coverage.

            Args:
                graph_size: the number of nodes in the tissue graph
        """

        self.report_coverage = True
        self.graph_size = graph_size
        self.covered_nodes = set()

    def set_report_positions(self):
        """ Call this method before the simulation to report cell positions."""
        self.report_positions = True
    
    def set_report_metadata(self, metadict):
        """ Call this method to report simulation metadata.

            Args:
                metadata: A dictionary of metadata. Keys can be any string.
        """
        self.report_metadata = True
        self.metadata = metadict

    def update_move(self, iteration, cell_id, cell_pos, path):
        """ Call this method every time a cell is moved in the simulation.

            Args:
                iteration: the current iteration (an integer)
                cell_id: an identifier of the cell being moved
                cell_pos: a GraphVector indicating the cell's new position
                path: the sequence of nodes which the cell moved through
        """
        if self.report_positions:
            self.positions_fp.write(f"{iteration},{cell_id},{cell_pos.tail},{cell_pos.head},{cell_pos.edge_loc}\n")
        
        if self.report_coverage:
            self.covered_nodes.update(path)

    def end_iteration(self, iteration):
        """ Call this method after each iteration.

            Args:
                iteration: the iteration which just ended (an integer)
        """
        if self.report_coverage:
            self.coverage_fp.write(f"{iteration},{len(self.covered_nodes) / self.graph_size}\n")

    