
import search
import glob
import time
from sokoban import Warehouse
from mySokobanSolver import *
import threading

all_warehouses = sorted(glob.glob('warehouses/warehouses/*.txt'))

class TimeoutException(Exception):
        pass

def isValidWarehouse(problemFile): 
    """ Check if the warehouse can be loaded successfully.

    @param problem_file: path to the warehouse file

    @return: False if there's an error, otherwise return True
    """
    try:
        warehouse = Warehouse()
        # Attempt to load the warehouse
        warehouse.load_warehouse(problemFile)
        return True  # Return the loaded warehouse if successful
    except AssertionError:
        # If there's an assertion error, return 'Invalid Warehouse'
        return False
    except Exception as e:
        # Handle any other unforeseen exceptions
        print(f"Error loading warehouse: {e}")
        return False


def solve_sokoban(warehouse, search_type="bfs", timeout=180):
    '''
    Takes a path to a warehouse file.
    search_type : bfs, dfs, greedy, astar, uniform
    timeout: the time in second given to solve a problem
    '''
    result = [None]

    def run_solver():
        try:
            sokoban = SokobanPuzzle(warehouse, macro=True, allow_taboo_push=False)
            
            # Use breadth-first search
            if search_type == 'bfs':
                solution_node = search.breadth_first_graph_search(sokoban)
            # Use breadth-first search
            if search_type == 'dfs':
                solution_node = search.depth_first_graph_search(sokoban)
            # Use A* search
            if search_type == "greedy":
                solution_node = search.best_first_graph_search(sokoban, sokoban.h)
            if search_type == "aStar":
                solution_node = search.astar_graph_search(sokoban, sokoban.h)
            if search_type == "uniform":
                solution_node = search.uniform_cost_search(sokoban)
            if solution_node is None:
                #catch the None return
                result[0] = 'Impossible'
            else:
                result[0] = solution_node.solution()
        except Exception as e:
            result[0] = str(e)

    # Create a thread to run the solver
    solver_thread = threading.Thread(target=run_solver)
    solver_thread.start()
    solver_thread.join(timeout)  # Wait for the solver to finish or timeout

    if solver_thread.is_alive():
        solver_thread.join(0)  # Make sure the thread terminates
        return "Timeout"
    
    return result[0]

def test_solver(result_filename, index_range=[], search_type="bfs"):
    failed_warehouse = []
    solutions = []
    steps = []
    times = []
    #create a file witht the result
    with open(result_filename, "w") as file:
        #header for the file
        file.write("Warehouse,TimeTaken,Result,Cost\n")

        for i in index_range:
            try:
                filename = f'warehouses/warehouses/warehouse_{i:02}.txt'
                #if the warehouse cannot be loaded, skip
                if not isValidWarehouse(filename):
                    continue
            except Exception as error:
                continue
            print(f'Testing {filename }')
            start_time = time.time()
            # Load the warehouse
            wh = Warehouse()
            wh.load_warehouse(filename)
            # Solve the Sokoban puzzle
            solution = solve_sokoban(wh, search_type=search_type)
            total_time = time.time() - start_time
            times.append(total_time)
            
            if solution == "Timeout":
                print(f'Timed out: {filename }')
                failed_warehouse.append(filename)
                file.write(f"{i},{total_time},Timeout,0\n")
            elif solution == "Impossible":
                solutions.append("Impossible")
                file.write(f"{i},{total_time},Impossible,0\n")
            else:
                solutions.append(solution)
                steps.append(len(solution))
                file.write(f"{i},{total_time},Solved,{len(solution)}\n")
    print(f"It took {sum(times)} to complete the test. The solver failed to solve {len(failed_warehouse)} probles, and the total number of steps is {sum(steps)}")
    return times, len(failed_warehouse), steps, solutions


if __name__ == "__main__":

    
    indexes = list(range(75))
    filename = "075_macro"
    search_types = [
    
    #("bfs", f"analyses_results/bfs_search_{filename}.txt"),
    #("dfs", f"analyses_results/dfs_search_{filename}.txt"),
    #("aStar", f"analyses_results/astar_search_{filename}.txt"),
    #("greedy", f"analyses_results/greedy_search_{filename}.txt"),
    ("uniform", f"analyses_results/uniform_search_{filename}.txt")
]
    
    for search_type, file_path in search_types:
        test_solver(file_path, indexes, search_type)

    