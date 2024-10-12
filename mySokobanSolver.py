'''
IFN680 Sokoban Assignment

The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

You are not allowed to change the defined interfaces.
That is, changing the formal parameters of a function will break the 
interface and triggers to a fail for the test of your code.
'''

import search
import sokoban
import numpy as np
import time
import math
from search import Node

def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    e.g.  [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
    '''

    return [(11393611, 'Yu-Ying', 'Tang'), (11371200, 'Arthur', 'Guillaume')]
 

def taboo_cells(warehouse):
    '''  
    Identify the taboo cells of a warehouse. A cell inside a warehouse is 
    called 'taboo' if whenever a box get pushed on such a cell then the puzzle 
    becomes unsolvable.  
    When determining the taboo cells, you must ignore all the existing boxes, 
    simply consider the walls and the target cells.  
    Use only the following two rules to determine the taboo cells;
     Rule 1: if a cell is a corner inside the warehouse and not a target, 
             then it is a taboo cell.
     Rule 2: all the cells between two corners inside the warehouse along a 
             wall are taboo if none of these cells is a target.
    
    @param warehouse: a Warehouse object

    @return
       A string representing the puzzle with only the wall cells marked with 
       an '#' and the taboo cells marked with an 'X'.  
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.  
    '''
    ##         "INSERT YOUR CODE HERE"  

    # retrieving warehouse
    warehouse = str(warehouse).split('\n')
    warehouse = [list(row) for row in warehouse]
    rows, cols = len(warehouse), len(warehouse[0])

    # mark out of boundary position as NA
    ## left and right detection
    for r in range(rows):
        # from left
        l_ptr = 0
        while l_ptr < cols and warehouse[r][l_ptr] != '#':
            if warehouse[r][l_ptr] == ' ':
                warehouse[r][l_ptr] = 'NA'
            l_ptr += 1

        # from right
        r_ptr = cols - 1
        while r_ptr >= 0 and warehouse[r][r_ptr] != '#':
            if warehouse[r][r_ptr] == ' ':
                warehouse[r][r_ptr] = 'NA'
            r_ptr -= 1

    ## top and bottom detection
    for c in range(cols):
        # from top
        t_ptr = 0
        while t_ptr < rows and warehouse[t_ptr][c] != '#':
            if warehouse[t_ptr][c] == ' ':
                warehouse[t_ptr][c] = 'NA'
            t_ptr += 1

        # from bottom
        b_ptr = rows - 1
        while b_ptr >= 0 and warehouse[b_ptr][c] != '#':
            if warehouse[b_ptr][c] == ' ':
                warehouse[b_ptr][c] = 'NA'
            b_ptr -= 1

    # check taboo cells
    # rule 1: mark taboo cell at corners
    for r in range(rows):
        for c in range(cols):
            if warehouse[r][c] not in ('.', 'NA', '#'):
                # top-left
                if r > 0 and c > 0 and warehouse[r - 1][c] == '#' and warehouse[r][c - 1] == '#':
                    warehouse[r][c] = 'X'
                # top-right
                if r > 0 and c + 1 < cols and warehouse[r - 1][c] == '#' and warehouse[r][c + 1] == '#':
                    warehouse[r][c] = 'X'
                # bottom-left
                if r + 1 < rows and c > 0 and warehouse[r + 1][c] == '#' and warehouse[r][c - 1] == '#':
                    warehouse[r][c] = 'X'
                # bottom-right
                if r + 1 < rows and c + 1 < cols and warehouse[r + 1][c] == '#' and warehouse[r][c + 1] == '#':
                    warehouse[r][c] = 'X'

    # rule 2: mark taboo cells between two corners along walls
    ## horizontal detection (left to right)
    for r in range(rows):
        left_corner = -1
        for c in range(cols):
            if warehouse[r][c] == 'X':  # Found a corner
                if left_corner == -1:
                    left_corner = c
                else:
                    # check if all cells between the corners are empty spaces and not targets
                    if all(warehouse[r][i] == ' ' for i in range(left_corner + 1, c)) and all(warehouse[r][i] not in '.' for i in range(left_corner + 1, c)):
                        for i in range(left_corner + 1, c):
                            warehouse[r][i] = 'X'
                    left_corner = c

    # vertical detection (top to bottom)
    for c in range(cols):
        top_corner = -1
        for r in range(rows):
            if warehouse[r][c] == 'X':  # Found a corner
                if top_corner == -1:
                    top_corner = r
                else:
                    # check if all cells between the corners are empty spaces and not targets
                    if all(warehouse[i][c] == ' ' for i in range(top_corner + 1, r)) and all(warehouse[i][c] not in '.' for i in range(top_corner + 1, r)):
                        for i in range(top_corner + 1, r):
                            warehouse[i][c] = 'X'
                    top_corner = r

    # remove other symbols and swap 'NA' back to empty spaces
    for r in range(rows):
        for c in range(cols):
            if warehouse[r][c] in '.*@$':
                warehouse[r][c] = ' '
            if warehouse[r][c] == 'NA':
                warehouse[r][c] = ' '

    return '\n'.join([''.join(row) for row in warehouse])

class SokobanPuzzle(search.Problem):
    '''
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.

    Your implementation should be fully compatible with the search functions of 
    the provided module 'search.py'. It uses search.Problem as a sub-class. 
    That means, it should have a:
    - self.actions() function
    - self.result() function
    - self.goal_test() function
    See the Problem class in search.py for more details on these functions.
    
    Each instance should have at least the following attributes:
    - self.allow_taboo_push
    - self.macro
    
    When self.allow_taboo_push is set to True, the 'actions' function should 
    return all possible legal moves including those that move a box on a taboo 
    cell. If self.allow_taboo_push is set to False, those moves should not be
    included in the returned list of actions.
    
    If self.macro is set True, the 'actions' function should return 
    macro actions. If self.macro is set False, the 'actions' function should 
    return elementary actions.
    '''
    
    def __init__(self, warehouse, goal=None, allow_taboo_push=False, macro=False):
        # directions to move in (x, y) -> NOTE: origin at TOP-LEFT
        self.directions = { 'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1) }

        # retrieve key state of the warehouse
        self.walls = set(warehouse.walls)
        self.targets = set(warehouse.targets)
        self.boxes = set(warehouse.boxes)
        self.worker = warehouse.worker
        self.goal = goal
        
        # get maximum game boundaries
        ## compute number of rows and columns from the layout by joining sets and get maximum
        all_positions = self.walls | self.boxes | self.targets | {self.worker} # output: {(2,4), (5,3), (1,0), ...}
        self.nys = max(y for _, y in all_positions) + 1 # output: {(_,4), (_,3),(_,0)} --> 4
        self.nxs = max(x for x, _ in all_positions) + 1 # output: {(2,_), (5,_),(1,_)} --> 5

        # get taboo cell from the function to know if a position is 'X'
        self.taboo_layout = taboo_cells(warehouse).splitlines()
        #need taboo cell for the lookup in solve_macro
        self.taboo_cells = set(sokoban.find_2D_iterator(self.taboo_layout , "X"))

        # save the state for type of movement
        self.allow_taboo_push = allow_taboo_push
        #need the warehouse for the can_go_there, at the moment
        self.warehouse = warehouse
        self.macro = macro

        # save the original state for the warehouse
        super().__init__((self.worker, frozenset(self.boxes)))

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        """
        # get CURRENT worker position from the state
        # worker = (x, y); boxes (hashset) = {(x, y), (x, y) ...} -> They are the LATEST state not initial
        worker, boxes = state
        x, y = worker

        # legal action for the worker
        possible_actions = []
        # iterate over 4 directions { 'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1) }
        for move, coordinates in self.directions.items():
            dx, dy = coordinates
            # next potential position by adding the updating (x, y) position
            # eg. current = (4, 2) ; go UP (0,-1) --> new= (4, 1)
            nx, ny = x + dx, y + dy

            # check if the next position is in bound
            if 0 <= nx < self.nxs and 0 <= ny < self.nys:
                # if macro is True, only consider box pushing actions
                if self.macro:
                    for box in boxes:
                        # position of the worker next to the box
                        worker_pos = (box[0] - coordinates[0], box[1] - coordinates[1])
                        # check it the move is valid, saves time
                        new_box_pos = (box[0] + coordinates[0], box[1] + coordinates[1])
                        # cannot push a box on a wall or an other box
                        if new_box_pos in self.walls.union(boxes):
                            continue
                        # if allow_taboo push is false
                        if new_box_pos in self.taboo_cells and not self.allow_taboo_push:
                            continue
                        
                        tem_warehouse = self.warehouse.copy(worker=worker, boxes=boxes)
                        # check if the worker can go there, targer is y, x so needs to be inverted
                        if can_go_there(tem_warehouse, (worker_pos[1], worker_pos[0])):
                                possible_actions.append(((box[1], box[0]), move))
            
                else:
                    # if the next position is a box (efficient lookup from hashset)
                    if (nx, ny) in boxes:
                        # want to push box, so need to identify the 'box's next position'
                        nnx, nny = nx + dx, ny + dy

                        # if the 'box's next position' is in game boundary
                        if 0 <= nnx < self.nxs and 0 <= nny < self.nys:
                            # if the 'box's next position' is a vacant space or target cell
                            if (nnx, nny) not in boxes and (nnx, nny) not in self.walls:
                                # check if that vacant cell might be a taboo from taboo_layout
                                # if 'taboo_push' is False, cannot push to taboo cell
                                if not self.allow_taboo_push and self.taboo_layout[nny][nnx] == 'X': 
                                    continue  # skip and continue directly from next iteration

                                possible_actions.append(move) # add valid move to the possible action list

                    # if the next position is a vacant space or the target spot -> move worker
                    elif (nx, ny) not in self.walls:
                        possible_actions.append(move) # add valid move to the possible action list

        return possible_actions

    def result(self, state, action): # now worker is ABOUT to perform an action
        # get current worker position from the state
        worker, boxes = state
        x, y = worker

        if self.macro:
            box, direction = action
            box = (box[1], box[0])  # Convert to (x, y) format
            dx, dy = self.directions[direction]
            new_box = (box[0] + dx, box[1] + dy)
            new_boxes = set(boxes)
            new_boxes.remove(box)
            new_boxes.add(new_box)
            new_worker = box
            return (new_worker, frozenset(new_boxes))
        
        else:
            # directions = { 'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1) }
            dx, dy = self.directions[action]

            # next position for the worker (x, y) -> (nx, ny)
            nx, ny = x + dx, y + dy 

            # if the new location is a box, worker will push the box
            if (nx, ny) in boxes:
                # find the 'box's next position'
                nnx, nny = nx + dx, ny + dy

                # push the box to the new position
                new_boxes = set(boxes)
                new_boxes.remove((nx, ny))
                new_boxes.add((nnx, nny))

                # return new state with updated worker and boxes positions
                return ((nx, ny), frozenset(new_boxes))

            # if worker is moving to a vacant space
            else:
                # return new state with updated worker position
                return ((nx, ny), boxes)

    def goal_test(self, state):
        # get current boxes position
        _, boxes = state
        
        # check if all boxes are in target positions
        return all(box in self.targets for box in boxes)

    def path_cost(self, c, state1, action, state2):
        # record length of path
        return c + 1
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def euclidean_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def h(self, node):
        """
        Heuristic function for A* search.
        Estimates the cost(distance) from the current state to the goal state.
        Evaluates states: It assigns a value to each state in the search space, indicating how "promising" that state is.
        Guides search: It helps the search algorithm decide which states to explore next.
        """
        _, boxes = node.state
        boxes = set(boxes) 
        remaining_targets = self.targets - boxes 

        if not remaining_targets:
            return 0  # All boxes are on targets

        total_cost = 0 
        for box in boxes - self.targets: 
            # Find the minimum distance from the box to any target 
            min_dist = min(self.manhattan_distance(box, target) for target in remaining_targets) 
            total_cost += min_dist 
        return total_cost
        
        # _, boxes = node.state
        # boxes = set(boxes)
        
        # distance_matrix = []
        # for box in boxes:
        #     box_distances = [self.manhattan_distance(box, target) for target in self.targets]
        #     distance_matrix.append(box_distances)
        
        # # Use the Hungarian algorithm to find the minimum cost matching
        # from scipy.optimize import linear_sum_assignment
        # row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        # total_cost = sum(distance_matrix[i][j] for i, j in zip(row_ind, col_ind))
        
        # return total_cost
    
    def verify_consistency(self):
        # the initial state of the problem
        initial_state = self.initial
        # the frontier with the initial state and its cost (0)
        frontier = [(initial_state, 0)]
        # keep track of explored states
        explored = set()

        while frontier:
            # pops the first state and its associated cost
            state, cost = frontier.pop(0)
            if state not in explored:
                # adds it to the explored set
                explored.add(state)
                # the heuristic value of current state and the goal state
                h_state = self.h(Node(state))

                # loop possible actions
                for action in self.actions(state):
                    # each action, generates the child state
                    child = self.result(state, action)
                    # the heuristic value of the child state
                    h_child = self.h(Node(child))
                    # the step cost from the current state to the child state
                    step_cost = self.path_cost(0, state, action, child)

                    # checks the consistency condition (h(current state)-h(child state) <= cost(current state to child state))
                    if h_state > step_cost + h_child:
                        print(f"Inconsistency found:")
                        print(f"State: {state}, h(state): {h_state}")
                        print(f"Child: {child}, h(child): {h_child}")
                        print(f"Step cost: {step_cost}")
                        return False
                    # no inconsistency is found, the child state and its cumulative cost is added to the frontier for further exploration
                    frontier.append((child, cost + step_cost))

        print("Heuristic is consistent")
        return True

    def verify_admissibility(self):
        # calculate the true cost from current state to the goal state (including all necessary moves - player movements between box pushes)
        def true_cost(state):
            temp_puzzle = SokobanPuzzle(self.warehouse)
            temp_puzzle.initial = state
            solution = search.astar_graph_search(temp_puzzle, self.h)  # Use zero heuristic for true cost
            return len(solution.solution()) if solution else float('inf')
        curr_state = self.initial
        h_value = self.h(Node(curr_state))
        true_value = true_cost(curr_state)
        if h_value > true_value:
                print(f"Inadmissibility found:")
                print(f"Current State: {curr_state}")
                print(f"Heuristic Value: {h_value}")
                print(f"True cost: {true_value}")
                return False   
        else:
            print(f"Heuristic is admissible")
            return True
    

def check_action_seq(warehouse, action_seq):
    '''
    Determine if the sequence of actions listed in 'action_seq' is legal or not.
    
    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.
        
    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
           
    @return
        The string 'Failure', if one of the action was not successul.
           For example, if the agent tries to push two boxes at the same time,
                        or push one box into a wall, or walk into a wall.
        Otherwise, if all actions were successful, return                 
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
    '''
    ##         "INSERT YOUR CODE HERE"

    # retrieve states from the warehouse
    walls = set(warehouse.walls)
    targets = set(warehouse.targets)
    boxes = set(warehouse.boxes)
    worker = warehouse.worker 

    # directions to move in (x, y) -> NOTE: origin at TOP-LEFT
    directions = { 'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1) }
    
    for action in action_seq:
        if action not in directions:
            return 'Failure'

        # get worker position
        x, y = worker

        # next potential position
        dx, dy = directions[action]
        nx, ny = x + dx, y + dy

        # if next position is not reachable (a wall or a box already on a target)
        if (nx, ny) in walls:
            return 'Failure'

        # if next position is a box
        if (nx, ny) in boxes:
            # need to push box to box'x next position
            nnx, nny = nx + dx, ny + dy

            # check if box's next position is reachable (can't push into walls or other boxes)
            if (nnx, nny) in walls or (nnx, nny) in boxes:
                return 'Failure'

            # move the box to the box's next position
            boxes.remove((nx, ny))
            boxes.add((nnx, nny))

        # update worker position to the next position
        worker = (nx, ny)
    
    # compute number of rows and columns from the layout
    all_positions = walls | boxes | targets | {worker} # output: {(2,4), (5,3), (1,0), ...}
    nys = max(y for _, y in all_positions) + 1 # output: {(_,4), (_,3),(_,0)} --> 4
    nxs = max(x for x, _ in all_positions) + 1 # output: {(2,_), (5,_),(1,_)} --> 5

    # construct new warehouse layout
    new_warehouse = []
    for y in range(nys):
        line = []
        for x in range(nxs):
            pos = (x, y)  # x and y coordinates with origin at TOP-LEFT
            if pos in walls:
                line.append('#')
            elif pos in boxes and pos in targets:
                line.append('*')  # box on target
            elif pos in boxes:
                line.append('$')  # box
            elif pos in targets:
                line.append('.')  # target
            elif pos == worker:
                line.append('@' if pos not in targets else '!')  # worker or worker on target
            else:
                line.append(' ')  # free space
        new_warehouse.append(''.join(line))
    
    return '\n'.join(new_warehouse)

def solve_sokoban_elem(warehouse):
    '''    
    This function should solve using elementary actions 
    the puzzle defined in a file.
    
    @param warehouse: a valid Warehouse object

    @return
        If puzzle cannot be solved return the string 'Impossible'
        If a solution was found, return a list of elementary actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
    '''

    ##         "INSERT YOUR CODE HERE"
    start_time = time.time()
    timeout = 180 # 3 mins
    
    sokoban = SokobanPuzzle(warehouse, macro=False, allow_taboo_push=False)
    is_consistent = sokoban.verify_consistency()
    is_admissible = sokoban.verify_admissibility()
    print(f"Consistent: {is_consistent}; Admissible: {is_admissible}")
    
    if sokoban.goal_test(sokoban.initial):
        return []
    
    # use breadth-first search as default
    solution_node = search.breadth_first_graph_search(sokoban)

    # check for timeout
    if time.time() - start_time > timeout:
        return "Timeout"
    
    if solution_node is None:
        return 'Impossible'
    
    # Calculate total cost using path_cost
    total_cost = 0
    # get the full path of nodes from the initial state to the goal state
    path = solution_node.path()
    # start from 1: each node is compared to its previous node
    for i in range(1, len(path)):
        total_cost = sokoban.path_cost(total_cost, path[i-1].state, path[i].action, path[i].state)
    print('elem total cost:',total_cost)
    
    return solution_node.solution()

def can_go_there(warehouse, dst):
    '''    
    Determine whether the worker can walk to the cell dst=(row,column) 
    switch because the row is y and column i s x
    without pushing any box.

    overRide the Sokoban Solver, changes the goal and the goal_test
    
    @param warehouse: a valid Warehouse object

    @return
      True if the worker can walk to cell dst=(row,column) without pushing any box
      False otherwise
    '''
   
    ##         "INSERT YOUR CODE HERE"
    goal = (dst[1], dst[0])
    sokoban = SokobanPuzzle(warehouse, goal=goal, allow_taboo_push=True, macro=False)      

    # Override the goal_test method for this specific use case
    sokoban.goal_test = lambda state: state[0] == goal
    
    # Override the actions method
    sokoban.actions = lambda state: [
        action for action, (dx, dy) in sokoban.directions.items()
        if (state[0][0] + dx, state[0][1] + dy) not in state[1]
        and (state[0][0] + dx, state[0][1] + dy) not in sokoban.walls
    ]
    
    # Use breadth-first search 
    solution = search.breadth_first_graph_search(sokoban)
    
    return solution is not None
            
def solve_sokoban_macro(warehouse):
    '''    
    Solve using macro actions the puzzle defined in the warehouse passed as
    a parameter. A sequence of macro actions should be 
    represented by a list M of the form
            [ ((r1,c1), a1), ((r2,c2), a2), ..., ((rn,cn), an) ]
    For example M = [ ((3,4),'Left') , ((5,2),'Up'), ((12,4),'Down') ] 
    means that the worker first goes the box at row 3 and column 4 and pushes it left,
    then goes to the box at row 5 and column 2 and pushes it up, and finally
    goes the box at row 12 and column 4 and pushes it down.
    
    @param warehouse: a valid Warehouse object

    @return
        If puzzle cannot be solved return the string 'Impossible'
        Otherwise return M a sequence of macro actions that solves the puzzle.
        If the puzzle is already in a goal state, simply return []
    '''
     
    
    ##         "INSERT YOUR CODE HERE"
    start_time = time.time()
    timeout = 180 # 3 mins
    
    sokoban = SokobanPuzzle(warehouse, allow_taboo_push=False, macro=True) 
    is_consistent = sokoban.verify_consistency()
    is_admissible = sokoban.verify_admissibility()
    print(f"Consistent: {is_consistent}; Admissible: {is_admissible}")
     
    if sokoban.goal_test(sokoban.initial):
        return [] 
    
    try:

        # use breadth-first search as default
        solution_node = search.breadth_first_graph_search(sokoban)

        # check timeout
        if time.time() - start_time > timeout:
                return "Timeout"
        
        # Calculate total cost using path_cost
        total_cost = 0
        # get the full path of nodes from the initial state to the goal state
        path = solution_node.path()
        # start from 1: each node is compared to its previous node
        for i in range(1, len(path)):
            total_cost = sokoban.path_cost(total_cost, path[i-1].state, path[i].action, path[i].state)
        print('macro total cost:',total_cost)
    
        if solution_node is not None:
            return solution_node.solution()
        else:
            return "Impossible"
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Impossible"


