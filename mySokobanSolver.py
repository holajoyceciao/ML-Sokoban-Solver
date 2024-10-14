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
import signal

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
    
    def __init__(self, warehouse, goal=None, allow_taboo_push=False, macro=False, allow_push=True):
        # directions to move in (x, y) -> NOTE: origin at TOP-LEFT
        self.directions = { 'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1) }

        # retrieve key state of the warehouse
        self.walls = set(warehouse.walls)
        self.targets = set(warehouse.targets)
        self.boxes = set(warehouse.boxes)
        self.worker = warehouse.worker
        self.allow_push = allow_push
        
   
        
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
        self.goal = goal
        

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
                        # check it the move is valid, saves time
                        new_box_pos = (box[0] + coordinates[0], box[1] + coordinates[1])#changed order of checks
                        # cannot push a box on a wall or an other box
                        if new_box_pos in self.walls.union(boxes):
                            continue
                        # if allow_taboo push is false and the box would be on a taboo cell, skip iteration
                        if new_box_pos in self.taboo_cells and not self.allow_taboo_push:#
                            continue
                        # position of the worker next to the box
                        worker_pos = (box[0] - coordinates[0], box[1] - coordinates[1])
                        
                        
                        
                        temp_warehouse = self.warehouse.copy(worker=worker, boxes=boxes)
                        # check if the worker can go there, targer is y, x so needs to be inverted
                        if can_go_there(temp_warehouse, (worker_pos[1], worker_pos[0])):
                                possible_actions.append(((box[1], box[0]), move))
            
                else:
                    if (nx, ny) in self.walls:
                        continue
                    # if the next position is a box (efficient lookup from hashset)
                    if (nx, ny) in boxes:
                        if not self.allow_push:
                            continue
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
        worker, boxes = state
        
        if self.goal is None:
            # check if all boxes are in target positions
            return all(box in self.targets for box in boxes)
        else:
           
             #if there is a goal, for the can_go_there, instead look if the worker is on on the goal
            return worker == self.goal

    def path_cost(self, c, state1, action, state2):
        
        return c+1
    def print_solution(self, goal_node):

        path = goal_node.path()

        print( f"Solution takes {len(path)-1} steps from the initial state to the goal state\n")
        print( "Below is the sequence of moves\n")
        moves = []
        for node in path:
            if node.action:
                moves.append(node.action)
        return(moves)
    '''
    def path_cost_test(self, c, state1, action, state2):
        # record length of path
        """
        return the cost of going from state1 to state2 via an action
        the deafult cost of a move is 1
        we will add a higher cost to moving a box: 2
        we will add an higher cost to moving a box out of a target: 3
        """
        action_cost = c +1
        #default move cost is 1
        
        worker_1, boxes_1 = state1
        worker_2, boxes_2 = state2
        #check if a box has changed position
        box_pushed = boxes_1 != boxes_2
        previous_position = None
        next_position = None
        if box_pushed:
        #find the box
            action_cost +=1
            for box in boxes_1:
                if box not in boxes_2:
                    previous_position = box

        if previous_position is not None:
         #find the new position
            for box in boxes_2:
                if box not in boxes_1:
                    next_position = box
            if previous_position in self.targets and next_position not in self.targets:
                #check if the box has been move out of a target
                action_cost += 1
        return action_cost

    '''
    
    def euclidean_distance(self, box):

        box_x, box_y = box

        return np.sqrt((box_x - self.goal[0])**2+ (box_y - self.goal[1])**2)   

    def manhattan_distance(self, box):
    # simple heuristic used for the macro_search, uses the min distance betwenn each box and target as a cost
        box_x, box_y = box
       
        min_box_goal_distance = min(
            abs(box_x - goal_x) + abs(box_y - goal_y) for goal_x, goal_y in self.targets
        )
            
        return min_box_goal_distance

    def h_ontarget(self, node):
        '''
        INPUT: a problem node
        OUTPUT: the number of box not in target.
        '''
        count = 0

        state = node.state

        for box in state[1]:
            if box not in self.target:
                count += 1
        return count

    def h_2(self, node):
        """
        heuristic for the can go there function
        """

        worker, _ = node.state
        player_x, player_y = worker
        # using Manhattan Distance
        distance =abs(player_x - self.goal[0]) + abs(player_y - self.goal[1])  
        return distance
    

    def h(self, node):
        """
        Heuristic function for A* search.
        Estimates the cost from the current state to the goal state.
        """
        _, boxes = node.state

        boxes_cost = 0

        for box_x, box_y in boxes:
    # Cost of moving each box to the nearest goal
            min_box_goal_distance = min(
                abs(box_x - goal_x) + abs(box_y - goal_y) for goal_x, goal_y in self.targets
            )
            boxes_cost += min_box_goal_distance

        return boxes_cost



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


class TimeoutException(Exception):
    pass

# Define the signal handler
def timeout_handler(signum, frame):
    raise TimeoutException("Computation took too long!")



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
    signal.signal(signal.SIGALRM, timeout_handler)

    timeout = 500 # 3 mins
    
    signal.alarm(timeout)  # Set the alarm for 5 minutes
    start_time = time.time()
   
    try:
        sokoban = SokobanPuzzle(warehouse, macro=False, allow_taboo_push=False)
        
        if sokoban.goal_test(sokoban.initial):
            return []
        
        search_type = 'dasfd'

        # use breadth-first search
        if search_type == 'bfs':
            solution_node = search.breadth_first_graph_search(sokoban)
        # use A* search
        else:
            solution_node = search.astar_graph_search(sokoban, sokoban.h)

        if solution_node is None:
            return 'Impossible'
        
        return solution_node.solution()
        
    except TimeoutException:
        return "Timeout"
    
    finally:
        # Cancel the alarm after completion or timeout
        signal.alarm(0)

    

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
    can_go_sokoban = SokobanPuzzle(warehouse, goal=goal, allow_push=False)
    sol_ts = search.best_first_graph_search(can_go_sokoban, can_go_sokoban.h_2)
    if sol_ts is not None:
        return True
    else:
        return False
            
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
    sokoban = SokobanPuzzle(warehouse, allow_taboo_push=False, macro=True) 

   

    # use breadth-first search as default
    solution_node = search.astar_graph_search(sokoban, sokoban.h)

 
    
    if solution_node is None:
        return "Impossible"
    
    # Calculate total cost using path_cost
    total_cost = 0
    # get the full path of nodes from the initial state to the goal state
    path = solution_node.path()
    # start from 1: each node is compared to its previous node
    for i in range(1, len(path)):
        total_cost = sokoban.path_cost(total_cost, path[i-1].state, path[i].action, path[i].state)
    print('macro total cost:',total_cost)

    return solution_node.solution()


