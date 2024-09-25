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

    # making warehouse into matrix for easy management
    warehouse_layout = str(warehouse).splitlines()
    rows, cols = len(warehouse_layout), len(warehouse_layout[0])
    matrix = [['']*cols for _ in range(rows)]

    # representing matrix as warehouse
    for r in range(rows):
        for c in range(cols):
            matrix[r][c] = warehouse_layout[r][c]


    # mark the out of bound region as NA
    for r in range(rows):
        l_pointer = 0
        r_pointer = cols - 1
        while l_pointer < cols and matrix[r][l_pointer] != '#':
            if matrix[r][l_pointer] == ' ':
                matrix[r][l_pointer] = 'NA'
            l_pointer += 1

        while r_pointer >= 0 and matrix[r][r_pointer] != '#':
            if matrix[r][r_pointer] == ' ':
                matrix[r][r_pointer] = 'NA'
            r_pointer -= 1 


    # check taboo cell
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] != '.' and matrix[r][c] != 'NA' and matrix[r][c] != '#':
                # top left
                if r > 0 and c > 0 and matrix[r - 1][c] == '#' and matrix[r][c - 1] == '#':
                    matrix[r][c] = 'X'
                # top right
                if r > 0 and c + 1 < cols and matrix[r - 1][c] == '#' and matrix[r][c + 1] == '#':
                    matrix[r][c] = 'X'
                # bottom left
                if r + 1 < rows and c > 0 and matrix[r + 1][c] == '#' and matrix[r][c - 1] == '#':
                    matrix[r][c] = 'X'
                # bottom right
                if r + 1 < rows and c + 1 < cols and matrix[r + 1][c] == '#' and matrix[r][c + 1] == '#':
                    matrix[r][c] = 'X'
                    
            # remove other symbols        
            if matrix[r][c] in '.*@$':
                matrix[r][c] = ' '

            # change NA back to ' '        
            if matrix[r][c] in 'NA':
                matrix[r][c] = ' '


    return '\n'.join([''.join(row) for row in matrix])


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
    
    def __init__(self, warehouse, allow_taboo_push=False, macro=False):
        # define directions for the worker to move
        self.directions = [(-1, 0, 'Up'), (1, 0, 'Down'), (0, -1, 'Left'), (0, 1, 'Right')]

        # note the key state of the warehouse
        self.walls = set()
        self.targets = set()
        self.boxes = set()
        self.worker = tuple()

        # making warehouse a matrix
        self.warehouse = warehouse
        self.matrix = self.create_matrix(warehouse)

        # save the state for type of movement
        self.allow_taboo_push = allow_taboo_push
        self.macro = macro

        # save the original state for the warehouse
        self.initial = (self.worker, frozenset(self.boxes))

    def create_matrix(self, warehouse):
        # turning warehouse into matrix
        warehouse_layout = str(warehouse).splitlines()
        rows, cols = len(warehouse_layout), len(warehouse_layout[0])
        matrix = [[' '] * cols for _ in range(rows)]

        # make note on key areas
        for r in range(rows):
            for c in range(cols):
                cur = warehouse_layout[r][c]
                matrix[r][c] = cur

                if cur == '#':
                    self.walls.add((r, c))
                elif cur == '.':
                    self.targets.add((r, c))
                elif cur == '$':
                    self.boxes.add((r, c))
                elif cur == '*':
                    self.boxes.add((r, c))
                    self.targets.add((r, c))
                elif cur == '@':
                    self.worker = (r, c)
                elif cur == '!':
                    self.worker = (r, c)
                    self.targets.add((r, c))

        return matrix

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        """
        # get current worker position from the state
        ## worker = (r, c); boxes (hashset) = {(r, c), (r, c) ...} -> They are the LATEST state not initial
        worker, boxes = state
        r, c = worker

        # total game map dimensions from length of matrix
        rows, cols = len(self.matrix), len(self.matrix[0])

        # get taboo cell from the function to know if a position is 'X'
        taboo_layout = taboo_cells(self.warehouse).splitlines()

        # legal action for the worker
        possible_actions = []

        # iterate over 4 directions [(-1, 0, 'Up'), (1, 0, 'Down'), (0, -1, 'Left'), (0, 1, 'Right')]
        for dr, dc, move in self.directions:
            # next potential position by adding the grid row and grid col
            ## eg. current = 4, 2; go up one grid = 3, 2
            nr, nc = r + dr, c + dc

            # check if the next position is in bound
            if 0 <= nr < rows and 0 <= nc < cols:

                # if macro is True, only consider box pushing actions
                if self.macro:
                    pass # NOT implemented (Joyce)

                else:
                    # if the next position is a box (efficient lookup from hashset)
                    if (nr, nc) in boxes:
                        # want to push box, so need to identify the 'next next position'
                        nnr, nnc = nr + dr, nc + dc

                        # if the 'next next position' is in game boundary
                        if 0 <= nnr < rows and 0 <= nnc < cols:
                            # if the 'next next position' is a vacant space or target cell
                            if (nnr, nnc) not in boxes and self.matrix[nnr][nnc] in [' ', '.']:
                                # check if that vacant cell might be a taboo from taboo_layout
                                ## if 'taboo_push' is False, cannot push to taboo cell
                                if not self.allow_taboo_push and taboo_layout[nnr][nnc] == 'X':
                                    continue # skip and continue directly from next iteration

                                possible_actions.append(move)

                    # if the next position is a vacant space or the target spot -> move worker
                    elif self.matrix[nr][nc] not in '#*': 
                        possible_actions.append(move)

        return possible_actions

    def result(self, state, action): # now the worker is ABOUT TO perform an action
        # get current worker position from the state
        ## worker = (r, c); boxes (hashset) = {(r, c), (r, c) ...} -> They are the LATEST state not initial
        worker, boxes = state
        r, c = worker

        # iterate over 4 directions [(-1, 0, 'Up'), (1, 0, 'Down'), (0, -1, 'Left'), (0, 1, 'Right')]
        for dr, dc, move in self.directions:
            # if the iterated move == action that the worker is going to perform
            if move == action:
                # next position for the worker (r, c) -> (nr, nc)
                ## eg. current = 4, 2; go up one grid = 3, 2
                nr, nc = r + dr, c + dc 

                # if the new location is a box, worker will push the box
                ## use hashset for efficient look up
                if (nr, nc) in boxes:
                    # find the 'next next position'
                    nnr, nnc = nr + dr, nc + dc

                    # push the box to the new position
                    ## remove old box position
                    ## add new box position
                    new_boxes = set(boxes)
                    new_boxes.remove((nr, nc))
                    new_boxes.add((nnr, nnc))

                    # record and return new state
                    ## worker new position is updated as (nr, nc)
                    return ((nr, nc), frozenset(new_boxes))

                # if worker is moving to a vacant space
                else:
                    # record and return new state
                    ## worker new position is updated as (nr, nc)
                    return ((nr, nc), boxes)
                
        # if action is not legal, should not happen
        raise Exception('Invalid action')

    def goal_test(self, state):
        # get current boxes position
        _, boxes = state

        # check if all boxes are in targets position
        return all(box in self.targets for box in boxes)

    def path_cost(self, c, state1, action, state2):
        # record length of path
        return c + 1



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
    warehouse_layout = str(warehouse).splitlines()
    rows, cols = len(warehouse_layout), len(warehouse_layout[0])
    matrix = [[''] * cols for _ in range(rows)]

    # Create 2D matrix for warehouse
    for r in range(rows):
        for c in range(cols):
            matrix[r][c] = warehouse_layout[r][c]

    # define the initial positions of variables 
    walls, boxes, targets = [], [], []
    worker = ()
    for r in range(rows):
        for c in range(cols):
            # initial position
            if matrix[r][c] == '#':
                walls.append((r,c))
            elif matrix[r][c] == '$':
                boxes.append((r,c))
            elif matrix[r][c] == '.':
                targets.append((r,c))
            elif matrix[r][c] == '*': # box on a target
                targets.append((r,c))  
                boxes.append((r,c))  
            elif matrix[r][c] == '@':
                worker = (r,c)

    # define the directions
    move_dic = {
        'Left': (0, -1),
        'Right': (0, 1),
        'Up': (-1, 0),
        'Down': (1, 0)
    }

    # Check the action in seq is valid or not
    for action in action_seq:
        if action not in move_dic:
            return "Failure"
    
        # worker next move
        worker_row, worker_col = worker
        dic_x, dic_y = move_dic[action]
        next_move = (worker_row + dic_x, worker_col + dic_y)

        # check if next move is the wall
        if next_move in walls:
            return "Failure"
        
        # check if next move is the box
        if next_move in boxes:
            box_next_position = (next_move[0] + dic_x, next_move[1] + dic_y)
            # check the box’s next position is a wall or a box 
            if box_next_position in walls or box_next_position in boxes:
                return "Failure"
            # push the box
            else:
                boxes.remove(next_move)
                boxes.append(box_next_position)
        # update worker's position
        worker = next_move

    # update worker's position in the matrix
    # clear all the related signs
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == '$' or matrix[r][c] == '@' or matrix[r][c] == '*':
                matrix[r][c] = ' '
    # update worker's position
    matrix[worker[0]][worker[1]] = '@'
    # update boxes' and targets' positions
    for box in boxes:
        if box in targets:
            matrix[box[0]][box[1]] = '*'
        else:
            matrix[box[0]][box[1]] = '$'

    return '\n'.join([''.join(row) for row in matrix])

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
    sokoban = SokobanPuzzle(warehouse, macro=False, allow_taboo_push=False)
    if sokoban.goal_test(sokoban.initial):
        return []
    
    solution_node = search.breadth_first_graph_search(sokoban)
    if solution_node is None:
        return 'Impossible'

    return solution_node.solution()

def dfs(matrix, source, target, visited=None):
    '''
    A function that search a matrix of movements, return true if the target is accessible from the origin
    '''
    x, y = source

    if visited is None:
        visited = set()
    if source == target:
        return True
    visited.add(source)
    
    rows = len(matrix)
    cols = len(matrix[0])
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for direction in directions:
        nx, ny = x + direction[0], y + direction[1]
        if 0<=nx <rows and 0 <= ny < cols:
            neighbor = (nx, ny)
            if matrix[nx][ny] == 0 and neighbor not in visited:  
                if dfs(matrix, neighbor, target, visited): 
                    return True
    return False

def can_go_there(warehouse, dst):
    '''    
    Determine whether the worker can walk to the cell dst=(row,column) 
    without pushing any box.
    
    @param warehouse: a valid Warehouse object

    @return
      True if the worker can walk to cell dst=(row,column) without pushing any box
      False otherwise
    '''
    
    ##         "INSERT YOUR CODE HERE"
    y, x = warehouse.worker
    x1, y1 = dst
    target = tuple((x+x1, y+y1))
    x_max = max([x for x, y in warehouse.walls])
    x_min = min([x for x, y in warehouse.walls])

    y_max = max([y for x, y in warehouse.walls])
    y_min = min([y for x, y in warehouse.walls])

    dist = tuple((x_max-x_min, y_max-y_min))
     
    matrix = [[0 for _ in range(dist[0] + 1)] for _ in range(dist[1] + 1)]

    for wall_x, wall_y in warehouse.walls:
        matrix[wall_y - y_min][wall_x - x_min]  = 1 

    for box_x, box_y in warehouse.boxes:
        matrix[box_y - y_min][box_x - x_min] = 1 
    
    return dfs(matrix, tuple((x,y)), target)

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
    
    raise NotImplementedError()

