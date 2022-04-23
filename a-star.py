# Kumara_Ritvik.py
"""
Authors: Kumara Ritvik Oruganti (okritvik@umd.edu), 2022
         Adarsh Malapaka (amalapak@umd.edu), 2022
Brief: Computes and visualizes an optimal path between the start and goal posiitons in a 5-connected (-60,-30, 0, 30, 60) degrees
       obstacle map for a point mobile robot with non-zero radius and clearance and a defined step size using A* Algorithm. 
Course: ENPM661 - Planning for Autonomous Robotics [Project-03, Phase 01]
        University of Maryland, College Park (MD)
Date: 21st March, 2022
"""
# Importing the required libraries
import time
import cv2
import numpy as np
import heapq as hq


def take_robot_inputs():
    """
    Gets the robot radius and clearance inputs from the user.
                
    Parameters:
        None
    Returns:
        clearance: int
                Robot's clearance
    """
    clearance = 0
    robot_radius = 0
    left_wheel_vel = 0
    right_wheel_vel = 0

    while True:
        clearance = input("Enter the robot's clearance: ")
        if(int(clearance)<0):
            print("Enter a valid Robot Clearance!")
        else:
            break
    while True:
        left_wheel_vel = input("Enter Left Wheel Velocity in RPM: ")
        if(int(left_wheel_vel)<0):
            print("Enter a valid input velocity")
        else:
            break
    while True:
        right_wheel_vel = input("Enter Right Wheel Velocity in RPM: ")
        if(int(right_wheel_vel)<0):
            print("Enter a valid input velocity")
        else:
            break
    
    return int(clearance), int(robot_radius), int(left_wheel_vel), int(right_wheel_vel)

def take_map_inputs(canvas):
    """
    Gets the initial node, final node coordinates, heading angles and step-size from the user.
                
    Parameters:
        canvas: NumPy array
                Map matrix
    Returns:
        initial_state: List
                List to hold the initial node coordinates and heading angle
        final_state: List 
                List to hold the final node coordinates and heading angle
    """
    initial_state = []
    final_state = []
    initial_angle = 0

    while True:
        while True:
            state = input("Enter the X Coordinate of the Start Node: ")
            if(int(state)<0 or int(state)>canvas.shape[1]-1):
                print("Enter a valid X Coordinate!")
                continue
            else:
                initial_state.append(int(state))
                break
        while True:
            state = input("Enter the Y Coordinate of the Start Node: ")
            if(int(state)<0 or int(state)>canvas.shape[0]-1):
                print("Enter a valid Y Coordinate!")
                continue
            else:
                initial_state.append(int(state))
                break
        
        if(canvas[canvas.shape[0]-1 - initial_state[1]][initial_state[0]][0]==255):
            print("*** The entered start node is in the Obstacle Space! ***")
            initial_state.clear()
        else:
            break

    while True:
        while True:
            state = input("Enter the X Coordinate of the Goal Node: ")
            if(int(state)<0 or int(state)>canvas.shape[1]-1):
                print("Enter a valid X Coordinate!")
                continue
            else:
                final_state.append(int(state))
                break
        while True:
            state = input("Enter the Y Coordinate of the Goal Node: ")
            if(int(state)<0 or int(state)>canvas.shape[0]-1):
                print("Enter a valid Y Coordinate!")
                continue
            else:
                final_state.append(int(state))
            break

        if(canvas[canvas.shape[0]-1 - final_state[1]][final_state[0]][0]==255):
            print("*** The entered goal node is in the obstacle space! ***")
            final_state.clear()
        else:
            break
    while True:
        initial_angle = input("Enter the Initial Head Angle: ")
        
        # if(int(initial_angle)<0 or int(initial_angle)>359 or (int(initial_angle)%30 != 0)):
        if int(initial_angle) < 0:
            initial_angle = int(initial_angle)%360
            initial_angle = 360 + int(initial_angle)
        else:
            initial_angle = int(initial_angle)%360

        initial_state.append(int(initial_angle))
        break
            
    
    return initial_state,final_state


def draw_obstacles(canvas, offset=10):
    """
    Draws the obstacles and walls in the map incorporating the robot's offset.
                
    Parameters:
        canvas: NumPy array
                Map matrix
        offset: int
                Offset is robot radius + clearance 
    Returns:
        canvas: NumPy array
                Map matrix with drawn obstacles
    """
    height, width, __ = canvas.shape

    for i in range(width):
        for j in range(height):
            if(i<=offset) or (i>=(1000-offset)) or (j<=offset) or (j>=(1000-offset)):
                canvas[j][i] = [255,0,0]

            # Drawing scaled obstacles with robot clearance and radius
            # if ((i-300)**2+(j-65)**2-((40+offset)**2))<=0:
            #     canvas[j][i] = [255,0,0]
            
            # if (j+(0.57*i)-(224.285-offset*1.151))>=0 and (j-(0.57*i)+(4.285+offset*1.151))>=0 and (i-(235+offset))<=0 and (j+(0.57*i)-(304.285+offset*1.151))<=0 and (j-(0.57*i)-(75.714+offset*1.151))<=0 and (i-(165-offset))>=0:
            #     canvas[j][i] = [255,0,0]

            # if ((j+(0.316*i)-(76.392-offset*1.048)>=0) and (j+(0.857*i)-(138.571+offset*1.317)<=0) and (j-(0.114*i)-60.909)<=0) or ((j-(3.2*i)+(186+offset*3.352)>=0) and (j-(1.232*i)-(20.652+offset*1.586))<=0 and (j-(0.114*i)-60.909)>=0):
            #     canvas[j][i] = [255,0,0]

            # # Drawing actual obstacles without robot clearance and radius
            # if ((i-300)**2+(j-65)**2-((40)**2))<=0:
            #     canvas[j][i] = [255,255,0]
            
            # if (j+(0.57*i)-(224.285))>=0 and (j-(0.57*i)+(4.285))>=0 and (i-(235))<=0 and (j+(0.57*i)-(304.285))<=0 and (j-(0.57*i)-(75.714))<=0 and (i-(165))>=0:
            #     canvas[j][i] = [255,255,0]

            # if ((j+(0.316*i)-(76.392)>=0) and (j+(0.857*i)-(138.571)<=0) and (j-(0.114*i)-60.909)<=0) or ((j-(3.2*i)+(186)>=0) and (j-(1.232*i)-(20.652))<=0 and (j-(0.114*i)-60.909)>=0):
            #     canvas[j][i] = [255,255,0]
            
            #Offset should be given as cm in input
            if ((i-200)**2+(j-800)**2-((offset+100)**2))<=0:
                canvas[j][i] = [255,255,0]

            if ((i-200)**2+(j-200)**2-((offset+100)**2))<=0:
                canvas[j][i] = [255,255,0]

            if (j<=(1000-425)+offset) and (j>=(1000-575)-offset) and (i>=25-offset) and (i<=175+offset):
                canvas[j][i] = [255,255,0]

            if (j<=(1000-425)+offset) and (j>=(1000-575)-offset) and (i>=375-offset) and (i<=625+offset):
                canvas[j][i] = [255,255,0]
                
            if (j<=(1000-200)+offset) and (j>=(1000-400)-offset) and (i>=725-offset) and (i<=875+offset):
                canvas[j][i] = [255,255,0]
    return canvas


def threshold(num):
    """
    Rounds the given number to the nearest 0.5 value.
    For ex: 4.61 is rounded to 4.5 whereas 4.8 is rounded to 5.0.
                
    Parameters:
        num: float
                Number to be rounded
    Returns:
        num: float
                Number rounded to nearest 0.5
    """
    return round(num*2)/2


def check_goal(node, final):
    """
    Checks if the given current node is within the goal node's threshold distance of 1.5.
                
    Parameters:
        node: List
                Current node
        final: List
                Goal node 
    Returns:
        flag: bool
                True if the present node is the goal node, False otherwise
    """
    if(np.sqrt(np.power(node[0]-final[0],2)+np.power(node[1]-final[1],2))<15):
        return True
    else:
        return False


def cost_to_goal(node, final):
    """
    Computes the Cost To Goal between present and goal nodes using a Euclidean distance heuristic.
                
    Parameters:
        node: List
                Current node
        final: List
                Goal node 
    Returns:
        flag: bool
                True if the present node is the goal node, False otherwise
    """
    return np.sqrt(np.power(node[0]-final[0],2)+np.power(node[1]-final[1],2))


def check_obstacle(next_width, next_height, canvas):    
    """
    Checks if the generated/next node is in the obstacle region.
              
    Parameters:
        next_width: float
                Width of the next node from the present node
        next_height: float
                Height of the next node from the present node
        
        canvas: NumPy array
                Map matrix with drawn obstacles 
    Returns:
        flag: bool
                True if the next node is NOT in the obstacle region, False otherwise
    """
    if int(round(next_height))<0 or int(round(next_width))<0 or int(round(next_height))>=canvas.shape[0] or int(round(next_width))>=canvas.shape[1] or canvas[int(round(next_height))][int(round(next_width))][0]==255:
        return False
    else:
        return True


def action(node, canvas, vel_l, vel_r, R, L):    # Local angles
    """
    Moves the robot at 0 degree angle (wrt robot's frame) by the step amount. 
              
    Parameters:
        node: List
                List of node's x, y and theta parameters
        
        canvas: NumPy array
                Map matrix with drawn obstacles 
        visited: NumPy array
                Visited matrix of size 500x800x12 to keep track of duplicate nodes  
        step: int
               Step size of the robot 
    Returns:
        Next Node flag: bool
                True if the child node can be generated, False otherwise
        next_node: List
                Child node generated after performing the action
        Duplicate Node flag: bool
                True if generated next node is already visited, False otherwise
    """
    next_node = node.copy()
    
    # Angle in Cartesian System
    # print("In action Generation")

    trajectory_list = []
    
    
    dt = 0.1

    x_init = next_node[0]
    y_init = next_node[1]
    theta_init = next_node[2]
    trajectory_list.append((x_init,y_init,theta_init))
    #print("Initial Vals: ",x_init,y_init)
    cost = 0
    for i in np.arange(0, 8, dt):
        x = x_init + (0.5*R*(vel_l+vel_r)*np.cos(np.deg2rad(next_node[2])))*dt
        y = y_init + (0.5*R*(vel_l+vel_r)*np.sin(np.deg2rad(next_node[2])))*dt
        theta = theta_init + ((R/L)*(vel_r - vel_l))*dt
        cost += np.abs(x-x_init) + np.abs(y-y_init)

        #Check if in obstacle space
        #print(x_init,y_init)
        if check_obstacle(x,y,canvas):
            trajectory_list.append((x,y,theta))
        else:
            # print("Returning False")
            return False, trajectory_list, next_node, cost

        x_init, y_init, theta_init = x, y, theta

    next_node[0] = int(round(x_init))
    next_node[1] = int(round(y_init))
    next_node[2] = theta
    return True, trajectory_list, next_node, cost


def astar(initial_state, final_state, canvas,vel_l, vel_r, R, L):
    """
    Implements the A* algorithm to find the path between the user-given start node and goal node.  
    It is robust enough to raise a 'no solution' prompt for goal/start states in the obstacle space.
    The open list is a heap queue which uses the Total Cost as the key to sort the heap.
    The closed list is a dictionary with key as the current node and value as the parent node.
    
    Parameters:
        initial_state: List
                List of start node's x, y and theta parameters
        
        final_state: List
                List of goal node's x, y and theta parameters 
        canvas: NumPy array
                Map matrix with drawn obstacles  
        step: int
               Step size of the robot 
    Returns:
            None
    """
    open_list = []    # Format: {(TotalCost): CostToCome, CostToGo, PresentNode, ParentNode}
    closed_list = {}    # Format: {(PresentNode): ParentNode}
    back_track_flag = False
    trajectory_dict = {}
    # visited_nodes = np.zeros((500,800,12))
    
    hq.heapify(open_list)
    present_c2c = 0
    present_c2g = cost_to_goal(initial_state,final_state)
    total_cost = present_c2c + present_c2g
    hq.heappush(open_list,[total_cost,present_c2c,present_c2g,initial_state,initial_state]) #parent and present
    
    while len(open_list)!=0:
        node = hq.heappop(open_list)
        # print("\nPopped node: ",node)
        closed_list[tuple(node[4])] = node[3]
        if(check_goal(node[4],final_state)):
            print("\nGoal Reached!")
            back_track_flag = True
            back_track(initial_state,node[4],closed_list,canvas,trajectory_dict)
            break

        present_c2c = node[1]
        present_c2g = node[2]
        total_cost = node[0]
        print(node[4])
        flag, traj_list, n_state, cost = action(node[4], canvas, vel_l, vel_l, R, L)
        if(flag):
            if tuple(n_state) not in closed_list:
                dup = False
                for i in range(len(open_list)):
                    if (open_list[i][4][0], open_list[i][4][1]) == (n_state[0],n_state[1]):
                        dup = True
                        n_cost = present_c2c+cost+cost_to_goal(n_state,final_state)
                        if(n_cost<open_list[i][0]):    # Updating the cost and parent info of the node
                            open_list[i][1] = present_c2c+cost
                            open_list[i][0] = n_cost
                            open_list[i][3] = node[4]
                            trajectory_dict[tuple(n_state)] = traj_list
                            hq.heapify(open_list)
                        break
                if(not dup):
                    hq.heappush(open_list,[present_c2c+cost+cost_to_goal(n_state,final_state),present_c2c+cost,cost_to_goal(n_state,final_state),node[4],n_state])
                    hq.heapify(open_list)
                    trajectory_dict[tuple(n_state)] = traj_list

        flag, traj_list, n_state, cost = action(node[4], canvas, vel_l, vel_r, R, L)
        if(flag):
            if tuple(n_state) not in closed_list:
                dup = False
                for i in range(len(open_list)):
                    if (open_list[i][4][0], open_list[i][4][1]) == (n_state[0],n_state[1]):
                        dup = True
                        n_cost = present_c2c+cost+cost_to_goal(n_state,final_state)
                        if(n_cost<open_list[i][0]):    # Updating the cost and parent info of the node
                            open_list[i][1] = present_c2c+cost
                            open_list[i][0] = n_cost
                            open_list[i][3] = node[4]
                            trajectory_dict[tuple(n_state)] = traj_list
                            hq.heapify(open_list)
                        break
                if(not dup):
                    hq.heappush(open_list,[present_c2c+cost+cost_to_goal(n_state,final_state),present_c2c+cost,cost_to_goal(n_state,final_state),node[4],n_state])
                    hq.heapify(open_list)
                    trajectory_dict[tuple(n_state)] = traj_list

        flag, traj_list, n_state, cost = action(node[4], canvas, vel_l, 0, R, L)
        if(flag):
            if tuple(n_state) not in closed_list:
                dup = False
                for i in range(len(open_list)):
                    if (open_list[i][4][0], open_list[i][4][1]) == (n_state[0], n_state[1]):
                        dup = True
                        n_cost = present_c2c+cost+cost_to_goal(n_state,final_state)
                        if(n_cost<open_list[i][0]):    # Updating the cost and parent info of the node
                            open_list[i][1] = present_c2c+cost
                            open_list[i][0] = n_cost
                            open_list[i][3] = node[4]
                            trajectory_dict[tuple(n_state)] = traj_list
                            hq.heapify(open_list)
                        break
                if(not dup):
                    hq.heappush(open_list,[present_c2c+cost+cost_to_goal(n_state,final_state),present_c2c+cost,cost_to_goal(n_state,final_state),node[4],n_state])
                    hq.heapify(open_list)
                    trajectory_dict[tuple(n_state)] = traj_list

        flag, traj_list, n_state, cost = action(node[4], canvas, 0, vel_l, R, L)
        if(flag):
            if tuple(n_state) not in closed_list:
                dup = False
                for i in range(len(open_list)):
                    if (open_list[i][4][0], open_list[i][4][1]) == (n_state[0],n_state[1]):
                        dup = True
                        n_cost = present_c2c+cost+cost_to_goal(n_state,final_state)
                        if(n_cost<open_list[i][0]):    # Updating the cost and parent info of the node
                            open_list[i][1] = present_c2c+cost
                            open_list[i][0] = n_cost
                            open_list[i][3] = node[4]
                            trajectory_dict[tuple(n_state)] = traj_list
                            hq.heapify(open_list)
                        break
                if(not dup):
                    hq.heappush(open_list,[present_c2c+cost+cost_to_goal(n_state,final_state),present_c2c+cost,cost_to_goal(n_state,final_state),node[4],n_state])
                    hq.heapify(open_list)
                    trajectory_dict[tuple(n_state)] = traj_list

        flag, traj_list, n_state, cost = action(node[4], canvas, vel_r, vel_r, R, L)
        if(flag):
            if tuple(n_state) not in closed_list:
                dup = False
                for i in range(len(open_list)):
                    if (open_list[i][4][0], open_list[i][4][1]) == (n_state[0],n_state[1]):
                        dup = True
                        n_cost = present_c2c+cost+cost_to_goal(n_state,final_state)
                        if(n_cost<open_list[i][0]):    # Updating the cost and parent info of the node
                            open_list[i][1] = present_c2c+cost
                            open_list[i][0] = n_cost
                            open_list[i][3] = node[4]
                            trajectory_dict[tuple(n_state)] = traj_list
                            hq.heapify(open_list)
                        break
                if(not dup):
                    hq.heappush(open_list,[present_c2c+cost+cost_to_goal(n_state,final_state),present_c2c+cost,cost_to_goal(n_state,final_state),node[4],n_state])
                    hq.heapify(open_list)
                    trajectory_dict[tuple(n_state)] = traj_list

        flag, traj_list, n_state, cost = action(node[4], canvas, vel_r, vel_l, R, L)
        if(flag):
            if tuple(n_state) not in closed_list:
                dup = False
                for i in range(len(open_list)):
                    if (open_list[i][4][0], open_list[i][4][1]) == (n_state[0],n_state[1]):
                        dup = True
                        n_cost = present_c2c+cost+cost_to_goal(n_state,final_state)
                        if(n_cost<open_list[i][0]):    # Updating the cost and parent info of the node
                            open_list[i][1] = present_c2c+cost
                            open_list[i][0] = n_cost
                            open_list[i][3] = node[4]
                            trajectory_dict[tuple(n_state)] = traj_list
                            hq.heapify(open_list)
                        break
                if(not dup):
                    hq.heappush(open_list,[present_c2c+cost+cost_to_goal(n_state,final_state),present_c2c+cost,cost_to_goal(n_state,final_state),node[4],n_state])
                    hq.heapify(open_list)
                    trajectory_dict[tuple(n_state)] = traj_list

        flag, traj_list, n_state, cost = action(node[4], canvas, vel_r, 0, R, L)
        if(flag):
            if tuple(n_state) not in closed_list:
                dup = False
                for i in range(len(open_list)):
                    if (open_list[i][4][0], open_list[i][4][1]) == (n_state[0],n_state[1]):
                        dup = True
                        n_cost = present_c2c+cost+cost_to_goal(n_state,final_state)
                        if(n_cost<open_list[i][0]):    # Updating the cost and parent info of the node
                            open_list[i][1] = present_c2c+cost
                            open_list[i][0] = n_cost
                            open_list[i][3] = node[4]
                            trajectory_dict[tuple(n_state)] = traj_list
                            hq.heapify(open_list)
                        break
                if(not dup):
                    hq.heappush(open_list,[present_c2c+cost+cost_to_goal(n_state,final_state),present_c2c+cost,cost_to_goal(n_state,final_state),node[4],n_state])
                    hq.heapify(open_list)
                    trajectory_dict[tuple(n_state)] = traj_list

        flag, traj_list, n_state, cost = action(node[4], canvas, 0, vel_r, R, L)
        if(flag):
            if tuple(n_state) not in closed_list:
                dup = False
                for i in range(len(open_list)):
                    if (open_list[i][4][0], open_list[i][4][1]) == (n_state[0],n_state[1]):
                        dup = True
                        n_cost = present_c2c+cost+cost_to_goal(n_state,final_state)
                        if(n_cost<open_list[i][0]):    # Updating the cost and parent info of the node
                            open_list[i][1] = present_c2c+cost
                            open_list[i][0] = n_cost
                            open_list[i][3] = node[4]
                            trajectory_dict[tuple(n_state)] = traj_list
                            hq.heapify(open_list)
                        break   
                if(not dup):
                    hq.heappush(open_list,[present_c2c+cost+cost_to_goal(n_state,final_state),present_c2c+cost,cost_to_goal(n_state,final_state),node[4],n_state])
                    hq.heapify(open_list)
                    trajectory_dict[tuple(n_state)] = traj_list
        #print(len(closed_list))
        print(len(open_list))
    if not back_track_flag:    
        print("\nNo Solution Found!")
        print("Total Number of Nodes Explored: ",len(closed_list))


def back_track(initial_state, final_state, closed_list, canvas, trajectory_dict):
    """
    Implements backtracking to the start node after reaching the goal node.
    This function is also used for visualization of explored nodes and computed path using OpenCV.
    A stack is used to store the intermediate nodes while transversing from the goal node to start node.
    
    Parameters:
        initial_state: List
                List of start node's x, y and theta parameters
        
        final_state: List
                List of goal node's x, y and theta parameters 
        closed_list: Dictionary
                Dictionary containing explored nodes and corresponding parents  
        canvas: NumPy array
                Map matrix with drawn obstacles 
    Returns:
            None
    """

    fourcc = cv2.VideoWriter_fourcc(*'XVID')    # Creating video writer to generate a video.
    out = cv2.VideoWriter('A-Star-amalapak-okritvik_testCase.avi',fourcc,500,(canvas.shape[1],canvas.shape[0]))
    
    print("Total Number of Nodes Explored = ",len(closed_list)) 
    
    keys = closed_list.keys()    # Returns all the nodes that are explored
    path_stack = []    # Stack to store the path from start to goal
    
    # Visualizing the explored nodes
    keys = list(keys)
    for key in keys:
        p_node = closed_list[tuple(key)]
        cv2.circle(canvas,(int(key[0]),int(key[1])),2,(0,0,255),-1)
        cv2.circle(canvas,(int(p_node[0]),int(p_node[1])),2,(0,0,255),-1)
        # canvas = cv2.arrowedLine(canvas, (int(p_node[0]),int(p_node[1])), (int(key[0]),int(key[1])), (0,255,0), 1, tipLength = 0.2)
        if(tuple(initial_state)!=tuple(key)):
            traj = trajectory_dict[tuple(key)]
            for i in range(0,len(traj)-1):
                cv2.line(canvas,(int(round(traj[i][0])),int(round(traj[i][1]))), (int(round(traj[i+1][0])),int(round(traj[i+1][1]))), (0,255,0),1)            
        cv2.imshow("A* Exploration and Optimal Path Visualization",canvas)
        cv2.waitKey(1)
        out.write(canvas)

    parent_node = closed_list[tuple(final_state)]
    path_stack.append(final_state)    # Appending the final state because of the loop starting condition
    
    while(parent_node != initial_state):
        path_stack.append(parent_node)
        parent_node = closed_list[tuple(parent_node)]
    
    path_stack.append(initial_state)    # Appending the initial state because of the loop breaking condition
    print("\nOptimal Path: ")
    start_node = path_stack.pop()
    print(start_node)

    # Visualizing the optimal path
    while(len(path_stack) > 0):
        path_node = path_stack.pop()
        traj = trajectory_dict[tuple(path_node)]
        for i in range(0,len(traj)-1):
            cv2.line(canvas,(int(round(traj[i][0])),int(round(traj[i][1]))), (int(round(traj[i+1][0])),int(round(traj[i+1][1]))), (255,0,196),3)
        # cv2.line(canvas,(int(start_node[0]),int(start_node[1])),(int(path_node[0]),int(path_node[1])),(255,0,196),5)
        print(path_node)
        start_node = path_node.copy()
        out.write(canvas)
    
    out.release()

if __name__ == '__main__':
    
    canvas = np.ones((1000,1000,3), dtype="uint8")    # Creating a blank canvas/map
    clearance, robot_radius, vel_l, vel_r = take_robot_inputs()
    canvas = draw_obstacles(canvas,offset = (clearance + robot_radius))
    initial_state, final_state = take_map_inputs(canvas) #Take the start and goal node from the user
    
    # Uncomment the below 3 lines to view the obstacle space. Press Any Key to close the image window
    # cv2.imshow("Canvas",canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Changing the input Cartesian Coordinates of the Map to Image Coordinates:
    initial_state[1] = canvas.shape[0]-1 - initial_state[1]
    final_state[1] = canvas.shape[0]-1 - final_state[1]
    # print(initial_state, final_state)

    # Converting the angles with respect to the image coordinates
    if initial_state[2] != 0:
        initial_state[2] = 360 - initial_state[2]

    print("\nStart Node (image coords): ", initial_state)
    print("Goal Node (image coords): ", final_state)

    start_time = time.time()

    cv2.circle(canvas,(int(initial_state[0]),int(initial_state[1])),2,(0,0,255),-1)
    cv2.circle(canvas,(int(final_state[0]),int(final_state[1])),2,(0,0,255),-1)
    R = 3.3
    L = 16
    astar(initial_state,final_state,canvas, vel_l, vel_r, R, L)    # Compute the optimal path using A* Algorithm
    
    end_time = time.time()    # Time taken for the algorithm to find the optimal path
    print("\nCode Execution Time (sec): ", end_time-start_time)    # Computes & prints the total execution time

    cv2.imshow("A* Exploration and Optimal Path Visualization", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()