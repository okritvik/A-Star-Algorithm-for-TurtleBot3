# A-Star Path Planning - Turtlebot3 (Gazebo)

Project - 03 (Phase 02, Part-02) for the course, 'ENPM661 - Planning for Autonomous Robots' at the University of Maryland, College Park.

Implementation of the A-star algorithm for path planning of the Turtlebot3 robot with a Non-Holonomic action set in Gazebo. 


<p align="center">
  <img src="https://user-images.githubusercontent.com/40534801/165009740-0e67c4b0-60e1-4f25-8f84-a1e2f49e42b4.png" width="800" height="500">
</p>


## Team Members:
* Kumara Ritvik Oruganti (117368963)
* Adarsh Malapaka (118119625)

## Required Libraries: 
* cv2 : To add lines or circles in the map at the desired coordinates.
* time: To calculate the running time for the Dijkstra algorithm.
* numpy: To define the obstacle map matrix
* heapq: Heap Queue to store opened_list nodes 
* rospy: To import ROS functionalities into the code
* geometry_msgs.msg: To define Twist() velocity messages 

Note: Make sure TurtleBot3 packages are installed in your catkin workspace.

## Test Case: 

Video Link: https://youtu.be/QSFRc_03UiY </br></br>
  [co-ordinates with respect to bottom left corner origin of the map in the PPT (scaled by 100)]  
  
Note: The origin in Gazebo is in the center whereas the origin defined in the PPT is in the bottom-left corner.

	Start-Node: (50, 50)   ---> Gazebo Map Coordinates: (-4.5, -4.5)

	Goal-Node:  (600, 300)   ---> Gazebo Map Coordinates: (1.0, -2.0)
	
	Robot Clearance: 10

	RPM1: 50 
	
	RPM2: 100

	Initial Heading Angle: 0


## Running the Code:

The code accepts the start and goal positions from the user through the command-line interface as mentioned below.
Add the package into your the src directory of your catkin workspace and run catkin_make

**Format/Syntax:**
		
    'roslaunch <package-name> <launch-file> <x_pos> <y_pos> <z_pos>'

Here: x_pos, y_pos, z_pos are optional arguments. Default: -4.5,-4.5, 0 

Note: The map is scaled by 100. So, while giving the optional arguments, the user has to account for the start co-ordinates.
For ex:  (-4.5, -4.5) ---> (50, 50)
	       (-4,-4) ---> (100, 100)
 

**Test Case:**	
		
     ```
     roslaunch proj3_phase2_okritvik_amalapak turtlebot3_661_proj3.launch x_pos:=-4.5 y_pos:=-4.5 z_pos:=0.0
     
     Enter the robot's clearance: 10
     Enter Left Wheel Velocity in RPM: 200
     Enter Right Wheel Velocity in RPM: 300
     Enter the X Coordinate of the Start Node: 50
     Enter the Y Coordinate of the Start Node: 50
     Enter the X Coordinate of the Goal Node: 600
     Enter the Y Coordinate of the Goal Node: 300
     Enter the Initial Head Angle: 0 
     ```
