# ME 459 Trajectory Planning and Optimization final

## Description  
This repository contains solutions for **ME 459 Exam 2**, focusing on trajectory planning, pursuit-evasion strategies, and the traveling salesman problem. The project includes Python scripts for solving optimization problems using A* and Dijkstra's algorithms, Monte Carlo simulations for projectile motion, and proportional navigation for robotic control.

## Files Included  

### **Part 1: Traveling Salesman Problem with A***  
- **File:** ME 459 Final TSP.py  
- **Topics Covered:**  
  - A* search for pathfinding  
  - Grid-based obstacle avoidance  
  - Traveling salesman problem (TSP) optimization  
  - Distance computation and heuristic evaluation  

### **Part 2: Proportional Navigation Pursuer**  
- **File:** Final_pursuer.py  
- **Topics Covered:**  
  - Proportional navigation (PN) control  
  - Lidar-based target tracking  
  - Adaptive heading control for pursuit  
  - Real-time trajectory adjustment  

### **Part 3: Dijkstra-Based Evader**  
- **File:** Final_evader.py  
- **Topics Covered:**  
  - Dijkstra's algorithm for trajectory planning  
  - Obstacle-aware pathfinding  
  - Autonomous evasion strategies  
  - Waypoint navigation  

### **Part 4: Monte Carlo Simulation for Projectile Motion**  
- **File:** Cannon_Ball_MC_Part4_v3.py  
- **Topics Covered:**  
  - Monte Carlo simulation for trajectory prediction  
  - Drag coefficient and wind variations  
  - Statistical analysis of landing locations  
  - Sensitivity analysis of launch angles  

### **Exam Document**  
- **File:** ME5501_Robot_Exam2_Sp22.pdf  
- **Contents:**  
  - Exam rules and guidelines  
  - Conceptual questions on search algorithms and optimization  
  - Problem descriptions for trajectory planning and pursuit-evasion tasks  

## Installation  
Ensure Python and the necessary dependencies are installed before running the scripts.

### Required Python Packages  
- numpy  
- matplotlib  
- scipy  
- tqdm  
- seaborn  
- rospy (for TurtleBot control)  

To install the required packages, run:  
```pip install numpy matplotlib scipy tqdm seaborn```  

## Usage  

### **Running the Traveling Salesman Problem (TSP) Solver**  
1. Open a terminal or command prompt.  
2. Navigate to the directory containing `ME 459 Final TSP.py`.  
3. Run the script using:  
   ```python ME 459 Final TSP.py```  
4. View the plotted shortest path and cost output.  

### **Executing the Pursuer and Evader Scripts**  
1. Ensure the ROS environment is properly set up.  
2. Start the pursuer script:  
   ```rosrun Final_pursuer.py```  
3. Start the evader script:  
   ```rosrun Final_evader.py```  
4. Observe the pursuer tracking the evader in real time.  

### **Running the Monte Carlo Simulation for Projectile Motion**  
1. Navigate to the directory containing `Cannon_Ball_MC_Part4_v3.py`.  
2. Run the script using:  
   ```python Cannon_Ball_MC_Part4_v3.py```  
3. View statistical results and plots for landing distances.  

## Example Output  

- **TSP Solver Output**  
  - Shortest Path: (2, 1, 3, 4, 5)  
  - Travel Cost: 52.87  

- **Pursuit-Evasion Simulation**  
  - Evader reaches target in 45 seconds  
  - Pursuer intercepts at (4,2)  

- **Monte Carlo Simulation Results**  
  - Best launch angle: 50 degrees  
  - Average landing distance: 1.93 km  

## Contributions  
This repository is intended for academic use. Contributions and modifications are welcome.  

## License  
This project is available for educational and research purposes.  

---  
**Author:** Alexander Dowell

