# README

This project visualizes a potential field used for navigation. It includes two plots that illustrate how the robot experiences forces in its environment.

Figure_1.png shows the vector field on the entire workspace. Every arrow represents the combined attractive and repulsive force at that position. This offers an intuitive sense of how the field guides the robot toward the target while pushing it away from obstacles.

Figure_2.png shows the same potential field but as a continuous heatmap. Colors encode the magnitude of the total force. Valleys reveal places where the robot might get trapped in local minima. Peaks mark areas where repulsion is strongest.

Both plots help with debugging and tuning of the potential field, and they highlight behaviors that only become visible when the field is viewed at the global scale.
