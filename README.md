# README

This project visualizes a potential field used for navigation. It includes two plots that illustrate how the robot experiences forces in its environment.

Figure_1.png shows the vector field on the entire workspace. Every arrow represents the combined attractive and repulsive force at that position. This offers an intuitive sense of how the field guides the robot toward the target while pushing it away from obstacles.

Figure_2.png shows how the robot responds to that field along its trajectory. It plots left wheel speed, right wheel speed, and yaw over time. These curves reveal how the controller reacts to attractive and repulsive forces, and they make it easy to spot oscillations, saturation, or sudden steering events.

Both plots help with debugging and tuning of the potential field, and they highlight behaviors that only become visible when the field is viewed at the global scale.
