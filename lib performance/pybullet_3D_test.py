import pybullet as p
import pybullet_data
import time
import random
import math

# Constants
NUM_BODIES = 100
G = 6.67430e-11  # Gravitational constant
TIME_STEP = 1.0 / 300.0  # Target time step
MASS = 1000  # Mass of each body
RADIUS = 0.5  # Radius of each sphere
INIT_VELOCITY_MAGNITUDE = 10  # Adjust this to control initial speed
ALPHA = 0.5  # Transparency

# Initialize PyBullet in GUI mode
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=50, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

bodies = []
initial_velocities = []

# Create 100 spheres
for _ in range(NUM_BODIES):
    init_position = [random.uniform(-50, 50), random.uniform(-50, 50), random.uniform(-50, 50)]
    init_velocity = [random.uniform(-INIT_VELOCITY_MAGNITUDE, INIT_VELOCITY_MAGNITUDE) for _ in range(3)]
    initial_velocities.append(init_velocity)

    sphere_color = [random.random(), random.random(), random.random(), ALPHA]  # Translucent color
    sphere_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=RADIUS)
    sphere_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=RADIUS, rgbaColor=sphere_color)

    body_id = p.createMultiBody(baseMass=MASS, baseCollisionShapeIndex=sphere_collision_shape,
                                baseVisualShapeIndex=sphere_visual_shape, basePosition=init_position)
    p.resetBaseVelocity(body_id, linearVelocity=init_velocity)
    bodies.append(body_id)

# Function to apply gravitational forces between all bodies
def apply_gravitational_forces():
    for i in range(NUM_BODIES):
        pos_i, _ = p.getBasePositionAndOrientation(bodies[i])
        for j in range(i + 1, NUM_BODIES):
            pos_j, _ = p.getBasePositionAndOrientation(bodies[j])
            dist_vector = [pos_j[k] - pos_i[k] for k in range(3)]
            dist_squared = sum([d**2 for d in dist_vector])
            dist = math.sqrt(dist_squared)
            if dist > 0:
                force_magnitude = (G * MASS * MASS) / dist_squared
                force_vector = [force_magnitude * (d / dist) for d in dist_vector]
                p.applyExternalForce(bodies[i], -1, force_vector, pos_i, p.WORLD_FRAME)
                opposite_force = [-f for f in force_vector]
                p.applyExternalForce(bodies[j], -1, opposite_force, pos_j, p.WORLD_FRAME)

# Camera control variables
camera_distance = 50
camera_yaw = 0
camera_pitch = -30

# Simulation loop
frame_count = 0
start_time = time.time()

while True:
    # Get camera controls (optional, for user)
    keys = p.getKeyboardEvents()
    if ord('w') in keys:
        camera_distance -= 0.5
    if ord('s') in keys:
        camera_distance += 0.5
    if ord('a') in keys:
        camera_yaw -= 1
    if ord('d') in keys:
        camera_yaw += 1
    if ord('q') in keys:
        camera_pitch += 1
    if ord('e') in keys:
        camera_pitch -= 1

    # Update the camera position
    p.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                                 cameraYaw=camera_yaw,
                                 cameraPitch=camera_pitch,
                                 cameraTargetPosition=[0, 0, 0])

    # Apply gravitational forces
    apply_gravitational_forces()

    # Step the simulation
    p.stepSimulation()

    # Measure frame time
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time

    # Calculate actual FPS
    if elapsed_time > 0:
        actual_fps = frame_count / elapsed_time
        print(f"Actual FPS: {actual_fps:.2f}")

    # Enforce the frame rate limit
    time.sleep(max(0, TIME_STEP - (time.time() - current_time)))  # Sleep to maintain target FPS

    # Check if the actual FPS is less than the target
    if actual_fps < 30:
        print("Warning: Simulation may be too heavy for the system.")

# Disconnect PyBullet after use
p.disconnect()
