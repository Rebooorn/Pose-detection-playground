import numpy as np
import json
import matplotlib.pyplot as plt

# This is not used, however is important for 3D plot
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# path = os.path.join(os.path.dirname(__file__), 'data', 'car_models', '019-SUV.pkl')
with open('019-SUV.json', 'rb') as jf:
    car_model = json.load(jf)

# load the vertex of car model
vertex = car_model['vertices']
vertex = np.array(vertex, dtype=np.float16)
print(vertex.shape)

faces = car_model['faces']
faces = np.array(faces)
print(faces.shape)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

z = [np.ones(vertex.shape[0]), vertex[:,2]]
z = np.array(z)
ax.plot_surface(vertex[:, 0], vertex[:, 1], z)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_zlim(-2.5, 2.5)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
for i in range(faces.shape[0]):
    vert = np.array([vertex[faces[i, 0] - 1],
                     vertex[faces[i, 1] - 1],
                     vertex[faces[i, 2] - 1]])
    ax.add_collection3d(Poly3DCollection([vert]))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_zlim(-2.5, 2.5)
plt.show()