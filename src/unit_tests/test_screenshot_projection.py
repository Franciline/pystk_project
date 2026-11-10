"""Run with PYTHONPATH=src python src/unit_tests/test_screenshot_projection.py."""
import numpy as np
from main.plot_trajectory_screenshot import project

points, visible = project(np.array([[0,0,0], [2,0,0], [0,0,2]]), np.eye(4), np.eye(4))
assert visible.tolist() == [True, False, False]
assert np.allclose(points[0], 0)
# Translation and perspective catch transposed Irrlicht matrix handling.
view = np.eye(4)
view[3, 0] = -1
projection = np.array([[1.,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,-.1,0]])
points, visible = project(np.array([[2,0,2],[1,0,-2]]), view, projection)
assert np.allclose(points[0], [.5,0,.95])
assert visible.tolist() == [True,False]
future = np.column_stack([np.linspace(0, .9, 10), np.zeros(10), np.zeros(10)])
points, visible = project(future, np.eye(4), np.eye(4))
assert points.shape == (10,3) and visible.all()
assert np.allclose(points, future)
print('Projection check passed')
