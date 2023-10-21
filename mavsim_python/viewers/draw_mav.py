"""
mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        4/15/2019 - BGM
"""
import numpy as np
import pyqtgraph.opengl as gl
from tools.rotations import Euler2Rotation


class DrawMav:
    def __init__(self, state, window):
        """
        Draw the MAV.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.north  # north position
            state.east  # east position
            state.altitude   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        # get points that define the non-rotated, non-translated mav and the mesh colors
        self.mav_points, self.mav_meshColors = self.get_points()

        mav_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining mav
        rotated_points = self.rotate_points(self.mav_points, R)
        translated_points = self.translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self.points_to_mesh(translated_points)
        self.mav_body = gl.GLMeshItem(vertexes=mesh,  # defines the triangular mesh (Nx3x3)
                                      vertexColors=self.mav_meshColors,  # defines mesh colors (Nx1)
                                      drawEdges=True,  # draw edges between mesh elements
                                      smooth=False,  # speeds up rendering
                                      computeNormals=False)  # speeds up rendering
        #self.mav_body.setGLOptions('translucent')
        # ============= options include
        # opaque        Enables depth testing and disables blending
        # translucent   Enables depth testing and blending
        #               Elements must be drawn sorted back-to-front for
        #               translucency to work correctly.
        # additive      Disables depth testing, enables blending.
        #               Colors are added together, so sorting is not required.
        # ============= ======================================================
        window.addItem(self.mav_body)  # add body to plot
        # default_window_size = (500, 500)
        # window.resize(*default_window_size)


    def update(self, state):
        mav_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining mav
        rotated_points = self.rotate_points(self.mav_points, R)
        translated_points = self.translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self.points_to_mesh(translated_points)
        # draw MAV by resetting mesh using rotated and translated points
        self.mav_body.setMeshData(vertexes=mesh, vertexColors=self.mav_meshColors)

    def rotate_points(self, points, R):
        "Rotate points by the rotation matrix R"
        rotated_points = R @ points
        return rotated_points

    def translate_points(self, points, translation):
        "Translate points by the vector translation"
        translated_points = points + np.dot(translation, np.ones([1, points.shape[1]]))
        return translated_points

    def get_points(self):
        """"
            Points that define the mav, and the colors of the triangular mesh
            Define the points on the aircraft following diagram in Figure C.3
        """
        # define MAV body parameters
        fuse_h = 3
        fuse_w = 2
        fuse_l1 = 4
        fuse_l2 = 2
        wing_l = 4
        wing_w = 12
        fuse_l3 = 12
        tailwing_l = 2.5
        tailwing_w = 6
        tail_h = 5

        # Define the points on the aircraft following diagram Fig 2.14
        # points are in NED coordinates
        points = np.array([[fuse_l1,                0,                0],             # point 1 [0]
                           [fuse_l2,                0.5 * fuse_w,    -0.5 * fuse_h],  # point 2 [1]
                           [fuse_l2,               -0.5 * fuse_w,    -0.5 * fuse_h],  # point 3 [2]
                           [fuse_l2,               -0.5 * fuse_w,     0.5 * fuse_h],  # point 4
                           [fuse_l2,                0.5 * fuse_w,     0.5 * fuse_h],  # point 5
                           [-fuse_l3,               0,                0],             # point 6
                           [0,                      0.5 * wing_w,     0],             # point 7
                           [-wing_l,                0.5 * wing_w,     0],             # point 8
                           [-wing_l,               -0.5 * wing_w,     0],             # point 9
                           [0,                     -0.5 * wing_w,     0],             # point 10
                           [-fuse_l3 + tailwing_l,  0.5 * tailwing_w, 0],             # point 11
                           [-fuse_l3,               0.5 * tailwing_w, 0],             # point 12
                           [-fuse_l3,              -0.5 * tailwing_w, 0],             # point 13
                           [-fuse_l3 + tailwing_l, -0.5 * tailwing_w, 0],             # point 14
                           [-fuse_l3 + tailwing_l,  0,                0],             # point 15
                           [-fuse_l3,               0,               -tail_h]         # point 16
                           ]).T

        # scale points for better rendering
        scale = 1
        points = scale * points
        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        meshColors = np.empty((13, 3, 4), dtype=np.float32)

        # Assign colors for each mesh section
        for i in range(4):
            meshColors[i] = yellow
        for i in range(4, 8):
            meshColors[i] = blue
        for i in range(8, 12):
            meshColors[i] = green
        meshColors[12] = red

        return points, meshColors

    def points_to_mesh(self, points):
        """"
        Converts points to triangular mesh
        Each mesh face is defined by three 3D points
          (a rectangle requires two triangular mesh faces)
        """
        points = points.T

        #Define each section of the mesh with 3 points
        mesh = np.array([
            [points[0], points[1], points[2]],     # nose top
            [points[0], points[2], points[3]],     # nose west
            [points[0], points[3], points[4]],     # nose bottom
            [points[0], points[1], points[4]],     # nose east
            [points[1], points[2], points[5]],     # body top
            [points[2], points[3], points[5]],     # body west
            [points[3], points[4], points[5]],     # body bottom
            [points[1], points[4], points[5]],     # body east
            [points[6], points[7], points[9]],     # wing a
            [points[8], points[9], points[7]],     # wing b
            [points[10], points[11], points[13]],  # tail wing a
            [points[11], points[12], points[13]],  # tail wing b
            [points[14], points[5], points[15]],   # tail
        ])
        return mesh
