import numpy as np
import cv2
from scipy.optimize import newton

class Speckle:
    def __init__(
        self
    ):
        pass

    def generate_intensity(self, peak, R, x, y):
        return peak * np.exp(-((x - R) * (x - R) + (y - R) * (y - R)) / (R * R))

    def generate_speckle(self, image, R):
        height, width = image.shape
        peak = np.clip(np.random.normal(0, 1), 0, 1) * 255
        center = [np.random.randint(0, height), np.random.randint(0, width)]
        g = np.zeros((2 * R + 1, 2 * R + 1))

        for i in range(2 * R + 1):
            for j in range(2 * R + 1):
                x = center[0] - R + i
                y = center[1] - R + j
                if x < 0 or y < 0 or x >= height or y >= width:
                    continue

                image[x][y] = image[x][y] + self.generate_intensity(peak, R, i, j)

        return image
    
    def generate(self, image_shape, R, K):
        R //= 2
        image = np.zeros(shape = image_shape)
        for _ in range(K):
            image = self.generate_speckle(image, R)

        return image
    
    def get_M(self, wdith, height, focal, theta, phi, gamma, dx, dy, dz):
        
        w = wdith
        h = height
        f = focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])
        
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(theta), -np.sin(theta), 0],
                        [0, np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1]])
        
        RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                        [0, 1, 0, 0],
                        [np.sin(phi), 0, np.cos(phi), 0],
                        [0, 0, 0, 1]])
        
        RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                        [np.sin(gamma), np.cos(gamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))
    
    def rotate_along_axis(self, image, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
        width, height = image.shape
        # Get radius of rotation along 3 axes
        rtheta = np.deg2rad(theta)
        rphi = np.deg2rad(phi)
        rgamma = np.deg2rad(gamma)
        
        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = np.sqrt(height ** 2 + width ** 2)
        focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = focal

        # Get projection matrix
        mat = self.get_M(height, width, focal, rtheta, rphi, rgamma, dx, dy, dz)
        
        return cv2.warpPerspective(image, mat, (height, width), flags = cv2.INTER_CUBIC)
    
    def get_distorted_pattern(self, image, camera_matrix, distortion_coefficient):
        width, height = image.shape
        map_x = np.zeros((width, height)).astype(np.float32)
        map_y = np.zeros((width, height)).astype(np.float32)
        pts_distort = []
        for y in range(height):
            for x in range(width):
                pts_distort.append([x, y])
    
        pts_distort = np.array(pts_distort).reshape((-1, 1, 2)).astype(np.float32)
        pts_distort = cv2.undistortPoints(np.array(pts_distort), camera_matrix, distortion_coefficient, None, np.eye(3), camera_matrix)
        for y in range(height):
            for x in range(width):
                pt = pts_distort[y * width + x][0]
                map_x[x][y] = pt[1]
                map_y[x][y] = pt[0]
        distorted_image = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC)
        rotated_images = []
        rotated_image = self.rotate_along_axis(distorted_image, 45, 0, 0, 0, 0, 0)
        for degree in range(0, 360, 30):
            rotated_images.append(self.rotate_along_axis(rotated_image, 0, 0, degree, 0, 0, 0))
        return distorted_image, rotated_images
    
# class patch:
#     def __init__(
#         self,
#         image,
#         n_rows,
#         n_cols
#     ):
#         self.image = image
#         self.n_rows = n_rows
#         self.n_cols = n_cols
#         self.patchs = []
    
#     def split(self):
#         height, width = self.image.shape
#         n_row_pixels = height // self.n_rows
#         n_col_pixels = width // self.n_cols

#         for i in range(self.n_rows - 1):
#             row_patch = []
#             for j in range(self.n_cols - 1):
#                 row_patch.append(self.image[i * n_row_pixels: (i+1) * n_row_pixels])


if __name__ == '__main__':
    S = Speckle()
    image = S.generate((768, 768), 4, 100000)
    cv2.imwrite("standard_pattern.png", image)
    border_image = cv2.copyMakeBorder(image, 156, 156, 576, 576, borderType = cv2.BORDER_CONSTANT, value = (0, 0, 0))
    cv2.imwrite("border_standard_pattern.png", border_image)
    camera_matrix = np.array([[972.972972, 0, 540], [0, 972.972972, 960], [0, 0, 1]])
    distortion_coefficient = np.array([-0.1, -0.02, 0, 0, 0])
    distorted_image, rotated_images = S.get_distorted_pattern(border_image, camera_matrix, distortion_coefficient)
    cv2.imwrite("distorted_image.png", distorted_image)
    cv2.imwrite("rotated_image.png", rotated_images[0])
    for degree in range(0, 360, 30):
        cv2.imwrite(f"result/{degree}.png", rotated_images[degree // 30])