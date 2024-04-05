"""Homework 08: 3D measurements of flat surface objects

    This Python script is a modification of the version found in the following URL:
    https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html

    Author: Jorge Rodrigo Gómez Mayo
    Organisation: Universidad de Monterrey
    Contact: jorger.gomez@udem.edu

    EXAMPLE OF USAGE
    python get-measurements.py
    -c 0
    --z 50
    -j calibration_data.json

    python get-measurements.py -c 0 --z 50 -j calibration_data.json

"""
import numpy as np
import cv2
import glob 
import os
import argparse
import sys
import textwrap
import json
import platform
from numpy.typing import NDArray
from typing import List, Tuple, Optional

# Global variables to store points and the drawing state
points: List[Tuple[int, int]] = [] # List to store the points of the drawing
drawing: bool = False # Flag to indicate if the drawing is complete

def user_arguments() -> argparse.Namespace:
    """
    Parses user arguments from the command line.

    Returns:
        A namespace with the parsed arguments.
    """
    parser = argparse.ArgumentParser(prog='HW8 - 3D measurements of flat surface objects', 
                                    description='Calculate dimensions of user provided geometries.', 
                                    epilog='JRGM - 2024')
    parser.add_argument('--camera_index', '-c', type=int, required=True, help="Index for desired camera ")
    parser.add_argument('--z', type=float, required=True, help="Distance between camera and object")
    parser.add_argument('--input_calibration_parameters', '-j', type=str, required=True, help='JSON file with calibration parameters')
    return parser.parse_args()

def load_calibration_parameters_from_json_file(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads camera calibration parameters from a specified JSON file.

    Args:
        args: Command line arguments provided by the user.

    Returns:
        A tuple containing the camera matrix and distortion coefficients.
    """
    json_filename = args.input_calibration_parameters
    if not os.path.isfile(json_filename):
        print(f"The file {json_filename} does not exist!")
        sys.exit(-1)

    with open(json_filename) as f:
        json_data = json.load(f)
        
    camera_matrix = np.array(json_data['camera_matrix'])
    distortion_coefficients = np.array(json_data['distortion_coefficients'])
    return camera_matrix, distortion_coefficients

def initialize_camera(args: argparse.Namespace) -> Optional[cv2.VideoCapture]:
    """
    Initializes the camera using the index specified in the arguments.

    Args:
        args: Command line arguments provided by the user.

    Returns:
        An initialized VideoCapture object if successful, None otherwise.
    """
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("Error initializing the camera")
        return None
    return cap

def mouse_callback(event: int, x: int, y: int, flags: int, param: any) -> None:
    """
    Callback function for mouse events.

    Args:
        event: The type of the mouse event.
        x: The x-coordinate of the mouse event.
        y: The y-coordinate of the mouse event.
        flags: Flags associated with the mouse event.
        param: Additional parameters.
    """
    global points, drawing

    if event == cv2.EVENT_RBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_ALTKEY:
            if points:
                points.clear()
        else:
            if points:
                points.pop()
    elif event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
    elif event == cv2.EVENT_MBUTTONDOWN:
        points.append(points[0])
        drawing = True

def undistort_images(frame: NDArray, mtx: NDArray, dist: NDArray) -> NDArray:
    """
    Undistorts the given image frame using the provided camera matrix and distortion coefficients.

    Args:
        frame: The image frame to be undistorted.
        mtx: The camera matrix.
        dist: The distortion coefficients.

    Returns:
        The undistorted image.
    """
    # Get size
    h,  w = frame.shape[:2]

    # Get optimal new camera
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    # Undistort image
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Crop image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

def compute_line_segments(points: List[Tuple[int, int]]) -> List[float]:
    """
    Computes the lengths of line segments between consecutive points.

    Args:
        points: A list of points defining the line segments.

    Returns:
        A list of lengths for each line segment.
    """
    line_length = [] #matriz donde se guardarán las distancias medidas
    for i in range(1, len(points)):
        x1, y1 = points[i-1] #punto punto anterior
        x2, y2 = points[i] # punto nuevo
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) #distancia entre dos puntos
        line_length.append(length) #agrega dato a matriz
    return line_length

def compute_perimeter(points: List[Tuple[int, int]], z: float, mtx: np.ndarray, height: int, width: int) -> Tuple[List[float], float]:
    """
    Computes the distance between points and the perimeter of the shape formed by those points.

    Args:
        points: A list of tuples representing the points (x, y).
        z: The distance from the camera to the object.
        mtx: The camera matrix.
        height: The height of the image.
        width: The width of the image.

    Returns:
        A tuple containing a list of distances between consecutive points and the total perimeter.
    """
    distance = []
    perimeter = 0.0

    Cx = mtx[0,2]*width/1280
    Cy = mtx[1,2]*height/720
    
    fx = mtx[0,0]*width/1280
    fy = mtx[1,1]*height/720

    for i in range(1, len(points)):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        
        # Convertir de píxeles a coordenadas
        X1 = (x1 - Cx) * z / fx
        Y1 = (y1 - Cy) * z / fy
        X2 = (x2 - Cx) * z / fx
        Y2 = (y2 - Cy) * z / fy
        
        # Calcular distancia entre puntos
        dist = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2 )
        distance.append(dist)

        perimeter += dist
    return distance, perimeter

def pipeline() -> None:
    """
    Main function to run the measurement pipeline. Allows user to draw on the live camera feed, compute distances and perimeter, 
    and closes the camera when 'c' key is pressed.
    """
    global drawing
    args = user_arguments()

    cam = initialize_camera(args)
    if cam is None:
        return  # If camera initialization failed, exit the function

    cv2.namedWindow('Live Camera View')
    cv2.setMouseCallback('Live Camera View', mouse_callback)

    # Attempt to load the camera calibration parameters
    try:
        camera_matrix, distortion_coefficients = load_calibration_parameters_from_json_file(args)
    except FileNotFoundError as e:
        print(e)
        sys.exit(-1)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: No video signal.")
            break
        
        h,w = frame.shape[:2]

        if drawing:
            #calcular distancias y perimetros
            distance,perimeter = compute_perimeter(points, args.z,mtx,h,w)

            # distancias a dos puntos decimales en orden de registro
            text = "Distancias (puntos seleccionados):\n"
            for i, dist in enumerate(distance, start=1):
                if i < len(points):
                    text += f"Punto {i}-{i+1}: {dist:.2f} cm\n"
                else:
                    text += f"Punto {i}-{1}: {dist:.2f} cm\n"

            text += f"\nPerímetro total: {perimeter:.2f} cm"

            text += "\nMedidas ordenadas de mayor a menor:\n"
            sorted_distance = sorted(distance, reverse=True)
            for i, dist in enumerate(sorted_distance, start=1):
                index = distance.index(dist)
                if index == len(points) - 1:
                    text += f"Punto {len(points)}-{1}: {dist:.2f} cm\n"
                else:
                    text += f"Punto {index+1}-{index+2}: {dist:.2f} cm\n"

            print(text)
            drawing = False  #reiniciar estado del dibujo 

        # Dibujar las líneas entre los puntos seleccionados
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], (0, 255, 0), 1)

        # Dibujar los puntos seleccionados
        for point in points:
            cv2.circle(frame, point, 3, (0, 255, 0), -1)

        cv2.imshow('Live Camera View', frame)

        # cv2.waitKey(1) returns a 32-bit integer corresponding to the pressed key
        # & 0xFF masks the integer to get the last 8 bits, which correspond to the key code on most systems
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):  # If 'c' key is pressed
            break
        elif k == 27:  # Optionally, also allow ESC key to exit
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    pipeline()            