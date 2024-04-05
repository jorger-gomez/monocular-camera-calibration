"""monocular_camera_calibration_helpers.py

    TODO: 
        + Print information using a flag called 'verbose'.
        + Include DocString 
"""

# Import standard libraries
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
from typing import List, Tuple


def parse_data_from_cli_monocular_camera_calibration()->argparse.ArgumentParser:
    """
    Parse command-line arguments for camera calibration.

    Returns:
        argparse.ArgumentParser: The argument parser object configured for camera calibration.
    """

    # Parse user's argument
    parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter,
                                    description=textwrap.dedent('''\
            This Python script performs the calibration process for a monocular
            camera. It uses a chessboard calibration panel to estimate both the
            camera matrix and the lens distortions needed to subsequently undis-
            tor any other image acquired by that particular camera.

            '''))
    parser.add_argument('-p',
                        '--path_to_calibration_images', 
                        type=str, 
                        required=True,
                        default='calibration-images',
                        help="Folder where the calibration images are")
    parser.add_argument('-c',
                        '--calibration_image_format',
                        type=str,
                        required=True,
                        help="Image format of calibration images e.g., JPG, PNG, or jpeg")
    parser.add_argument('-s',
                        '--chessboard_size',
                        nargs='+')
    parser.add_argument('-o',
                        '--output_calibration_parameters',
                        type=str,
                        required=True,
                        help='Path to save the calibration parameters')
    args = parser.parse_args()

    return args


def parse_data_from_cli_correct_image_distortion()->argparse.ArgumentParser:

    # Parse user's argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_calibration_parameters', 
                        type=str, 
                        required=True,
                        help="JSON file containing the camera calibration parameters")
    parser.add_argument('--path_to_distorted_images',
                        type=str,
                        required=True,
                        help="Path to distorted images")
    parser.add_argument('--format_of_distorted_images',
                        type=str,
                        required=True,
                        choices=['PNG', 'JPG', 'JPEG', 'BMP'],
                        help='Format of distorted images [PNG, JPG, JPEG, BMP]')
    parser.add_argument('--path_to_undistorted_images',
                        type=str,
                        required=False,
                        help='Path to saved undistorted images')
    args = parser.parse_args()
    return args


def detect_chessboard_corners(
        args:argparse.ArgumentParser
        )->Tuple[List[np.float64], List[np.float64], Tuple[int, int]]:
    """
    Detect chessboard corners in calibration images.

    Args:
        args: Parsed command-line arguments containing calibration information.

    Returns:
        objpoints: List of object points in real-world space.
        imgpoints: List of image points in image plane. 
        dim: dimensions of the resized calibration images.
    """

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Get number of corners that should be detected
    nrows = int(args.chessboard_size[0])
    ncols = int(args.chessboard_size[1])

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nrows*ncols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:ncols, 0:nrows].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    dim = (0, 0)

    # Path to calibration images
    path_to_calibration_images = args.path_to_calibration_images

    # Load calibration images
    full_path = ''.join([path_to_calibration_images, '*.', args.calibration_image_format])
    images = glob.glob(full_path)

    # Create a new window for visualisation purposes
    cv2.namedWindow('Current calibration image', cv2.WINDOW_NORMAL)
    
    # Do corner detection
    print("\nFinding corners in chessboard calibration pattern...")
    for fname in images:

        # Read current calibration image
        img = cv2.imread(fname)

        if img.size == 0:
            print(f"ERROR! - image {fname} does not exist")
            exit(1)

        # Convert from BGR to greyscale image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (ncols, nrows), None)

        # If found, add object points, image points (after refining them)
        if ret:

            # Find corners on chessboard
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (ncols, nrows), corners2, ret)

            # Resize current calibration image
            scale_percent = 100
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            # Visualise current calibration image with detected corners
            cv2.imshow('Current calibration image', img)
            cv2.waitKey(250)

    print("Process completed!")
    
    # Destroy all windows
    cv2.destroyAllWindows()

    # Return required parameters
    return (objpoints, imgpoints, dim)


def do_calibration(
        objpoints:List[np.float64], 
        imgpoints:List[np.float64], 
        img_size:Tuple[int, int])->Tuple[NDArray, 
                                         NDArray, 
                                         Tuple[np.float64], 
                                         Tuple[np.float64]]:
    """
    Perform camera calibration using object points and image points.

    Args:
        objpoints: List of object points in real-world space.
        imgpoints: List of image points in image plane.
        img_size: Size of the images (width, height).

    Returns:
        mtx: Camera matrix 
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """

    # Perform camera calibration
    print("\nPerforming camera calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                       imgpoints, 
                                                       img_size, 
                                                       None, 
                                                       None)
    # If the camera calibration could not be conducted...
    if not ret:
        print("The camera calibration process could not be completed!")
        print('Exiting the program ...')
        sys.exit(1)

    # Print calibration parameters
    print("\nCamera calibration parameters:")
    print(f"\n camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]:\n {mtx}")
    print(f"\n distortion coefficients [k1, k2, p1, p2, k3]:\n {dist}")
    print(" Process completed!")
    return (mtx, dist, rvecs, tvecs)


def compute_calibration_error(objpoints:List[float], 
                              imgpoints:List[float], 
                              mtx:NDArray, 
                              dist:NDArray, 
                              rvecs:Tuple[float], 
                              tvecs:Tuple[float])->None:
    """
    Compute the average calibration error.

    Args:
        objpoints: List of object points in real-world space.
        imgpoints: List of image points in image plane.
        mtx: Camera matrix.
        dist: Distortion coefficients.
        rvecs: Rotation vectors.
        tvecs: Translation vectors.

    Returns:
        None: The function does not return any value, but prints the 
        average calibration error.
    """

    print(f"\nComputing average calibration error...")
    # Compute the average calibration error by finding the distance
    # between the reprojected pixels and the detected corners
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "Total error: {}".format(mean_error/len(objpoints)) )
    print("Process completed!")


def write_calibration_parameters_to_disk(
        path_to_calibration_file:str, 
        mtx:NDArray, 
        dist:NDArray)->None:
    """
    Write camera calibration parameters to a JSON file.

    Args:
        path_to_calibration_file: Path to the output calibration file.
        mtx: Camera matrix.
        dist: Distortion coefficients.

    Returns:
        None: The function does not return any value, but saves the calibration 
        parameters into a JSON file.
    """

    # Convert 'mtx' and 'dist' to list()
    mtx_data = mtx.tolist()
    dist_data = dist.tolist()
    
    # Build a dictionary to hold the camera calibration parameters
    data = {
        "camera_matrix": mtx_data,
        "distortion_coefficients": dist_data
    }
    
    # Write the calibration parameters into a JSON file
    full_path = ''.join([os.getcwd(), path_to_calibration_file])
    print(f"\nSaving calibration parameters into file: \n{full_path}")
    with open(path_to_calibration_file, "w") as file:
        json.dump(data, file, indent=4)
    print("Process completed!\n")


def check_directories_exist(args:argparse.ArgumentParser)->None:
    """
    Check if the directories specified in the command-line arguments exist.
    If not, it exits the program. If the directory exist, then it continues
    creating the directory to hold the camera calibration parameters.

    Args:
        args (argparse.ArgumentParser): Parsed command-line arguments.

    Returns:
        None: The function does not return any value.
    """

    # Check if the path to calibration images exists
    if not os.path.isdir(args.path_to_calibration_images):
        print(f"The path to calibration images: {args.path_to_calibration_images} \
              does not exist!")
        sys.exit(1)

    # Retrieve the operating system on which Python interpreter is running 
    if platform.system() in 'Windows':

        # If the OS is Windows, then use '\'
        path_to_calibration_parameters = args.output_calibration_parameters.split('\\')
    else:

        # If the OS is Linux or Mac, then use '/'
        path_to_calibration_parameters = args.output_calibration_parameters.split('/')
        
    # Retrieve the path to JSON file
    path_to_calibration_parameters = os.path.join(*path_to_calibration_parameters[:-1])
    
    # If the path to JSON file does not exist, then it is created.
    # If that path already exists, then it is overwritten 
    if not os.path.isdir(args.output_calibration_parameters):
        os.makedirs(path_to_calibration_parameters, exist_ok=True)




def load_calibration_parameters_from_json_file(
        args:argparse.ArgumentParser
        )->None:
    """
    Load camera calibration parameters from a JSON file.

    Args:
        args: Parsed command-line arguments.

    Returns:
        camera_matrix: Camera matrix.
        distortion_coefficients: Distortion coefficients.

    This function may raise a warning if the JSON file 
    does not exist. In such a case, the program finishes.
    """

    # Check if JSON file exists
    json_filename = args.input_calibration_parameters
    check_file = os.path.isfile(json_filename)

    # If JSON file exists, load the calibration parameters
    if check_file:
        f = open(json_filename)
        json_data = json.load(f)
        f.close()
        
        camera_matrix = np.array(json_data['camera_matrix'])
        distortion_coefficients = np.array(json_data['distortion_coefficients'])
        return camera_matrix, distortion_coefficients
    
    # Otherwise, the program finishes
    else:
        print(f"The file {json_filename} does not exist!")
        sys.exit(-1)


def undistort_images(
        list_of_undistorted_images:str, 
        mtx:NDArray, 
        dist:NDArray, 
        path_to_saving_undistorted_images:str
        )->None:
    """
    Undistort a list of distorted images using camera calibration parameters and save 
    the undistorted images.

    Args:
        list_of_undistorted_images: List of paths to distorted images.
        mtx: Camera matrix.
        dist: Distortion coefficients.
        path_to_saving_undistorted_images: Path to save undistorted images.

    Returns:
        None: The function does not return any value.
    """

    # Loop through distorted images
    for fname in list_of_undistorted_images:

        print("Undistorting: {}".format(fname))
        img_names = fname.split('\\')[-1]

        # read current distorted image
        img = cv2.imread(fname)

        # Get size
        h,  w = img.shape[:2]

        # Get optimal new camera
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

        # Undistort image
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # Crop image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        # If the path to JSON file does not exist, then it is created.
        # If that path already exists, then it is overwritten 
        if not os.path.isdir(path_to_saving_undistorted_images):
            os.makedirs(path_to_saving_undistorted_images, exist_ok=True)
            
        cv2.imwrite(path_to_saving_undistorted_images+img_names, dst)
        print("Undistorted image saved in:{}".format(path_to_saving_undistorted_images+img_names)) 
