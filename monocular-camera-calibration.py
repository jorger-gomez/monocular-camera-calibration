"""monocular-camera-calibration.py

    This Python script is a modification of the version found in the following URL:
    https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html

    Author: Dr Andres Hernandez Gutierrez
    Organisation: Universidad de Monterrey
    Contact: andres.hernandezg@udem.edu
    First created: Friday 12 January 2024
    Last updated: Saturday 23 March 2024

    EXAMPLE OF USAGE:
    python monocular-camera-calibration.py --path_to_calibration_images sample-calibration-images/ \
    --calibration_image_format JPG \
    --chessboard_size 9 6 \
    --output_calibration_parameters calibration-parameters/calibration_data.json
"""

# Import standard libraries
import numpy as np
import argparse
from numpy.typing import NDArray
from typing import List

# Import user-defined Python module
from monocular_camera_calibration_helpers import( 
    parse_data_from_cli_monocular_camera_calibration, 
    check_directories_exist, 
    detect_chessboard_corners, 
    do_calibration, 
    compute_calibration_error, 
    write_calibration_parameters_to_disk
)

def run_pipeline(args:argparse.ArgumentParser)->None:   
    """
    Run the camera calibration pipeline.

    This function orchestrates the following steps:
    1. Check that directories exist; if not, create them.
    2. Detect corners on the chessboard calibration panel.
    3. Run the camera calibration process.
    4. Compute the calibration error in pixels.
    5. Write calibration parameters to a JSON file.

    Args:
        args: Parsed command-line arguments.

    Returns:
        None: The function does not return any value.
    """

    # Check that directories exist; if not, create them
    check_directories_exist(args)

    # Detect corners on the chessboard calibration panel
    objpoints, imgpoints, img_size = detect_chessboard_corners(args)

    # Run the camera calibration process
    mtx, dist, rvecs, tvecs = do_calibration(objpoints, 
                                                imgpoints, 
                                                img_size)

    # Compute the calibration error in pixels
    compute_calibration_error(objpoints, 
                                imgpoints, 
                                mtx, 
                                dist, 
                                rvecs, 
                                tvecs)

    # Write calibration parameters to JSON file
    write_calibration_parameters_to_disk(args.output_calibration_parameters, 
                                            mtx, 
                                            dist)
    

if __name__=="__main__":
    """
    This section of the script is executed only if the script is run directly 
    (not imported as a module). It parses command-line arguments using 
    'parse_data_from_cli' function, then runs the camera calibration pipeline 
    using 'run_pipeline' function.
    """
    args = parse_data_from_cli_monocular_camera_calibration()
    run_pipeline(args)
