"""correct-image-distortion.py

    This Python scripts correct distorted images given the camera calibration parameters.

    Author: Dr Andres Hernandez Gutierrez
    Organisation: Universidad de Monterrey
    Contact: andres.hernandezg@udem.edu
    First created: Friday 12 January 2024
    Last updated: Saturday 23 March 2024

    EXAMPLE OF USAGE:
    python correct-image-distortion.py --input_calibration_parameters calibration-parameters/calibration_data.json \
    --path_to_distorted_images distorted-images-car/ \
    --format_of_distorted_images JPG \
    --path_to_undistorted_images undistorted-images-car/
    
    TODO:
        + Print information using a flag called 'verbose'.
"""

# Import standard libraries
import glob
import argparse 

# Import user-define Python module
from monocular_camera_calibration_helpers import(
    parse_data_from_cli_correct_image_distortion,
    load_calibration_parameters_from_json_file,
    undistort_images
)


def run_pipeline(args:argparse.ArgumentParser)->None:
    """
    Run the pipeline for undistorting images using camera calibration parameters.

    Args:
        args: Parsed command-line arguments.

    Returns:
        None: The function does not return any value.
    """

    # Load calibration parameters
    camera_matrix, distortion_coefficients = load_calibration_parameters_from_json_file(args)
    
    # Load distorted images
    list_of_undistorted_images = glob.glob(''.join([args.path_to_distorted_images, 
                                                    '*', 
                                                    args.format_of_distorted_images]))
    
    # Undistort images
    undistort_images(list_of_undistorted_images, 
                     camera_matrix, 
                     distortion_coefficients, 
                     args.path_to_undistorted_images)


if __name__=='__main__':   
    """
    This section of the script is executed only if the script is run directly 
    (not imported as a module). It parses command-line arguments specific to 
    correcting image distortion using 'parse_data_from_cli_correct_image_distortion' 
    function, then runs the pipeline using 'run_pipeline' function.
    """
    args = parse_data_from_cli_correct_image_distortion()
    run_pipeline(args)
