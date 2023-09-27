# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import tempfile
import logging
import os
import sys
import pprint
import shutil
from distutils.dir_util import copy_tree


def default_log_setup(level=logging.INFO):
    """
    Sets up default logging, always logging to stdout.

    :param level: logging level, e.g. logging.INFO
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    logging.basicConfig(level=level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)


def args_to_log_format(args_dict) -> str:
    """Convert args hierarchy to string representation suitable for logging (i.e. with Tensorboard).

    Args:
        args_dict : The parsed arguments, grouped within a dictionary.

    Returns:
        arg_str : The args encoded in a string format.
    """
    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f'```{args_str}```'
    return args_str

def copy_file(input_file, dest_dir=None) -> str:
    """Copy file from input directory to a temporary directory.

    Args:
        input_file (str): Input file containing data to be copied.
        dest_dir (str): Path to directory to be copied. Creates temporary directory
                        if None.

    Returns:
        dest_dir (str): Destination directory path
    """
    # Create destination directory
    if dest_dir is None:
        dest_dir = tempfile.mkdtemp()
    else:
        dest_dir = dest_dir.rstrip("/")
        os.makedirs(dest_dir,exist_ok=True)

    filename = os.path.basename(input_file)
    shutil.copyfile(input_file, os.path.join(dest_dir,filename))
    return os.path.join(dest_dir,filename)

def copy_dir(input_dir, dest_dir=None) -> str:
    """Copy data from input directory to a temporary directory.

    Args:
        input_dir (str): Input directory containing data to be copied.
        dest_dir (str): Path to directory to be copied. Creates temporary directory
                        if None.

    Returns:
        dest_dir (str): Destination directory path
    """
    # Create destination directory
    if dest_dir is None:
        dest_dir = tempfile.mkdtemp()
    else:
        dest_dir = dest_dir.rstrip("/")
        os.makedirs(dest_dir,exist_ok=True)

    copy_tree(input_dir, dest_dir)
    return dest_dir

def copy_dir_msrsync(input_dir, num_procs, msrsync_exec, dest_dir=None) -> str:
    """Copy data from input directory to a temporary directory for faster data loading. 
       Uses msrsync (https://github.com/jbd/msrsync) for fast parallelized copy.

    Args:
        input_dir (str): Input directory containing data to be copied.
        num_procs (int): Number of processes to use for msrsync
        msrsync_exec (str): Path to the executable msrsync binary
        dest_dir (str): Path to directory to be copied. Creates temporary directory
                        if None.

    Returns:
        dest_dir (str): Destination directory path
    """

    # Create destination directory
    if dest_dir is None:
        dest_dir = tempfile.mkdtemp()
    else:
        dest_dir = dest_dir.rstrip("/")
        os.makedirs(dest_dir,exist_ok=True)

    data_name = input_dir.rstrip("/").split("/")[-1]

    # Maintain flag if copying is complete
    destination_dir = f"{dest_dir}/{data_name}"
    complete_flag = f"{destination_dir}/copy_complete"

    if os.path.exists(complete_flag):
        print(f'Found data already copied to {destination_dir}')
        return destination_dir
    else:
        print(
            f"Copying {input_dir} to dir {destination_dir} using {num_procs} processes..."
        )
        # We have to do multi-threaded rsync to speed up copy.
        cmd = (
            f"{msrsync_exec} -p {num_procs} {input_dir.rstrip('/')} {dest_dir}"
        )
        os.system(cmd)
        open(complete_flag, "a").close()
        print(f"Copied to directory {destination_dir}")
        return destination_dir
    