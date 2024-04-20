import numpy as np
import os
import time
import sys


def do_projection(u, delta, mask):
    # sum u**2 * dx = sum u**2 * L * (dx / L) = sum u**2 / n * L, let L = 1
    if mask is None:
        mask = np.ones_like(u, dtype=bool)
    else:
        if not mask.all():
            # if not all true
            raise NotImplementedError("Not sure masked projection works correctly in cnop.")
    if not np.array_equal(u.shape, mask.shape):
        raise ValueError("u and mask must have the same shape.")
    mean = (u[mask] ** 2.).mean()
    if np.sqrt(mean) <= delta:
        proj_u = u
    else:
        proj_u = u
        proj_u[mask] = delta / np.sqrt(mean) * u[mask]
    return proj_u


def usphere_sample(n):
    # Generate standard normal random variables
    tmp = np.random.randn(n)
    # Find the magnitude of each column.
    # Square each element, add and take the square root.

    mag = np.sqrt((tmp ** 2.).sum())

    # Make a diagonal matrix of them -- inverses.
    dm = 1.0 / mag
    # Multiply to scale properly.
    # Transpose so x contains the observations.
    x = dm * tmp
    return x


def compute_obj(process, u_pert, t):
    # compute the objective value
    ut = process.proceed(t)
    ut_pert = process.proceed(t, u_pert=u_pert)
    j_val = - ((ut_pert - ut) ** 2).sum()
    return j_val


def wait_for_file(file_path, timeout=60, poll_interval=1):
    """
    Wait for a file to appear
    :param file_path: path of the file
    :param timeout: timeout if file_path does not exist
    :param poll_interval: poll interval in seconds
    :return: True if the file appears before timeout, False otherwise
    """
    start_time = time.time()

    # first, check if the file already exists with the timeout
    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            return True
        else:
            time.sleep(poll_interval)
    return False


def wait_for_files(file_paths, timeout=60, poll_interval=1):
    """
    Wait for a list of files to appear
    :param file_paths: list of paths of the files
    :param timeout: timeout if file_paths do not exist
    :param poll_interval: poll interval in seconds
    :return: True if all files appear before timeout, False otherwise
    """
    start_time = time.time()

    # first, check if the file already exists with the timeout
    while time.time() - start_time < timeout:
        if all([os.path.exists(file_path) for file_path in file_paths]):
            return True
        else:
            time.sleep(poll_interval)
    return False


def wait_for_last_line(file_path, ending_remark, timeout=np.inf, poll_interval=10):
    """
    Wait for the last line of a file to contain a specific ending remark
    :param file_path: path of the file
    :param ending_remark: ending remark to be checked
    :param timeout: timeout if the file does not contain the ending remark
    :param poll_interval: poll interval in seconds
    :return: True if the file contains the ending remark before timeout, False otherwise
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if len(lines) > 0:
                if lines[-1].strip() == ending_remark:
                    return True
        time.sleep(poll_interval)

    return False


def generate_shell_wrapper(exec_command, wrapper_name, wrapper_output,
                           wrapper_path=None, ending_remark=""):
    """
    Generate a wrapper script for running the executable
    so that the stdout can be redirected to files when using MPI.COMM_SELF.Spawn
    :param exec_command: name of the executable written in the wrapper script
    :param wrapper_name: name of the wrapper script
    :param wrapper_output: name of the file to redirect the stdout
    :param wrapper_path: path of the wrapper script
    :param ending_remark: ending remark to be written to the wrapper_output
    :return: None
    """
    if wrapper_path is None:
        wrapper_path = os.getcwd()
    code_to_write = f"""#!/bin/bash
{exec_command} &> {wrapper_output}
echo "{ending_remark}" >> {wrapper_output}
    """
    file_path = f"{wrapper_path}/{wrapper_name}"
    with open(file_path, 'w') as file:
        file.write(code_to_write)

    return


def print_progress(info):
    # TODO: polish for ending and MPI rank; example: Pprogress package
    print()
    print(info, end="")
    sys.stdout.write('\x1b[1A')
    sys.stdout.flush()
