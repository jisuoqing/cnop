import numpy as np
import os
import time
import shutil


def do_projection(u, delta=8e-4):
    sum0 = (u ** 2.).sum()
    if np.sqrt(sum0) <= delta:
        proj_u = u
    else:
        proj_u = delta / np.sqrt(sum0) * u
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
            # then, check if the file is modified with the for the polling interval
            being_modified = True
            while being_modified:
                current_mtime = os.path.getmtime(file_path)
                time.sleep(poll_interval)
                if current_mtime == os.path.getmtime(file_path):
                    being_modified = False
            return True
        else:
            time.sleep(poll_interval)

    return False


def generate_python_wrapper(exec_command, wrapper_name="wrapper.py", wrapper_path=None):
    """
    Generate a wrapper script for running the executable
    so that the stdout and stderr can be redirected to files when using MPI.COMM_SELF.Spawn
    :param exec_command: name of the executable
    :param wrapper_name:
    :param wrapper_path: path of the wrapper script
    :return: None
    """
    if wrapper_path is None:
        wrapper_path = os.getcwd()
    code_to_write = f"""import subprocess
with open("stdout.txt", "w") as stdout, open("stderr.txt", "w") as stderr:
    p = subprocess.Popen(["{exec_command}"], shell=True, stdout=stdout, stderr=stderr)
    p.wait()
    """
    file_path = f"{wrapper_path}/{wrapper_name}"
    with open(file_path, 'w') as file:
        file.write(code_to_write)

    return


def wait_until_deleted(path, timeout=60., poll_interval=1.):
    """
    Remove a directory tree with multiple trials
    :param path: path of the directory tree
    :param timeout: timeout in seconds
    :param poll_interval: poll interval in seconds
    :return: True if the directory tree is removed before timeout, False otherwise
    """
    delete_success = False
    start_time = time.time()
    while not delete_success and time.time() - start_time < timeout:
        try:
            shutil.rmtree(path)
            delete_success = True
        except FileNotFoundError:
            delete_success = True
        except OSError:
            delete_success = False
            time.sleep(poll_interval)
    return delete_success
