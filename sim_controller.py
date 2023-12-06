import os
import glob
import h5py
import warnings


def update_parameter(file_path, params, new_file_path=None):
    """
    params = {
        "restart": ".true.",
        "checkpointFileNumber": 1,
        "plotFileNumber": 1,
        "tmax": 10,
        }
    update_parameter("flash.par", params, "flash_new.par")
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for param, value in params.items():
        found = False
        for i, line in enumerate(lines):
            if line.startswith(param):
                lines[i] = f"{param} = {value}\n"
                found = True

        if not found:
            raise ValueError(f"Parameter '{param}' not found in the file.")

    if new_file_path is None:
        new_file_path = file_path
    with open(new_file_path, 'w') as new_file:
        new_file.writelines(lines)

    # print(f"Updated file saved as {new_file_path}")


def save_checkpoint(process, method):
    iter0 = method.iter0
    checkpoint_fn = "%s_checkpoint_%04d.h5" % (process.__class__.__name__, iter0)
    with h5py.File(process.base_dir + "/" + checkpoint_fn, 'w') as f:
        process_group = f.create_group('process')
        for k, v in process.__dict__.items():
            # Don't save flags related to restart controller, otherwise the restart loop won't work
            # Do not save mpi or yt_derived_field function as well
            if k.startswith("restart") or k.startswith("mpi") or k == "yt_derived_fields":
                continue
            try:
                if v is None:
                    # If the value is None, use a placeholder string
                    process_group.create_dataset(k, data="None")
                else:
                    process_group.create_dataset(k, data=v)
            except TypeError:
                warnings.warn("The process attribute {} cannot be saved!".format(k))
        method_group = f.create_group('method')
        for k, v in method.__dict__.items():
            # Do not save mpi
            if k.startswith("mpi"):
                continue
            try:
                method_group.create_dataset(k, data=v)
            except TypeError:
                warnings.warn("The method attribute {} cannot be saved!".format(k))
    return


def find_latest_checkpoint(base_dir, prefix):
    """
    Find the latest checkpoint file in the directory
    :param base_dir: path
    :param prefix: prefix of the checkpoint file
    :return: latest checkpoint file path
    """

    # Find all checkpoint files
    chk_files = glob.glob(os.path.join(base_dir, prefix + "_*"))
    if len(chk_files) == 0:
        raise FileNotFoundError("No checkpoint file found.")
    # Find the latest checkpoint file
    latest_chk_file = max(chk_files, key=lambda fn: int(fn.split('_')[-1].split('.h5')[0]))
    return latest_chk_file


def load_checkpoint(file_path, group_name, handle):
    if group_name not in ["process", "method"]:
        raise ValueError("group_name must be either 'process' or 'method'.")
    group_data = load_h5_data(file_path)[group_name]
    for key in group_data.keys():
        # print(key, group_data[key])
        setattr(handle, key, group_data[key])
    return handle


def load_h5_data(file_path):
    with h5py.File(file_path, 'r') as hf:
        data_dict = {}
        for key in hf.keys():
            if isinstance(hf[key], h5py.Group):
                # If it's a Group type, recursively load data from the group
                data_dict[key] = load_h5_data_from_group(hf[key])
            elif isinstance(hf[key], h5py.Dataset):
                # If it's a byte string, decode it to a Unicode string
                if hf[key].dtype.kind in ['O', 'S']:
                    data_dict[key] = hf[key][()].decode('utf-8')
                else:
                    # For other data types, load the data directly
                    data_dict[key] = hf[key][()]
        return data_dict


def load_h5_data_from_group(group):
    data_dict = {}
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            # If it's a Group type, recursively load data from the group
            data_dict[key] = load_h5_data_from_group(group[key])
        elif isinstance(group[key], h5py.Dataset):
            # Check if the data is an object type (dtype 'O')
            if group[key].dtype.kind == 'O':
                # If it's an object type, check if it's a single string or a list
                if group[key].shape == ():
                    if group[key][()].decode('utf-8') == "None":
                        data_dict[key] = None
                    else:
                        data_dict[key] = group[key][()].decode('utf-8')
                else:
                    # If it's a list of bytes, decode each item
                    data_dict[key] = [item.decode('utf-8') for item in group[key][()]]
            elif group[key].dtype.kind in ['S', 'U']:
                # If it's a byte or unicode string, decode it to a Unicode string
                data_dict[key] = group[key][()].decode('utf-8')
            else:
                # For other data types, load the data directly
                data_dict[key] = group[key][()]
    return data_dict
