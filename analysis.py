from sim_controller import load_h5_data
import numpy as np
import yt
import os
import warnings


def load_cnop(filename, solution_file=None, bbox=None):
    f = load_h5_data(filename)

    if solution_file is not None:
        ds_solution = yt.load(solution_file)
        dims = ds_solution.domain_dimensions
        bbox = np.array([ds_solution.domain_left_edge, ds_solution.domain_right_edge]).T
        warnings.warn("Solution file is provided, bbox is ignored.")
    else:
        dims = f["method"]["u_pert_best"].shape
        if bbox is None:
            bbox = np.array([[0., 1.], [0., 1.], [0., 1.]])

    data_dict, parameters_dict = {}, {}
    load_group_data(f, "process", dims, data_dict, parameters_dict)
    load_group_data(f, "method", dims, data_dict, parameters_dict)

    sim_time = 0
    path, dataset_name = os.path.split(filename)
    ds = yt.load_uniform_grid(data_dict, dims, bbox=bbox, sim_time=sim_time, geometry="cartesian",
                              dataset_name=dataset_name)
    ds.parameters = parameters_dict
    return ds


def load_group_data(file, group, dims, data_dict, parameter_dict):
    for field in file[group].keys():
        field_unit = "dimensionless"

        try:
            if np.array_equal(file[group][field][...].shape, dims):
                field_data = np.array(file[group][field])  # np.transpose(np.array(f["method"][field]))[..., 0]
                data_dict[("gas", field)] = (field_data[...], field_unit)
                print("%s field loaded with unit of %s" % (field, field_unit))
            else:
                # Send field_data to parameter_dict if it is not a field with dimension of dims
                raise TypeError
        except TypeError:
            parameter_dict[field] = file[group][field]
    return
