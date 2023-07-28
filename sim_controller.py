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
                break

        if not found:
            raise ValueError(f"Parameter '{param}' not found in the file.")

    if new_file_path is None:
        new_file_path = file_path
    with open(new_file_path, 'w') as new_file:
        new_file.writelines(lines)

    print(f"Updated file saved as {new_file_path}")