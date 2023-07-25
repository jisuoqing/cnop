def update_parameter(file_path, options, new_file_path=None):
    """
    options = {
        "restart": ".true.",
        "checkpointFileNumber": 1,
        "plotFileNumber": 1,
        "tmax": 10,
        }
    update_parameter("flash.par", options, "flash_new.par")
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for option, value in options.items():
        found = False
        for i, line in enumerate(lines):
            if line.startswith(option):
                lines[i] = f"{option} = {value}\n"
                found = True
                break

        if not found:
            raise ValueError(f"Option '{option}' not found in the file.")

    if new_file_path is None:
        new_file_path = file_path
    with open(new_file_path, 'w') as new_file:
        new_file.writelines(lines)

    print(f"Updated file saved as {new_file_path}")