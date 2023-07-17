import configparser


def solve(*args, **kwargs):
    # read parameter file
    config = configparser.ConfigParser()
    config.read("config.par")

    # Get solver name
    solver_name = config.get("Solver", "name")
    if solver_name == "burgers":
        function_module = __import__("burgers_lib")
        function = getattr(function_module, "solve_burgers")
    elif solver_name == "flash":
        function = flash
    else:
        raise ValueError("Unknown solver name: {}".format(solver_name))

    return function(*args, **kwargs)


def flash(*args, **kwargs):
    pass
