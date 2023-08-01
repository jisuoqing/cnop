import psutil


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

    # print(f"Updated file saved as {new_file_path}")


# Function to get CPU information
def get_cpu_info():
    cpu_count = psutil.cpu_count(logical=False)  # Get the number of physical cores
    cpu_percent = psutil.cpu_percent(interval=1)  # Get the CPU usage percentage
    cpu_freq = get_cpu_frequency()  # Get the CPU frequency using platform-specific method
    return cpu_count, cpu_percent, cpu_freq


# Function to get CPU frequency
def get_cpu_frequency():
    try:
        freq_info = psutil.cpu_freq()  # Get CPU frequency using psutil
        current_freq = freq_info.current if freq_info else "N/A"
        min_freq = freq_info.min if freq_info else "N/A"
        max_freq = freq_info.max if freq_info else "N/A"
    except Exception as e:
        current_freq = min_freq = max_freq = "N/A"
    return {
        "current": current_freq,
        "min": min_freq,
        "max": max_freq
    }


# Function to get memory information
def get_memory_info():
    memory = psutil.virtual_memory()  # Get memory usage
    swap = psutil.swap_memory()  # Get swap memory usage
    return memory, swap


# Function to get disk information
def get_disk_info():
    disk = psutil.disk_usage('/')  # Get disk usage of the root directory
    return disk


def get_system_info():
    # Get and print CPU information
    cpu_count, cpu_percent, cpu_freq = get_cpu_info()
    print(f"CPU核心数: {cpu_count}")
    print(f"CPU使用率: {cpu_percent}%")
    print(f"CPU频率: 当前频率={cpu_freq['current']}MHz, 最小频率={cpu_freq['min']}MHz, 最大频率={cpu_freq['max']}MHz")

    # Get and print memory information
    memory, swap = get_memory_info()
    print(f"总内存: {memory.total / (1024 ** 3):.2f}GB")
    print(f"已使用内存: {memory.used / (1024 ** 3):.2f}GB")
    print(f"可用内存: {memory.available / (1024 ** 3):.2f}GB")
    print(f"内存使用率: {memory.percent}%")
    print(f"交换内存使用率: {swap.percent}%")

    # Get and print disk information
    disk = get_disk_info()
    print(f"总磁盘空间: {disk.total / (1024 ** 3):.2f}GB")
    print(f"已使用磁盘空间: {disk.used / (1024 ** 3):.2f}GB")
    print(f"可用磁盘空间: {disk.free / (1024 ** 3):.2f}GB")
    print(f"磁盘使用率: {disk.percent}%")