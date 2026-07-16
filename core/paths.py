import os

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATHS = {
    "setting_yaml": os.path.abspath(os.path.join(current_dir, "config", "setting.yaml")),
    "camera_intrinsics": os.path.abspath(os.path.join(current_dir, "config", "camera_intrinsics.yaml")),
    "result_dir": os.path.abspath(os.path.join(current_dir, "result", "result_step2")),
    "plot_dir": os.path.abspath(os.path.join(current_dir, "result", "result_img")),
    "txt_dir": os.path.abspath(os.path.join(current_dir, "result", "result_txt")),
}
