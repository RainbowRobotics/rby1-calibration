"""Filesystem access grouped by purpose; importing this module creates no files."""
import builtins
import json
import os
import shutil
import tempfile
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import RLock
from contextlib import contextmanager

import numpy as np
import yaml


class FileStorage:
    lock = RLock()

    @staticmethod
    def open(path, mode="r", *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and any(c in mode for c in "wax"):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        return builtins.open(path, mode, *args, **kwargs)

    @staticmethod
    def read_text(path, **kwargs):
        return Path(path).read_text(**kwargs)

    @staticmethod
    @contextmanager
    def atomic_writer(path, mode="w", **options):
        """Publish only after a successful write; retain the old file on failure."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileStorage.lock:
            fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
            try:
                with os.fdopen(fd, mode, **options) as stream:
                    yield stream
                    stream.flush()
                    os.fsync(stream.fileno())
                # Windows refuses to replace a file another program holds open (preview pane,
                # an editor, antivirus, the indexer). Those locks are usually brief, so retry
                # before giving up -- a silent failure here looked like the file "rolling back".
                for attempt in range(6):
                    try:
                        os.replace(temporary, path)
                        break
                    except PermissionError:
                        if attempt == 5:
                            raise PermissionError(
                                f"Could not save {path}: another program is holding the file open. "
                                "Close it (including an Explorer preview pane) and try again.")
                        time.sleep(0.15)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    @staticmethod
    def write_text(path, text, encoding="utf-8"):
        with FileStorage.atomic_writer(path, encoding=encoding) as stream:
            stream.write(text)

    @staticmethod
    def copy(source, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        return shutil.copy2(source, destination)

    @staticmethod
    def ensure_dir(path, **kwargs):
        return os.makedirs(path, **kwargs)

    @staticmethod
    def remove(path):
        return os.remove(path)


class ConfigStorage:
    @staticmethod
    def load(path):
        with FileStorage.open(path, encoding="utf-8") as stream:
            return yaml.safe_load(stream) or {}

    @staticmethod
    def save(path, data):
        FileStorage.write_text(path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    @staticmethod
    def update(path, update):
        with FileStorage.lock:
            data = ConfigStorage.load(path)
            update(data)
            ConfigStorage.save(path, data)
        return data

    @staticmethod
    def update_values(path, changes):
        """Set nested keys in a YAML config, keeping everything else as it was.

        `changes` maps a key path to a value: {("camera", "mount_to_cam"): [...]}.

        Parsing the file into a dict and writing it back is safer than patching lines: the
        result is always valid YAML, values keep full precision (the line patcher rounded
        positions to 5 and angles to 2 decimals), and it works for any key shape or depth.
        setting.yaml carries no comments, so nothing is lost by not preserving them.
        """
        path = Path(path)
        with FileStorage.lock:
            text = FileStorage.read_text(path, encoding="utf-8") if path.exists() else ""
            data = yaml.safe_load(text) or {}
            if not isinstance(data, dict):
                raise ValueError(f"{path} is not a YAML mapping; refusing to overwrite it")
            duplicates = ConfigStorage._duplicate_keys(text)
            for keys, value in changes.items():
                keys = (keys,) if isinstance(keys, str) else tuple(keys)
                node = data
                for key in keys[:-1]:
                    child = node.get(key)
                    if not isinstance(child, dict):
                        child = {}
                        node[key] = child
                    node = child
                node[keys[-1]] = value
            FileStorage.write_text(path, ConfigStorage.dump(data))
        return duplicates

    @staticmethod
    def _duplicate_keys(text):
        """Keys written twice at the same indent; a round-trip keeps only the last one."""
        seen, repeated = {}, []
        for raw in text.splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped or stripped.startswith("-"):
                continue
            indent = len(raw) - len(raw.lstrip())
            key = stripped.split(":", 1)[0].strip()
            for known_indent in [i for i in seen if i > indent]:
                seen.pop(known_indent, None)
            bucket = seen.setdefault(indent, set())
            if key in bucket:
                repeated.append(key)
            bucket.add(key)
        return repeated

    @staticmethod
    def dump(data):
        """YAML text that matches how these config files already look: insertion order kept,
        short lists on one line."""
        class _Dumper(yaml.SafeDumper):
            pass
        _Dumper.add_representer(list, lambda d, v: d.represent_sequence(
            "tag:yaml.org,2002:seq", v, flow_style=True))
        return yaml.dump(data, Dumper=_Dumper, default_flow_style=False,
                         sort_keys=False, allow_unicode=True, width=4096)

    @staticmethod
    def update_camera_key_in_lines(lines_list, key_str, new_vals_list):
        cam_idx = -1
        for idx, line in enumerate(lines_list):
            if line.strip().startswith("camera:"):
                cam_idx = idx
                break
        
        new_val_str = f"[{new_vals_list[0]:.5f}, {new_vals_list[1]:.5f}, {new_vals_list[2]:.5f}, {new_vals_list[3]:.2f}, {new_vals_list[4]:.2f}, {new_vals_list[5]:.2f}]"
        key_found = False
        if cam_idx != -1:
            i = cam_idx + 1
            while i < len(lines_list):
                line = lines_list[i]
                stripped = line.strip()
                if not stripped:
                    i += 1
                    continue
                if not line.startswith((" ", "\t")) and not stripped.startswith("#"):
                    break
                
                if stripped.startswith(f"{key_str}:"):
                    comment = ""
                    if "#" in line:
                        comment_idx = line.find("#")
                        comment = " " + line[comment_idx:].rstrip()
                    
                    indent = len(line) - len(line.lstrip())
                    lines_list[i] = " " * indent + f"{key_str}: {new_val_str}{comment}\n"
                    # Clean up any legacy multiline list items (- val) belonging to this key
                    j = i + 1
                    while j < len(lines_list):
                        sub_line = lines_list[j]
                        sub_stripped = sub_line.strip()
                        sub_indent = len(sub_line) - len(sub_line.lstrip())
                        if sub_stripped.startswith("-") and sub_indent > indent:
                            del lines_list[j]
                        else:
                            break
                    key_found = True
                    break
                i += 1
        
        if not key_found:
            if cam_idx == -1:
                lines_list.append("camera:\n")
                lines_list.append(f"  {key_str}: {new_val_str}\n")
            else:
                lines_list.insert(cam_idx + 1, f"  {key_str}: {new_val_str}\n")


    @staticmethod
    def update_marker_key_in_lines(lines_list, key_str, new_vals_list):
        marker_idx = -1
        for idx, line in enumerate(lines_list):
            if line.strip().startswith("marker:"):
                marker_idx = idx
                break
        
        new_val_str = f"[{new_vals_list[0]:.5f}, {new_vals_list[1]:.5f}, {new_vals_list[2]:.5f}, {new_vals_list[3]:.2f}, {new_vals_list[4]:.2f}, {new_vals_list[5]:.2f}]"
        key_found = False
        if marker_idx != -1:
            i = marker_idx + 1
            while i < len(lines_list):
                line = lines_list[i]
                stripped = line.strip()
                if not stripped:
                    i += 1
                    continue
                if not line.startswith((" ", "\t")) and not stripped.startswith("#"):
                    break
                
                if stripped.startswith(f"{key_str}:"):
                    comment = ""
                    if "#" in line:
                        comment_idx = line.find("#")
                        comment = " " + line[comment_idx:].rstrip()
                    
                    indent = len(line) - len(line.lstrip())
                    lines_list[i] = " " * indent + f"{key_str}: {new_val_str}{comment}\n"
                    # Clean up any legacy multiline list items (- val) belonging to this key
                    j = i + 1
                    while j < len(lines_list):
                        sub_line = lines_list[j]
                        sub_stripped = sub_line.strip()
                        sub_indent = len(sub_line) - len(sub_line.lstrip())
                        if sub_stripped.startswith("-") and sub_indent > indent:
                            del lines_list[j]
                        else:
                            break
                    key_found = True
                    break
                i += 1
        
        if not key_found:
            if marker_idx == -1:
                lines_list.append("marker:\n")
                lines_list.append(f"  {key_str}: {new_val_str}\n")
            else:
                lines_list.insert(marker_idx + 1, f"  {key_str}: {new_val_str}\n")



class ResultStorage:
    @staticmethod
    def load(path):
        with FileStorage.open(path, encoding="utf-8") as stream:
            return json.load(stream)

    @staticmethod
    def save(path, data):
        FileStorage.write_text(path, json.dumps(data, indent=4, ensure_ascii=False))


class DatasetStorage:
    @staticmethod
    def load(path):
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}

    @staticmethod
    def save(path, **arrays):
        with FileStorage.atomic_writer(path, mode="wb") as stream:
            np.savez_compressed(stream, **arrays)

    @staticmethod
    def load_calibration(path):
        data = DatasetStorage.load(path)
        return data.get("q_arm", data.get("q")), data.get("q_head"), data["marker"]

    @staticmethod
    def save_calibration(path, q_arm, T_meas, q_head=None):
        arrays = {"q": q_arm, "q_arm": q_arm, "marker": T_meas}
        if q_head is not None:
            arrays["q_head"] = q_head
        DatasetStorage.save(path, **arrays)


class StoragePaths:
    root = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]

    @staticmethod
    def resolve(path):
        path = Path(path).expanduser()
        return path if path.is_absolute() else StoragePaths.root / path

    @staticmethod
    def asset(path):
        base = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else StoragePaths.root
        return str(base / path)

    @staticmethod
    def initialize_defaults():
        """Explicit startup operation; path lookup/import never writes to disk."""
        if not getattr(sys, "frozen", False):
            return
        for source in (Path(sys._MEIPASS) / "config").rglob("*"):
            if source.is_file():
                destination = StoragePaths.root / "config" / source.relative_to(Path(sys._MEIPASS) / "config")
                if not destination.exists():
                    FileStorage.copy(source, destination)


CONFIG_PATHS = {key: str(StoragePaths.root / path) for key, path in {
    "setting_yaml": "config/setting.yaml", "camera_info": "config/camera_info.yaml",
    "ready_poses_yaml": "config/ready_poses.yaml", "camera_intrinsics": "config/camera_intrinsics.yaml",
    "simulation_yaml": "config/simulation.yaml", "ui_config_dir": "config/ui_config",
    "i18n_yaml": "config/ui_config/i18n.yaml", "ui_dropdowns_yaml": "config/ui_config/ui_dropdowns.yaml",
    "dark_theme_qss": "config/ui_config/dark_theme.qss", "result_dir": "result/result_step2",
    "plot_dir": "result/result_img", "txt_dir": "result/result_txt", "log_dir": "result/result_log",
}.items()}


class SessionLog:
    """Everything the GUI log shows, kept on disk: result/result_log/gui_log_<date>.txt.

    The GUI log was the only place some facts appeared (per-arm bracket fits, exposure changes,
    why a sequence stopped) and it was gone when the window closed. Lines are appended with the
    time; the folder is separate from result_txt, whose .txt files a new sweep clears."""

    @staticmethod
    def path(now=None):
        now = now or datetime.now()
        return os.path.join(CONFIG_PATHS["log_dir"], f"gui_log_{now:%Y%m%d}.txt")

    @staticmethod
    def append(message, now=None):
        now = now or datetime.now()
        with FileStorage.lock, FileStorage.open(SessionLog.path(now), "a", encoding="utf-8") as stream:
            stream.write(f"[{now:%H:%M:%S}] {message}\n")


def camera_intrinsics_path(family=None):
    """Path of the working intrinsics file, or of one model family's saved calibration.

    `camera_intrinsics.yaml` is what the detector reads; `camera_intrinsics_<family>.yaml`
    (e.g. `camera_intrinsics_d435.yaml`) is where each camera model's calibration is kept, so
    swapping cameras back and forth does not destroy either one. The family is always the bare
    body number, so a D435i and a D435f share the single `_d435` file.
    """
    working = CONFIG_PATHS["camera_intrinsics"]
    if not family:
        return working
    root, ext = os.path.splitext(working)
    return f"{root}_{str(family).lower()}{ext}"


def camera_intrinsics_serial_path(serial):
    """Path of one camera unit's own intrinsics: config/camera_intrinsics/<serial>.yaml.

    2026-09-30: two units of the same model differ enough that a new camera should get its own
    calibration; a file named after the serial is used before the per-model store. Camera serials
    are allowed in config (user decision); robot serials and robot constants are not.
    """
    if not serial:
        return None
    safe = "".join(ch for ch in str(serial) if ch.isalnum() or ch in "-_")
    if not safe:
        return None
    return os.path.join(os.path.dirname(CONFIG_PATHS["camera_intrinsics"]), "camera_intrinsics", f"{safe}.yaml")


def save_camera_intrinsics(data, family=None, serial=None):
    """Write a calibration to the working file, the model store and (with a serial) the unit's own
    file, which is always overwritten so a recalibration is not undone at the next start.
    Returns the paths written."""
    data = dict(data)
    if family:
        data["device_name"] = family
    if serial:
        data["serial_number"] = str(serial)
    paths = [camera_intrinsics_path()]
    if family:
        paths.append(camera_intrinsics_path(family))
    serial_path = camera_intrinsics_serial_path(serial)
    if serial_path:
        os.makedirs(os.path.dirname(serial_path), exist_ok=True)
        paths.append(serial_path)
    for path in paths:
        ConfigStorage.save(path, data)
    return paths


class ArtifactStorage:
    @staticmethod
    def read_image(path, *args):
        import cv2
        return cv2.imread(str(path), *args)

    @staticmethod
    def save_image(path, image, *args):
        import cv2
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(str(path), image, *args)

    @staticmethod
    def save_figure(path, **kwargs):
        from matplotlib import pyplot
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return pyplot.savefig(path, **kwargs)
