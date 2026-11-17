from setuptools import setup, find_packages

# Package layout: core/ and ui/ are real packages at the project root
# (imports throughout the codebase are `core.xxx`, `core.calibration.xxx`,
# `ui.xxx`), so no package_dir remapping and no py_modules list is needed.
# The old py_modules entries (marker_detection, robot_motion, paths,
# language, i18n, wizard_widget) referred to modules that no longer exist
# at that location: robot_motion -> core/robot/motion.py, paths -> core/storage.py,
# wizard_widget -> ui/wizard_widget.py, i18n was removed, marker_detection and
# language now live under core/ as core.marker_detection / core.language.
setup(
    name="camera_ws",
    version="0.1",
    packages=find_packages(include=["core", "core.*", "ui", "ui.*"]),
    # 3.12 only (2026-10-26): the one version with rby1_sdk, pyrealsense2 and osqp wheels on Windows,
    # Linux x86_64 and Jetson aarch64 (pyrealsense2 has no aarch64 wheel for 3.11).
    python_requires=">=3.12,<3.13",
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
    install_requires=[
        "typeguard",
        "numpy>=2.0.1",
        "opencv-contrib-python-headless==4.13.0.92",
        "rby1_sdk>=0.10.0",
        "scipy>=1.14.0",
        "matplotlib>=3.9.1",
        "pyrealsense2",
        "PySide6>=6.7.2",
        "qpsolvers[osqp]==4.11.0",
        "osqp==1.1.1",
        "pyyaml",
    ],
)