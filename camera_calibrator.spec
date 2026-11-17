# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from pathlib import Path
import platform

datas_rby, binaries_rby, hiddenimports_rby = collect_all('rby1_sdk')

# img/: only the app's own pictures; img/wizard_ui (manual screenshots) stays out of the executable.
# The ChArUco board PDF sits at the bundle root, where the wizard's "open the board file" button looks.
datas = [
    ('config', 'config'),
    ('calib.io_charuco_297x210_5x8_36_27_DICT_5X5.pdf', '.'),
] + [(str(p), 'img') for p in sorted(Path('img').iterdir()) if p.is_file()] + datas_rby

binaries = [] + binaries_rby
hiddenimports = [
    'osqp.ext_builtin',
    'mpl_toolkits.mplot3d',
    'mpl_toolkits.mplot3d.axes3d',
    'rby1_sdk',
    'rby1_sdk._bindings',
    'rby1_sdk._robot_command',
    'rby1_sdk.dynamics',
    'rby1_sdk.math',
    'rby1_sdk.upc',
] + hiddenimports_rby

a = Analysis(
    ['main_ui.py'],
    # No extra pathex needed: core/ and ui/ are real packages imported as
    # core.xxx / ui.xxx from the project root, which is already on sys.path
    # as main_ui.py's own directory. The old ['core', 'core/calibration']
    # entries were a leftover from when core/ contents were flat top-level
    # modules; keeping them risks resolving e.g. calibration_optimizer as
    # both a bare top-level module and core.calibration.calibration_optimizer.
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

arch = platform.machine()  # 'x86_64' (Linux) or 'AMD64' (Windows) or 'aarch64' (Jetson)
exe_name = f'camera_calibrator_{arch}'

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=exe_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

