# SignLingo.spec - recipe PyInstaller uses to build the app (run ./build_app.sh)
import sys
from PyInstaller.utils.hooks import collect_all

mp_datas, mp_binaries, mp_hidden = collect_all("mediapipe")

a = Analysis(
    ["main.py"],
    datas=[("hand_landmarker.task", "."), ("data/*.json", "data")] + mp_datas,
    binaries=mp_binaries,
    hiddenimports=["practice", "features", "motion_recognition"] + mp_hidden,
    excludes=["gradio", "pandas"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SignLingo", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="SignLingo")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="SignLingo.app",
        bundle_identifier="com.signlingo.app",
        info_plist={
            # macOS only lets an app use the camera if it says why
            "NSCameraUsageDescription": "SignLingo uses the camera to see your hand signs.",
        },
    )
