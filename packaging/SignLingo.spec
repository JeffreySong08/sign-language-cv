# SignLingo.spec - recipe PyInstaller uses to build the app (run ./packaging/build_app.sh)
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

root = Path(SPECPATH).parent  # this file lives in packaging/
mp_datas, mp_binaries, mp_hidden = collect_all("mediapipe")

a = Analysis(
    [str(root / "main.py")],
    pathex=[str(root)],
    # Same layout as the project, so signlingo/paths.py finds them in the app too
    datas=[(str(root / "models" / "hand_landmarker.task"), "models"),
           (str(root / "data" / "*.json"), "data")] + mp_datas,
    binaries=mp_binaries,
    # main.py starts the modes by name, so PyInstaller can't see them on its own
    hiddenimports=["signlingo.practice", "signlingo.words"] + mp_hidden,
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
