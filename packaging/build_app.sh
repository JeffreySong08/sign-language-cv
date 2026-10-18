#!/bin/bash
# Builds SignLingo into an app you can double-click and share.
# Run inside the project's .venv:  ./packaging/build_app.sh
set -e
cd "$(dirname "$0")/.."   # project folder, so dist/ and the zip land there
pip install pyinstaller
pyinstaller --noconfirm packaging/SignLingo.spec
if [ -d dist/SignLingo.app ]; then
    # Zip it for sharing (ditto keeps the Mac app intact)
    ditto -c -k --keepParent dist/SignLingo.app SignLingo-mac.zip
    echo "Done: dist/SignLingo.app and SignLingo-mac.zip"
else
    echo "Done: dist/SignLingo"
fi
