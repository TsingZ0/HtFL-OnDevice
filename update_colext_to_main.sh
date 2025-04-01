#!/bin/bash -l

set -e
mamba activate htfl-ondevice

INSTALL_DIR=$(pip show colext | awk '/^Location/ {print $2}')
VERSION=$(pip show colext | awk '/^Version/ {print $2}')
DIRECT_URL_FILE="$INSTALL_DIR/colext-$VERSION.dist-info/direct_url.json"

INSTALLED_COMMIT=$(jq -r .vcs_info.commit_id "$DIRECT_URL_FILE")
LATEST_COMMIT=$(git ls-remote $(jq -r .url "$DIRECT_URL_FILE") HEAD | awk '{print $1}')
if [ "$INSTALLED_COMMIT" != "$LATEST_COMMIT" ]; then
    echo "CoLExT commit hash does not match latest version."
    echo "Updating CoLExT"
    pip install --force-reinstall git+https://git@github.com/sands-lab/colext.git
else
    echo "CoLExT is up to date with latest version."
fi