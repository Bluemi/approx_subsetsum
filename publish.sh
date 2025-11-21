#!/bin/bash

set -e

read -p "have you tested with BuildAndTest-Workflow? y/[n]: " answer
if [ "$answer" != "y" ]; then
	exit 0
fi

# get old version from pypi
old_version=$(grep -oP '^version = "[\d\.]+"' pyproject.toml | grep -oP '(?<=version = ")[\d\.]+(?=")')

echo "old version: $old_version"
read -p 'new version> ' new_version

# compare versions
# if [ "$old_version" = "$new_version" ]; then
# 	echo "ERROR: current version equals old version"
# 	exit 0
# fi

# publish package
git add -A && git commit -m "v$new_version"
git push

# publish tag
git tag -a "v$new_version" -m "v$new_version"
git push origin --tags

