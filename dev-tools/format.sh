#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place "dataset_builders" "notebooks" --exclude=__init__.py
isort "dataset_builders" "notebooks"
black "dataset_builders" "notebooks" -l 80
