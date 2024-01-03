#!/usr/bin/env bash

set -e
set -x

mypy "dataset_builders" "notebooks"
flake8 "dataset_builders" "notebooks" --ignore=E501,W503,E203,E402
black "dataset_builders" "notebooks" --check -l 80
