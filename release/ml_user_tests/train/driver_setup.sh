#!/bin/bash

cd "${0%/*}" || exit 1

uv pip install --system -U -r ./driver_requirements.txt
