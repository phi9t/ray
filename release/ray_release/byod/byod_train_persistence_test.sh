#!/bin/bash
# This script is used to build an extra layer on top of the base anyscale/ray image
# to run the train_multinode_persistence test.

set -exo pipefail

uv pip install --system -U torch fsspec s3fs gcsfs pyarrow>=9.0.0 pytest
