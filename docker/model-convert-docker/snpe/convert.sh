#!/bin/sh

source ${SNPE_ROOT}/bin/envsetup.sh -o ${ONNX_DIR}
python3 /tools/convert.py $@
