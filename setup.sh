#!/bin/bash
set -e

git clone https://github.com/tensorflow/tflite-micro.git
uv venv venv
. venv/bin/activate
uv pip install numpy Pillow

TENSORFLOW_ROOT=tflite-micro/
make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile clean TENSORFLOW_ROOT=${TENSORFLOW_ROOT}
make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile third_party_downloads TENSORFLOW_ROOT=${TENSORFLOW_ROOT}
make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile clean TENSORFLOW_ROOT=${TENSORFLOW_ROOT}
make -s -j8 -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile build TENSORFLOW_ROOT=${TENSORFLOW_ROOT}
make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile test TENSORFLOW_ROOT=${TENSORFLOW_ROOT} >test.out 2>test.err
