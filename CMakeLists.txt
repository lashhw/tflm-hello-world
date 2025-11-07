cmake_minimum_required(VERSION 3.25)

project(tflm)
add_executable(tflm main.cpp)

target_include_directories(tflm PRIVATE
    tflite-micro
    tflite-micro/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
    tflite-micro/tensorflow/lite/micro/tools/make/downloads/gemmlowp
)

target_compile_definitions(tflm PRIVATE
    TF_LITE_STATIC_MEMORY
    TF_LITE_DISABLE_X86_NEON
    TF_LITE_USE_CTIME
)

target_link_libraries(tflm PRIVATE
    ${PROJECT_SOURCE_DIR}/tflite-micro/gen/linux_x86_64_default/lib/libtensorflow-microlite.a
)