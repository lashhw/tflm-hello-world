#include <cstdio>
#include "model.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

int main() {
    //RegisterDebugLogCallback(debug_log_printf);
    MicroPrintf("started");

    tflite::InitializeTarget();

    const tflite::Model* model = tflite::GetModel(model_tflite);
    TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

    tflite::MicroMutableOpResolver<1> op_resolver;
    TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());

    constexpr int kTensorArenaSize = 3000;
    uint8_t tensor_arena[kTensorArenaSize];

    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
    TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

    TfLiteTensor* input = interpreter.input(0);
    TFLITE_CHECK_NE(input, nullptr);

    TfLiteTensor* output = interpreter.output(0);
    TFLITE_CHECK_NE(output, nullptr);

    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;
    MicroPrintf("input_scale = %f", input_scale);
    MicroPrintf("input_zero_point = %d", input_zero_point);

    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;
    MicroPrintf("output_scale = %f", output_scale);
    MicroPrintf("output_zero_point = %d", output_zero_point);

    // Check if the predicted output is within a small range of the
    // expected output
    float epsilon = 0.05f;
    constexpr int kNumTestValues = 4;
    float golden_inputs[kNumTestValues] = {0.77, 1.57, 2.3, 3.14};

    for (int i = 0; i < kNumTestValues; ++i) {
        input->data.int8[0] = (golden_inputs[i] / input_scale) + input_zero_point;
        MicroPrintf("input = %d", input->data.int8[0]);
        TF_LITE_ENSURE_STATUS(interpreter.Invoke());
        MicroPrintf("output = %d", output->data.int8[0]);
        float y_pred = (output->data.int8[0] - output_zero_point) * output_scale;
        MicroPrintf("y_pred = %f", y_pred);
        float y_gold = sin(golden_inputs[i]);
        MicroPrintf("y_gold = %f", y_gold);
        float difference = fabs(y_gold - y_pred);
        MicroPrintf("difference = %f", difference);
        TFLITE_CHECK_LE(difference, epsilon);
    }

    MicroPrintf("all correct");
    MicroPrintf("");

    return kTfLiteOk;
}
