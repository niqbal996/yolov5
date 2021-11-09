import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger()


def build_engine(onnx_file_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
        print('Completed parsing of ONNX file')
    builder.max_workspace_size = 1 << 30
    builder.max_batch_size = 2
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
        print('Building an engine...')
        engine = builder.build_cuda_engine(network)
        context = engine.create_execution_context()
        print("Completed creating Engine")
        return engine, context


def main():
    engine, context = build_engine(onnx_file_path="./yolov5ncrop.onnx")
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)
    stream = cuda.Stream()
    # host_input = np.array(preprocess_image("turkish_coffee.jpg").numpy(), dtype=np.float32, order='C')
    # cuda.memcpy_htod_async(device_input, host_input, stream)
    # context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    # cuda.memcpy_dtoh_async(host_output, device_output, stream)
    # stream.synchronize()
    # output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
    # postprocess(output_data)
