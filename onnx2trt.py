
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import common
TRT_LOGGER = trt.Logger()


def build_engine(onnx_file_path):
    builder = trt.Builder(TRT_LOGGER)
    builder.max_workspace_size = 1 << 30
    builder.max_batch_size = 2
    #config = builder.create_builder_config()
    #config.max_workspace_size = common.GiB(1)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    # network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
        print('Completed parsing of ONNX file')
    builder.max_workspace_size = 1 << 30
    builder.max_batch_size = 2

    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
    #last_layer = network.get_layer(network.num_layers - 1)
    #network.mark_output(last_layer.get_output(0))
    #print('Building a serialized network . . .')
    #serialized_engine = builder.build_serialized_network(network, config)
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    with open("yolov5ncrop.engine", "wb") as f:
        f.write(engine)

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
    # serialized_engine = builder.build_serialized_network(network, config)
    stream = cuda.Stream()
    # host_input = np.array(preprocess_image("turkish_coffee.jpg").numpy(), dtype=np.float32, order='C')
    # cuda.memcpy_htod_async(device_input, host_input, stream)
    # context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    # cuda.memcpy_dtoh_async(host_output, device_output, stream)
    # stream.synchronize()
    # output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
    # postprocess(output_data)
if __name__ == "__main__":
   main()