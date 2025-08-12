python export.py 

trtexec \
--onnx=onnx_export/PE-Core-L14-336/PE-Core-L14-336_vision.onnx \
--saveEngine=trt_export/PE-Core-L14-336/PE-Core-L14-336_vision.engine \
--minShapes=image:1x3x336x336 \
--optShapes=image:4x3x336x336 \
--maxShapes=image:16x3x336x336