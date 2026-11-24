# Operator coverage

This reference describes the internal static-graph representation. Users supply Python models; they do not need to write these operations directly. All graph inputs have fixed shapes, maximum rank eight and a safety limit of 2²⁵ elements per tensor. The target's actual RAM limit can be much smaller.

| Group | Operations | Restrictions |
|---|---|---|
| Dense computation | Gemm, MatMul | Gemm rank two; MatMul rank two or higher with batch broadcasting |
| Convolution | Conv | One or two spatial dimensions; groups, strides, dilation, explicit or SAME padding |
| Pooling | MaxPool, AveragePool, GlobalAveragePool | 1D or 2D local pooling; no ceil mode; one output for MaxPool |
| Normalization | BatchNormalization, LayerNormalization | Evaluation mode; one output; LayerNorm across a suffix of dimensions |
| Activations | Relu, LeakyRelu, Sigmoid, Tanh, Elu, Softplus, Clip | Clip bounds constant |
| Math | Add, Sub, Mul, Div, Abs, Neg, Exp, Log, Sqrt, Erf | Standard broadcasting for binary operations; finite supported computations |
| Powers and extrema | Pow, Max, Min | Pow exponent 2, 0.5 or -1; Max and Min two operands |
| Distribution | Softmax, LogSoftmax | Any fixed axis in the exported opset 17 graph |
| Reductions | ReduceSum, ReduceMean, ReduceMax | Constant axes, keepdims supported |
| Shape and layout | Identity, Flatten, Reshape, Squeeze, Unsqueeze, Transpose, Expand | Static output shapes; fixed reshape and expansion targets |
| Joining and indexing | Concat, Split, Slice, Gather | Slice bounds and steps constant; Gather bounds checked at runtime |
| Padding | Pad | Constant fill; negative padding crops; full-rank pads |
| Conditions | Where, Equal, Greater, Less, GreaterOrEqual, LessOrEqual, And, Or | Boolean intermediates represented as zero or one |
| Conversion | Cast | float32, int32, int64 or bool targets; exact runtime integers limited to ±2²⁴ |
| Inference cleanup | Dropout | Training disabled; one output |
| Compile-time folding | Constant, Shape and constant-only subgraphs | Evaluated during conversion using the graph reference evaluator |

Some Python layers export into combinations of operations above. For example, GELU can export as Erf plus arithmetic, and embeddings export as Gather. Coverage is defined by actual exported operations and attributes, not the Python class name.

No support is claimed for arbitrary models, dynamic sequence lengths, data-dependent control flow, recurrent loops, sparse tensors, custom kernels, all attention implementations, all pooling attributes, general resize modes, or training. Unsupported graphs stop with an operator-specific error. Tested cases are listed in `validation/`; an implemented branch without a listed case has less validation than an exercised branch.

Integer arithmetic is not a general int64 implementation. Shape integers are handled at compile time; runtime integer data is converted to exactly representable FP32 values within the stated range. Models needing wider integers or exact integer division need another backend.
