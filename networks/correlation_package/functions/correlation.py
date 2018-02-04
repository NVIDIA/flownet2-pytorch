from torch.autograd import Function, Variable
from .._ext import correlation


class CorrelationFunction(Function):

    @staticmethod
    def forward(ctx,
                input1,
                input2,
                pad_size=3,
                kernel_size=3,
                max_displacement=20,
                stride1=1,
                stride2=2,
                corr_multiply=1):
        assert input1.is_contiguous()
        assert input2.is_contiguous()

        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        rbot1 = input1.new()
        rbot2 = input2.new()
        output = input1.new()

        correlation.Correlation_forward_cuda(
            input1, input2, rbot1, rbot2, output, pad_size, kernel_size,
            max_displacement, stride1, stride2, corr_multiply)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_contiguous()

        input1, input2 = ctx.saved_tensors

        rbot1 = input1.new()
        rbot2 = input2.new()

        grad_input1 = Variable(input1.new())
        grad_input2 = Variable(input2.new())

        correlation.Correlation_backward_cuda(
            input1, input2, rbot1, rbot2, grad_output.data, grad_input1.data,
            grad_input2.data, ctx.pad_size, ctx.kernel_size,
            ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)

        return (grad_input1, grad_input2) + (None, ) * 6
