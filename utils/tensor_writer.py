
import logging

logger = logging.getLogger(__name__)


class Tensor_Writer():
    writer = None
    first_call = True

    @staticmethod
    def set_writer(writer):
        Tensor_Writer.writer = writer

    @staticmethod
    def add_scalar(name, scalar, step):
        if (Tensor_Writer.writer is None):
            if (Tensor_Writer.first_call):
                logger.error("writer has not been set for tensorboard! tensorboard is diabled")
                Tensor_Writer.first_call = True
        else:
            Tensor_Writer.writer.add_scalar(name, scalar, step)