import os
from .utils import compile_source

import kp
import numpy as np
import logging

class ComputeInstance(shader):

    def __init__(self, device:int):
        self.mgr = kp.Manager(device)

    def run(self, shader, params:np.ndarray, types:list[np.dtype], consts:np.ndarray):

        tensors = []

        for vec, t in zip(params, types):
            tensors += [mgr.tensor_t(vec, dtype=t)]

        if type(shader) == str:
            shader_f = open(shader + ".spv", "rb").read()
        else:
            shader_f = shader.to_spirv()

        algo = self.mgr.algorithm(tensors=tensors, spirv=shader_f, spec_consts=consts, push_consts=consts)
        sq = mgr.sequence()
        sq.record(kp.OpTensorSyncDevice(tensors))
        sq.record(kp.OpAlgoDispatch(algo))
        sq.eval()

        sq.eval_async(kp.OpTensorSyncLocal(tensors))
        sq.eval_await()

        return tensors


