# -*- coding: utf-8 -*-
import struct
import numpy as np

def onehotEncoder(Y, ny):
      return np.eye(ny)[Y]

def readBinFile(address):
    file = open(address, 'rb')
    bin_data = file.read()
    file.close()
    
    data_types = ['B', 'b', '', 'h', 'i', 'f', 'd']
    offset = 0
    fmt_magic = '>2x2B'
    data_type_idx, dims_size = struct.unpack_from(fmt_magic, bin_data, offset)
    
    offset += struct.calcsize(fmt_magic)
    fmt_dims = '>' + str(dims_size) + 'i'
    dims = struct.unpack_from(fmt_dims, bin_data, offset)
    
    offset += struct.calcsize(fmt_dims)
    fmt_data = '>' + str(np.prod(dims)) + data_types[data_type_idx - 8]
    data = np.asarray(struct.unpack_from(fmt_data, bin_data, offset)).reshape(dims)
    
    return data