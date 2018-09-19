import numpy as np

TAG_CHAR = np.array([202021.25], np.float32)
TAG_FLOAT = 202021.25

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

def writeFlowJPEG(filename, flow):

    import cv2
    import cortex.vision.flow

    # Flow has shape [h,w,2]
    # Convert to range [0,1] and save min/max values for denormalization
    flow_u_norm, min_u, max_u = cortex.vision.flow.normalize_flow(flow[:,:,0])
    flow_v_norm, min_v, max_v = cortex.vision.flow.normalize_flow(flow[:,:,1])

    #print("[Flow-X] Min  = {:.3f}, Max = {:.3f}".format(min_u, max_u))
    #print("[Flow-Y] Min  = {:.3f}, Max = {:.3f}".format(min_v, max_v))

    # Write JPG image to disk
    flow_as_jpg = np.dstack((flow_u_norm, flow_v_norm, np.zeros_like(flow_u_norm)))
    flow_as_jpg = (flow_as_jpg*255.0).astype(np.uint8)
    cv2.imwrite(filename, flow_as_jpg)

    return min_u, max_u, min_v, max_v


##########################################################################################

def read_flo_file(file):
    # For EpicFlow
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    #if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    data = np.fromfile(f, np.float32, count=int(2*w*h))
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow

if __name__ == "__main__":

    import glob
    import os
    import cv2
    import cortex.utils
    import cortex.vision.flow

    dir_with_flo_files = '/home/tomrunia/dev/lib/flownet2-pytorch/work/inference/run.epoch-0-flow-field'
    output_dir = '/home/tomrunia/data/RepMeasureDataset/flow/'

    files = glob.glob(os.path.join(dir_with_flo_files, '*.flo'))
    files.sort()

    for i, file in enumerate(files):
        flow_uv = readFlow(file)
        print(i, 'flow_uv', flow_uv.min(), flow_uv.max())
        flow_color = cortex.vision.flow.flow_to_color(flow_uv)
        cv2.imshow('flow', flow_color)
        cv2.waitKey(0)

        #output_filename = os.path.join(output_dir, cortex.utils.basename(file) + '.jpg')
        #cv2.imwrite(output_filename, flow_color)




