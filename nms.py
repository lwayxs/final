import time
import numpy as np


def split_np_data(Np):
    box=[]
    cls=[]
    #mask=np.empty(shape=(Np.shape[0],0),dtype='float32')
    mask=np.empty(shape=(Np.shape[0],0),dtype='float32')
    for i in range(Np.shape[0]):
        box.append(list(Np[i][:4]))
        cls.append(list(Np[i][4:]))
    box = np.asarray(box)
    cls = np.asarray(cls)
    #box = np.array(box)
    #cls = np.array(cls)
    return box ,cls,mask

def py_cpu_nms(dets,scores, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    res = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]
        res.append(i)
        # x11 = np.maximum(x1[i], x1[index[1:]])
        # y11 = np.maximum(y1[i], y1[index[1:]])
        # x22 = np.minimum(x2[i], x2[index[1:]])
        # y22 = np.minimum(y2[i], y2[index[1:]])

        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        # w = np.maximum(0, x22 - x11 + 1)
        # h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        iou = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        #idx = np.where(iou <= thresh)[0]
        idx = np.where(iou <= thresh)[0]
        index = index[idx + 1]
    return res

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=300,
        nm=0,  # number of masks
):

    agnostic = False
    multi_label = False
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[1] - nm - 4  # number of classes
    mi = 4 + nc  # mask start index
    #xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    xc = np.amax(prediction[:, 4:mi],axis=1)> conf_thres

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    #output = [np.empty(shape=(0, 6 + nm),dtype='float32')] * bs
    output = [np.empty(shape=(0, 6 + nm),dtype='float32')] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x.T[xc[xi]]
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = split_np_data(x)
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        conf = cls
        j=np.array([[0]]*x.shape[0], dtype='float32')
        x = np.concatenate((box, conf, j, mask), 1)[conf.max(1) > conf_thres]

        #j = np.array([[0]] * x.shape[0], dtype='float32')
        #x = np.concatenate((box, conf, j, mask), 1)[conf.max(1) > conf_thres]


        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[np.flipud(np.argsort(x[:, 4]))[:max_nms]]  # sort by confidence and remove excess boxes
        #x = x[np.flipud(np.argsort(x[:, 4]))[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = py_cpu_nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def xywh2xyxy(x):

    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def clip_boxes(boxes, shape):


    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):

    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes
