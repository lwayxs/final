import cv2
import numpy as np
import onnxruntime as rt
import os

now_path = os.getcwd()
onnxModulePath = os.path.join(now_path,'weights','best.onnx')
smokeModulePath = os.path.join(now_path,'weights','smoke.onnx')

CLASSES = ['closed_eye', 'closed_mouth', 'open_eye', 'open_mouth']
COLORS = [(0,0,255),(255,0,0),(255,0,0),(0,0,255)]


now_path = os.getcwd()
#疲劳警告
warn_path = os.path.join(now_path,'1.jpg')
warn_img = cv2.imread(warn_path)
warn_img = cv2.resize(warn_img,(100,100))


smoke_path = os.path.join(now_path,'2.jpg')
smoke_img = cv2.imread(smoke_path)
smoke_img = cv2.resize(smoke_img,(100,100))

def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (np.min(a2, b2) - np.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding


    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

#加载模型
def onnx_inf(onnxModulePath, data):
    cuda = True
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    sess = rt.InferenceSession(onnxModulePath, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    pred_onnx = sess.run([output_name], {input_name: data.reshape(1, 3, 640, 640)
                         .astype(np.float32)})

    return pred_onnx


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # isinstance 用来判断某个变量是否属于某种类型
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= 0.45)[0]

        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    mi = 5 + nc  # mask start index
    output = [np.zeros((0, 6 + nm))] * bs

    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros(len(lb), nc + nm + 5)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)

        else:  # best class only
            conf = np.max(x[:, 5:mi], 1).reshape(box.shape[:1][0], 1)
            j = np.argmax(x[:, 5:mi], 1).reshape(box.shape[:1][0], 1)
            x = np.concatenate((box, conf, j, mask), 1)[conf.reshape(box.shape[:1][0]) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        index = x[:, 4].argsort(axis=0)[:max_nms][::-1]
        x = x[index]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms_boxes(boxes, scores)
        i = i[:max_det]  # limit detections

        # 用来合并框的
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = np.multiply(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

    return output


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)

    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
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



#数据前处理 将数据转为模型需要的尺寸
def proposscess(img):
    img0 = img.copy()


    img, ratio, dwdh = letterbox(img, auto=False)
    img = img.transpose((2, 0, 1))[::-1]
    img = np.expand_dims(img, 0)
    img = np.ascontiguousarray(img)

    im = img.astype(np.float32)
    im /= 255

    return img0,im

#后处理 将模型推理结果转为坐标
def postprocess(pred, img0, shape):

        #nms后处理 删减掉重复目标
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        # Process predictions
        seen = 0
        for i, det in enumerate(pred):  # per image
            seen += 1
            if len(det):
                # Rescale boxes from img_size to im0 size
                #将坐标框按照原始物体尺寸复原
                det[:, :4] = scale_boxes((640, 640), det[:, :4], shape).round()
        # print(pred)
        outputs = pred[0][:, :6]



        #将坐标 类别可视化在图片上
        if len(outputs[:, 4:] > 0):
            for i in outputs:
                prob = i[4]
                cls = int(i[5])
                if cls==0 or cls==3:
                    img0[0:100, 300:400] = warn_img
                all_pred_boxes = i[:4]
                x1 = int(all_pred_boxes[0])
                y1 = int(all_pred_boxes[1])
                x2 = int(all_pred_boxes[2])
                y2 = int(all_pred_boxes[3])
                cv2.rectangle(img0, (x1, y1), (x2, y2), COLORS[cls], 1)
                # cv2.putText(img0, CLASSES[cls] + ' ' + str(prob), (x1, y1), cv2.FONT_HERSHEY_TRIPLEX, 1,
                #             COLORS[cls], 1, 4)


        return img0


def postprocess_smoke(pred_smoke, img0, shape):
    # nms后处理 删减掉重复目标
    pred_smoke = non_max_suppression(pred_smoke, 0.25, 0.45, None, False, max_det=1000)
    # Process predictions


    seen_m = 0
    for i, det in enumerate(pred_smoke):  # per image
        seen_m += 1
        if len(det):
            # Rescale boxes from img_size to im0 size
            # 将坐标框按照原始物体尺寸复原
            det[:, :4] = scale_boxes((640, 640), det[:, :4], shape).round()
    # print(pred)
    outputs_smoke = pred_smoke[0][:, :6]


    if len(outputs_smoke[:, 4:] > 0):
        for i in outputs_smoke:
            prob = i[4]
            if prob>=0.45:

                img0[-100:, 300:400] = smoke_img
                all_pred_boxes = i[:4]
                x1 = int(all_pred_boxes[0])
                y1 = int(all_pred_boxes[1])
                x2 = int(all_pred_boxes[2])
                y2 = int(all_pred_boxes[3])
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return img0


#模型推理
def infer(img):
    shape = img.shape
    #数据前处理 将数据转为模型需要的尺寸
    img0,im=proposscess(img)

    #模型推理
    # inference
    pred = onnx_inf(onnxModulePath, im)

    #后处理，将推理结果转换为目标坐标
    output = postprocess(pred, img0, shape)



    return output

def smoke_infer(img):
    shape = img.shape
    #数据前处理 将数据转为模型需要的尺寸
    img0,im=proposscess(img)

    #模型推理
    # inference
    pred_smoke = onnx_inf(smokeModulePath,im)

    #后处理，将推理结果转换为目标坐标
    output = postprocess_smoke(pred_smoke, img0, shape)



    return output



if __name__ == "__main__":

    now_path = os.getcwd()
    input_dir = os.path.join(now_path,'input_dir')
    output_dir = os.path.join(now_path, 'result')

    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    files = os.listdir(input_dir)
    for file in files:
        file_path = os.path.join(input_dir,file)
        save_path = os.path.join(output_dir,file)
        img = cv2.imread(file_path)


        img0 = infer(img)

        cv2.imwrite(save_path,img0)
