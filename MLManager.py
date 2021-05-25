import os
from threading import Timer
import argparse
import time
from pathlib import Path
import shutil

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box, plot_one_box_PIL
from utils.torch_utils import select_device, load_classifier, time_synchronized

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MLManager():
    def __init__(self,weights_path) -> None:
        self.weights_path = weights_path
        self.opt = dotdict({
            "img_size":640,
            "conf_thres":0.25,
            "iou_thres":0.45,
            "max_det":1000,
            "device":'',
            "classes":None,
            "agnostic_nms":False,
            "augment":False,
            "save_crop":False,
            "line_thickness":3,
            "save_txt":True,
            "save_img":True,
            "project":"detection-result",
            "hide_labels":False,
            "save_conf":True,
            "hide_conf":False,
            "view_img":False,
            "name":"exp",
            "exist_ok":False
        })
        self.imgz = 0
        self.stride = 0
        self.device = None
        self.half = None
        self.names = None

        self.model = self.load_model(weights_path=weights_path)

    def delete_detection_folder(self):
        if os.path.exists(self.opt.project):
            shutil.rmtree(self.opt.project)
    
    def load_model(self,weights_path):
        # Initialize
        imgsz = self.opt.img_size
        device = select_device(self.opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights_path, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16

        # set model
        self.imgsz = imgsz
        self.stride = stride
        self.device = device
        self.half = half
        self.names = names
        print("Finish loading model")
        return model

    def predict_image(self,source):
        # will save (class_code,class_name,conf)
        detection_result = []
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)
        
        # Directories
        save_dir = increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok)  # increment run
        (save_dir / 'labels' if self.opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img,augment=self.opt.augment)[0]

            # use NMS

            pred = non_max_suppression(pred,self.opt.conf_thres,self.opt.iou_thres,
            self.opt.classes,self.opt.agnostic_nms,max_det=self.opt.max_det)    
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if self.opt.save_crop else im0  # for opt.save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:,:4] = scale_coords(img.shape[2:],det[:, :4],im0.shape).round()

                # Results
                # Plot the bounding box
                for *xyxy, conf,cls in reversed(det):
                    if self.opt.save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        f.close()

                    c = int(cls)  # integer class
                    label = None if self.opt.hide_labels else (self.names[c] if self.opt.hide_conf else f'{self.names[c]} {conf:.2f}')
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=self.opt.line_thickness)
                    conf = float(conf)
                    detection_result.append((c,self.names[c],conf))
                    # Print results
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            if self.opt.view_img:
                cv2.imshow(str(p),im0)
                cv2.waitKey(1)

            if self.opt.save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path,im0)    

        if self.opt.save_txt or self.opt.save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.opt.save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')
        return detection_result