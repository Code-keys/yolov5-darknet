from .layers import * 

import os 
from pathlib import Path
import numpy as np

from . import torch_utils

ONNX_EXPORT = False


activate = []


def parse_model_cfg(path):
    # Parse the yolo *.cfg file and return module definitions path may be 'cfg/yolov3.cfg', 'yolov3.cfg', or 'yolov3'
    if not path.endswith('.cfg'):  # add .cfg suffix if omitted
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):  # add cfg/ prefix if omitted
        path = 'cfg' + os.sep + path

    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    mdefs = []  # module definitions
    for idx_line, line in enumerate(lines):
        line.lstrip().rstrip()
        if line.startswith('['):  # This marks the start of a new block
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            # print(idx_line, line) 
            key, val = line.split("=")
            key = key.rstrip()

            if key == 'anchors':  # return nparray
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors
            elif (key in ['from', 'layers', 'mask']) or (key == 'size' and ',' in val):  # return array
                mdefs[-1][key] = [int(x) for x in val.split(',')]
            else:
                val = val.strip()
                # TODO: .isnumeric() actually fails to get the float case
                if val.isnumeric():  # return int or float
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val  # return string

    # Check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups', 'group_id',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'probability', 'dilation']

    f = []  # fields
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]  # unsupported fields
    assert not any(u), "Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631" % (u, path)

    return mdefs

def parse_data_cfg(path):
    # Parses the data configuration file
    if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options

def create_modules(module_defs, img_size, cfg):
    # Constructs module list of layer blocks from module configuration in module_defs

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size  # expand if necessary
    _ = module_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1
    cur_stride = 1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if 'stride' in mdef :
            if  mdef['type'] == 'upsample' :
                cur_stride  =  cur_stride//int(mdef['stride'])
            elif  mdef['type'] == 'convolutional':
                cur_stride =  cur_stride * int(mdef['stride']) 
            elif mdef['type'] == 'maxpool':
                if module_defs[i-2]["type"] == "convolutional" and  module_defs[i-2]["stride"] == 2:
                    cur_stride = cur_stride
                else:
                    cur_stride = cur_stride * int(mdef['stride']) 


        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            pad = k//2 if 'pad' in mdef else 0
            pad = mdef['pad'] if 'dilation' in mdef else pad 
            if isinstance(k, int):  # single-size conv
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       dilation= mdef['dilation'] if 'dilation' in mdef else 1,
                                                       padding= pad,
                                                       groups=mdef['groups'] if 'groups' in mdef else 1,
                                                       bias=not bn))
            else:  # multiple-size conv
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1],
                                                          out_ch=filters,
                                                          k=k,
                                                          stride=stride,
                                                          bias=not bn))

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())
            elif mdef['activation'] == 'silu':
                modules.add_module('activation', nn.SiLU())
            elif mdef['activation'] == 'relu':
                modules.add_module('activation', nn.ReLU())

        elif mdef['type'] == 'BatchNorm2d':
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
            if i == 0 and filters == 3:  # normalize RGB image
                # imagenet mean and var https://pytorch.org/docs/stable/torchvision/models.html#classification
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif mdef['type'] == 'maxpool':
            k = mdef['size']  # kernel size
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'upsample':
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))  # img_size = (320, 192)
            else:
                modules = nn.Upsample(scale_factor=mdef['stride'])

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            routs.extend([i + l if l < 0 else l for l in layers])
            if 'group_id' in mdef and 'groups' in mdef:
                filters = sum([output_filters[l + 1 if l > 0 else l] // int(mdef['groups'])  for l in layers])
                modules = GroupConcat(layers=layers, groups=mdef["groups"], group_id=mdef["group_id"])
            else:
                filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
                modules = FeatureConcat(layers=layers)

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)
        
        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            # stride = [32, 16, 8]  # P5, P4, P3 strides
            # if any(x in cfg for x in ['panet', 'yolov4', 'cd53']):  # stride order reversed
            #     stride = list(reversed(stride))
            layers = mdef['from'] if 'from' in mdef else []
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],  # anchor list
                                nc=mdef['classes'],  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1, 2...
                                layers=layers,  # output layers
                                # stride=stride[yolo_index]
                                stride=cur_stride
                                )

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                j = layers[yolo_index] if 'from' in mdef else -1
                # If previous layer is a dropout layer, get the one before
                if module_list[j].__class__.__name__ == 'Dropout':
                    j -= 1
                bias_ = module_list[j][0].bias  # shape(255,)
                bias = bias_[:modules.no * modules.na].view(modules.na, -1).contiguous()  # shape(3,85) 
                bias[:, 4] = -4.5  # obj
                print( bias.size() )
                bias[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                print('WARNING: smart bias initialization failure.')

        elif mdef['type'] == 'dropout':
            perc = float(mdef['probability'])
            modules = nn.Dropout(p=perc)
        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p, out):
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            i, n = self.index, self.nl  # index in layers, number of layers
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

            # outputs and weights
            # w = F.softmax(p[:, -n:], 1)  # normalized weights
            w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)
            # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

            # weighted ASFF sum
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * \
                         F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)
        elif ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        if self.layers.__len__() == 2 : 
            # for double-head
            reg = out[self.layers[0]].view(bs, self.na, 5, self.ny, self.nx).permute(0, 1, 3, 4, 2)  # prediction
            cls = out[self.layers[1]].view(bs, self.na, self.nc, self.ny, self.nx).permute(0, 1, 3, 4, 2) # prediction
            p = torch.cat( [reg, cls] , -1).contiguous()
        else:
            p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # Inference
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    # YOLOv3 object detection model
    def __init__(self, cfg, img_size=(800, 800), verbose=False):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg)
        # print(self.module_list)
        self.yolo_layers = get_yolo_layers(self)
        self.stride = get_yolo_strides(self)
        # torch_utils.initialize_weights(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        self.info(verbose) if not ONNX_EXPORT else None  # print model description
 
    def forward(self, x, augment=False, verbose=False,  DP_index=[]):
        if not augment:
            return self.forward_once(x, verbose=verbose, DP_index=DP_index)
        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale

            # for i, yi in enumerate(y):  # coco small, medium, large = < 32**2 < 96**2 <
            #     area = yi[..., 2:4].prod(2)[:, :, None]
            #     if i == 1:
            #         yi *= (area < 96. ** 2).float()
            #     elif i == 2:
            #         yi *= (area > 32. ** 2).float()
            #     y[i] = yi

            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x, augment=False, verbose=False, DP_index=[] ):
        img_size = x.shape[-2:]  # height, width
        yolo_out, out ,for_dilation = [], [], []
        if verbose:
            print('0', x.shape)
            str = '' 
        # Augment images (inference and test only)
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                           torch_utils.scale_img(x, s[1]),  # scale
                           ), 0)

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__ 
            # print(name) 
            # print(i, name)
            if name in ['WeightedFeatureFusion', 'FeatureConcat', 'GroupConcat' ]:  # sum, concat 
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
           
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat() GroupCncat

            elif name == 'YOLOLayer':
                yolo_out.append(module(x, out))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)
            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

            for_dilation.append( x if ( i not in DP_index or name != 'YOLOLayer') else [] )

        if DP_index :
            return yolo_out, for_dilation

        if self.training:  # train
            return yolo_out 

        elif ONNX_EXPORT:  # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4

        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            if augment:  # de-augment results
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # scale
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1)
            return x, p 

    def fuse(self,device):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv.cpu(), b.cpu()).to(device)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info() if not ONNX_EXPORT else None  # yolov3-spp reduced from 225 to 152 layers

    def info(self, verbose=False):
        model_info(self, verbose)
    
    def convert2rt(self, path =  'yolov4.wts'):
        import struct
        with open( path , 'w') as f:
            f.write('{}\n'.format(len(self.state_dict().keys())))
            for k, v in self.state_dict().items():
                vr = v.reshape(-1).cpu().numpy()
                f.write('{} {} '.format(k, len(vr)))
                for vv in vr:
                    f.write(' ')
                    f.write(struct.pack('>f',float(vv)).hex())
                f.write('\n')
        print('convert to trnsorrt done!')

    ############################# prune utils  #############################
    def Do_Prune(self, style="l1_norm", show=True, prune_percent= 0.25, save_path='' ): 

        from .prune_utils import bn_scale, parse_module_defs
        import time, copy

        bn_groups, cv_groups, other = [], [], [] 

        *other, CBL = parse_module_defs(self.module_defs)
 
        if style == "bn":    # "bn_scale":
            style = 'BN_SCALE'
            for idx, m in enumerate( self.module_list ):
                if isinstance(m, nn.Sequential):    
                    if m.__len__() == 1:
                        continue
                    if m[0].kernel_size == (3, 3):
                        continue
                    if idx in CBL :
                        # isinstance(m[1], nn.BatchNorm2d):
                        bn_groups.append( (f"The ID OF BN LAYER: {idx}", m[1].weight.abs().cpu().data.detach().numpy(), 0, idx) ) 
            group = bn_groups
            scores, thre = bn_scale( group , prune_percent )
        elif style == "hr":     # "bn_scale":
            return
        elif style == 'l1':
            return
        elif style == 'fpgm': 
            return
        elif style == "f1f": 
            return
        else:
            assert( "ERROR Prune Type ! ")  
        try: 
                        
            from matplotlib import pyplot as plt
            global point 
            def click(event):
                global point
                artist = event.artist 
                ind = event.ind[0] 
                xd = artist.get_xdata()[ind] 
                yd = artist.get_ydata()[ind] 
                point = ( xd, yd )
                print("Cut point is ", point )

            for i,  score in enumerate( scores ) :   # name, value, axis, pruned_idx, out_channels 
                point = (-1, -1)

                X = np.array( range(score[1].__len__()) )
                Y = np.sort( score[1] ) # plt.scatter(X, Y, s=10, alpha=.5)
                line, = plt.plot( X , Y, 'g^', picker=4) 

                plt.xlim(  -1 , score[1].__len__())
                # plt.ylim( min(0.5,score[1].min() ) ,max( 1.5, score[1].max()) )
                plt.ylim(  score[1].min() ,  score[1].max() )

                plt.xlabel("FILTER_ID")
                plt.ylabel("IMPORTANCE")
                plt.title( style +" TYPE: " + score[0] )
                plt.axhline( y= thre, ls=":", c="red", linestyle='--') #添加水平直线

                cid = plt.gcf().canvas.mpl_connect("pick_event", click)

                plt.ion()   
                plt.pause( 0.5 )
                # plt.show()
 
                if point !=  (-1, -1) :
                    channel  = score[1].__len__() - point[0]
                    scores[i][-1] = channel

                # scores[i][-1] =  (scores[i][-1] // 8) * 8 if scores[i][-1] % 8 < 4 else (scores[i][-1] // 8 ) * 8 + 8
                scores[i][-1] = (scores[i][-1] // 2 + scores[i][-1] % 2) * 2
                scores[i][-1] = 1 if scores[i][-1] <= 4 else scores[i][-1]
                    
                print("Type: %s, Layer: %s, %3.3f(Mean), %3.3f(Max), %3.3f (Min),  >>   %3d  >>   %3d" \
                    % (style, score[0], score[1].mean(),  score[1].max(), score[1].min(),score[1].__len__() ,score[-1]) )
                plt.close()
        except:
            print("Error On Plot.")

        if save_path :     
            import copy 
            import tqdm
            def Compute_time( model, tt ) :
                img = torch.rand([1, 3, 640, 640]).to( "cuda:0" )
                model = model.to( "cuda:0" ).train()
                model(img)
                model(img)
                model(img)
                print( f"Evaluate {tt} model , wait !" )
                t = time.time()
                for i in tqdm.tqdm( range(10) ):
                    model(img)
                return time.time() - t
            compact_module_defs = copy.deepcopy(self.module_defs)

            for id , score in enumerate( scores ):    # module_list.94.FilterStripe.FilterSkeleton
                idx = int(score[0].split(":")[-1].strip()) 
                module = self.module_list[id]
                filters_remain = int(score[-1])
 
                assert compact_module_defs[idx]['type'] == 'convolutional' 
                compact_module_defs[idx]['filters'] = str(filters_remain) 

            self.write_cfg( save_path + os.sep + style + f"_{prune_percent}.cfg", compact_module_defs) 

            model = Darknet( save_path + os.sep + style + f"_{prune_percent}.cfg"  ) 
            t2 = Compute_time(self,'org')
            t1 = Compute_time(model, 'new' )
            print("Time Waste comparation is :   %3.3f(before)   >>   %3.3f(after)\n   speed accelebrate rate : %3.3f     " % ( t2, t1, (t2-t1)/t2 ) )

    def write_cfg(self, cfg_file, module_defs): 
        with open(cfg_file, 'w') as f: 
            f.write('''[net]\n# Testing\n# batch=1\n# subdivisions=1\n# Training\nbatch=64\nsubdivisions=8\n''')
            f.write('''width=640\nheight=640\nchannels=3\nmomentum=0.949\n# decay=0.001\ndecay=0.0005\nangle=0.3\n''')
            f.write('''saturation=1.5\nexposure=1.5\nhue=.1\n\n# learning_rate=0.00261\nlearning_rate=0.001261\n''')
            f.write('''burn_in=1000\nmax_batches=25000\npolicy=steps\nsteps=12000,20000\nscales=.1,.1\n#cutmix=1\nmosaic=1\n\n''')
            for idx, module_def in enumerate( module_defs ): 
                f.write(f"#Layer {idx}\n") 
                f.write(f"[{module_def['type']}]\n") 
                for key, value in module_def.items():
                    if key == 'type':
                        continue
                    elif key == 'anchors':  # return nparray 
                        v = ''
                        for i in value:
                            for j in i:
                                j = int(j)
                                v += f"{j}," 
                        f.write(f"{key}=" + v[:-1] + "\n")  
                    elif (key in ['from', 'layers', 'mask']) or (key == 'size' and isinstance(value, list)):  # return array
                        v = ''
                        for i in value:
                            v += f"{i},"
                        f.write(f"{key}=" + v[:-1] + "\n") 
                    else:
                        f.write(f"{key}={value}\n")   
                f.write("\n") 
        return True

    def write_weight(self, path="" ):
        if path.endwith(".pt"):
            torch.save(self.state_dict, path)
        elif path.endwith(".weight"):
            save_weights(self, path)  


def get_yolo_layers(model):
    return [i for i, m in enumerate(model.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]

def get_yolo_strides(model):
    return np.array(
        [m.stride for i, m in enumerate(model.module_list) if m.__class__.__name__ == 'YOLOLayer'])  # [89, 101, 113]

def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw
    print('Transferred %g items from $old$.weights ' %   ptr  )  # report

def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        target = weights.rsplit('.', 1)[0] + '.weights'
        save_weights(model, path=target, cutoff=-1)
        print("Success: converted '%s' to '%s'" % (weights, target))

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        target = weights.rsplit('.', 1)[0] + '.pt'
        torch.save(chkpt, target)
        print("Success: converted '%s' to '%s'" % (weights, target))

    else:
        print('Error: extension not supported.')


def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip().replace("'", '')
    msg = weights + ' missing, try downloading from https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0'

    if len(weights) > 0 and not os.path.isfile(weights):
        d = {'yolov3-spp.weights': '16lYS4bcIdM2HdmyJBVDOvt3Trx6N3W2R',
             'yolov3.weights': '1uTlyDWlnaqXcsKOktP5aH_zRDbfcDp-y',
             'yolov3-tiny.weights': '1CCF-iNIIkYesIDzaPvdwlcf7H9zSsKZQ',
             'yolov3-spp.pt': '1f6Ovy3BSq2wYq4UfvFUpxJFNDFfrIDcR',
             'yolov3.pt': '1SHNFyoe5Ni8DajDNEqgB2oVKBb_NoEad',
             'yolov3-tiny.pt': '10m_3MlpQwRtZetQxtksm9jqHrPTHZ6vo',
             'darknet53.conv.74': '1WUVBid-XuoUBmvzBVUCBl_ELrzqwA8dJ',
             'yolov3-tiny.conv.15': '1Bw0kCpplxUqyRYAJr9RY9SGnOJbo9nEj',
             'yolov3-spp-ultralytics.pt': '1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4'}

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)
        else:  # download from pjreddie.com
            url = 'https://pjreddie.com/media/files/' + file
            print('Downloading ' + url)
            r = os.system('curl -f ' + url + ' -o ' + weights)

        # Error check
        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.system('rm ' + weights)  # remove partial downloads
            raise Exception(msg)
 

def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = torch.nn.Conv2d(conv.in_channels,
                                    conv.out_channels,
                                    kernel_size=conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    bias=True)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0))
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv


def model_info(model, verbose=False):
    
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        macs, _ = profile(model, inputs=(torch.zeros(1, 3, 480, 640),), verbose=False)
        fs = ', %.1f GFLOPS' % (macs / 1E9 * 2)
    except:
        fs = ''

    print('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))




if __name__ == "__main":
    pass
