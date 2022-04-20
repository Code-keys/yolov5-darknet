from utils.autoanchor import kmean_anchors #,kmean_anchors_plus_plus

from darknet.models import Darknet ,load_darknet_weights,save_weights

import torch


def transfer_from_pt(cfg, weights):

    model = Darknet(cfg ).to('cuda:0')
    pretrained = weights.endswith('.pt') 
    
    if weights.endswith('.pt'):  # pytorch format 
        try:
            try:
                model.load_state_dict(torch.load(weights, map_location='cuda:0')['model'].state_dict()) 
            except:
                model.load_state_dict(torch.load(weights, map_location='cuda:0')['model'])  
            print('Transferred %g items from $old$.pt ' %   len(model.state_dict()) )  # report
        except KeyError as e:
            ckpt = torch.load(weights, map_location='cuda:0')
            new_ckpt['model'] = {k: v for k, v in ckpt['model'].state_dict().items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'].state_dict() , strict=False)
            print('Transferred %g/%g items from %s' % (len(new_ckpt['model'].items()), len(model.state_dict()), weights))  # report

    save_weights(model, weights.replace(".pt", ".weights")  )
    print( f'saved {weights.replace(".pt", ".weights")}')


if __name__ == '__main__':

    # kmean_anchors(path='/home/gy/CX/VisDrone/VisDrone.yaml', n=12, img_size=2016, thr=3.0, gen=5000, verbose=True)


    
    cfg = '/home/gy/CX/head_yolov5/yolov5-darknet/weights/yolov5s-csp-1_3.cfg'

    weights = '/home/gy/CX/yolov5-4.0/darknet/yolov5sm/exp4/weights/last.pt'


    # transfer_from_pt(cfg, weights)

    model = Darknet( cfg  ).to('cuda:2').train()
    img = model( torch.Tensor(3,3,832,832).to("cuda:2"), verbose=True)