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
    # kmean_anchors(path='../../DataSets/Infrared/infrared-pv.yaml', n=9, img_size=640, thr=4.0, gen=5000, verbose=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/home/gy/CX/Ship/models/new_split/models/yolov5s/Jetson-nano/yolov5-v6-ship-2cls/exp.5/weights/model.cfg', help='*.cfg path') 
    parser.add_argument('--weights', type=str, default='/home/gy/CX/Ship/models/new_split/models/yolov5s/Jetson-nano/yolov5-v6-ship-2cls/exp.5/weights/best.weights', help='weights path')
    parser.add_argument('--r', type=float, default=0.9, help='weights path')
    parser.add_argument('--type', type=str, default="bn", help='weights path')  
    opt = parser.parse_args() 
    opt.save_pth = opt.weights[::-1].split("/", 1)[1][::-1]
 
    model = Darknet( opt.cfg  )

    model.to('cpu').eval()
    # model.convert2rt(opt.weights.replace(".weights", ".wts"))

    if opt.weights.endswith('.pt'):  # pytorch format 
        try:
            try:
                model.load_state_dict(torch.load( opt.weights, map_location='cpu')['model'].state_dict()) 
            except:
                model.load_state_dict(torch.load( opt.weights, map_location='cpu')['model'])  
            print('Transferred %g items from $old$.pt ' %   len(model.state_dict()) )  # report
        except :
            new_ckpt = {}
            ckpt = torch.load( opt.weights, map_location='cpu')
            new_ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
            print('Transferred %g/%g items from %s' % (len(new_ckpt['model'].items()), len(model.state_dict()), opt.weights))  # report
        try:
            save_weights(model, opt.weights.replace( ".pt", ".weights" ))
        except:
            print( "Transfer pt 2 weights failed !")
    elif opt.weights.endswith('.weights'): 
        load_darknet_weights(model, opt.weights )

    img = model(torch.rand(3, 3, 608, 608).to("cpu"))

    for ii in [ opt.r , ]: 
        model.Do_Prune( style=opt.type, show=True, prune_percent= 0.25, save_path=opt.save_pth  ) 