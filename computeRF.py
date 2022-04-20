 
def outFromIn(isz, net, layernum):
    totstride = 1
    insize = isz
    for layer in range(layernum):
        fsize, stride, pad = net[layer]
        outsize = (insize - fsize + 2*pad) / stride + 1
        insize = outsize
        totstride = totstride * stride
    return outsize, totstride

def inFromOut(net, layernum):
    RF = 1
    for layer in reversed(range(layernum)):
        fsize, stride, pad = net[layer]
        RF = ((RF -1)* stride) + fsize
    return RF 

def compute(arch : list) : 
    RF =  [1]
    for i in arch:
        if i == "c3" :
            __i = RF[-1]
            RF.append( (__i-1) * 1 + 3 )
        if i == "spp" :
            __i = RF[-1]
            RF.append( (__i+13) )
        if i == "cv32" :
            __i = RF[-1]
            RF.append( (__i-1) * 2 + 3 )
        if i ==  "cv31" :
            __i = RF[-1]
            RF.append( (__i-1) * 1 + 3 )
        if i ==  "focus":
            __i = RF[-1]
            RF.append( (__i) * 2 )
        if i ==  "dc31":
            __i = RF[-1]
            RF.append( (__i-1)*1 + 3*(3-1)+1 )
        if i ==  "dc32":
            __i = RF[-1]
            RF.append( (__i-1)*1 + 9 )

    print (  "%6s：%6s "  %(arch[0], RF[-1]) )  

a = [ "c3" , "spp" , "cv32" , "cv31" , "focus", "dc3" ]  # conv 3


yolov5gf = [ "c3","spp" ,"cv32"     ,"c3" ,"c3" ,"c3" ,"cv32"    ,"c3" ,"c3" ,"cv32"  , "c3" , "dc3" ,"cv32",  "cv31", "focus" ]
yolov5s = [ "c3","spp" ,"cv32"     ,"c3" ,"c3" ,"c3" ,"cv32"    ,"c3" ,"c3" ,"c3" ,"cv32"    , "c3" ,"cv32"    , "cv31" ,  "focus" ]

yolov4 = [   "c3","c3" ,"c3" ,"c3" ,"cv32"     
            ,"c3" ,"c3" ,"c3" ,"c3" ,"c3" ,"c3" ,"c3" ,"c3" ,"cv32" 
            ,"c3" ,"c3" ,"c3" ,"c3" ,"c3" ,"c3" ,"c3" ,"c3" ,"cv32"       
            ,"c3", "c3" ,"cv32"   
            ,"c3" , "cv32", 
             "cv31"]

yolov5sm = [  "c3", "spp", "cv32"   
            , "c3", "c3", "c3", "cv32"    
            , "c3", "c3", "c3", "cv32"  
            , "dc31", "cv31", "cv32"
            , "dc31", "cv32" ]

yolov5sm_tiny = [
             "c3",  "dc32", "spp",  "cv32"    
            , "c3", "dc32", "c3", "cv32"  
            , "dc32",   "cv32"
             , "cv32", 'cv31'] # 8

arch = yolov5sm_tiny
#  5,8,  7,17,  13,12, 
#  13,26, 23,16,  23,29,  42,21
#  25,48,  48,37,  51,73,  90,45,  119,106

for i in range(arch.__len__()):
    _t = arch[i:] 
    compute(_t)
