"""CPU-only ROI grouping. No GT, model inference, or target filtering."""
from __future__ import annotations
import math

MULTI_PROFILE="semantic-video-layered-v2"

def plan_halves(records,width,height,segment_frames=32,padding=16):
    """Up to two disjoint crops per window; preserve every predicted box."""
    midpoint=(width//4)*2
    if min(midpoint,width-midpoint)<16:
        raise ValueError("Two ROI partitions each require at least 16 pixels")
    patches=[]
    for start in range(0,len(records),segment_frames):
        window=records[start:start+segment_frames]
        for side,left,right in (("left",0,midpoint),("right",midpoint,width)):
            pieces=[]
            for record in window:
                for region in record["regions"]:
                    x,y,w,h=region["box"]
                    a,b=max(x,left),min(x+w,right)
                    if b>a:pieces.append((a,y,b-a,h))
            if not pieces:continue
            x=max(left,min(b[0] for b in pieces)-padding)//2*2
            y=max(0,min(b[1] for b in pieces)-padding)//2*2
            end_x=min(right,math.ceil((max(b[0]+b[2] for b in pieces)+padding)/2)*2)
            end_y=min(height,math.ceil((max(b[1]+b[3] for b in pieces)+padding)/2)*2)
            end_x=min(right,max(end_x,x+16));end_y=min(height,max(end_y,y+16))
            x=min(x,end_x-16);y=min(y,end_y-16)
            patches.append({"start":start,"frames":len(window),"box":[x,y,end_x-x,end_y-y],"partition":side})
    return patches

def spatial_overlap(a,b):
    return a[0]<b[0]+b[2] and b[0]<a[0]+a[2] and a[1]<b[1]+b[3] and b[1]<a[1]+a[3]
