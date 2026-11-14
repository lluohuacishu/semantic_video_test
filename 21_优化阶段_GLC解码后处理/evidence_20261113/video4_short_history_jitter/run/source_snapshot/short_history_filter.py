"""Research-only, causal three-frame blend using received-guide correspondence.

No recurrent output history: past inputs are original received frames. This
bounds additional temporal memory to two frames and avoids indefinite trails.
"""
import cv2
import numpy as np

SETTINGS = dict(weights=[.5, .3, .2], maximum_history_frames=2,
    maximum_history_seconds=2/15, flow_iterations=12, precision='FP32',
    cycle_sigma=1.5, guide_sigma=1., guide_difference_sigma=16.,
    edge_feather_pixels=2., current_guide_correction=.25,
    interpolation='linear', output_history_feedback=False,
    scene_cut_detection=False, lookahead_frames=0)


def maps_confidence(current, reference, backward, forward):
    h,w=current.shape[:2]
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    mx,my=xx+backward[...,0],yy+backward[...,1]
    inverse=cv2.remap(forward,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
    cycle=np.linalg.norm(backward+inverse,axis=2)
    low=cv2.GaussianBlur(current.astype(np.float32),(0,0),SETTINGS['guide_sigma'])
    old=cv2.GaussianBlur(reference.astype(np.float32),(0,0),SETTINGS['guide_sigma'])
    old=cv2.remap(old,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
    difference=np.max(np.abs(low-old),axis=2)
    distance=np.minimum.reduce((mx,my,w-1-mx,h-1-my))
    edge=np.clip(distance/SETTINGS['edge_feather_pixels'],0,1)
    confidence=edge*np.exp(-(cycle/SETTINGS['cycle_sigma'])**2
                           -(difference/SETTINGS['guide_difference_sigma'])**2)
    assert np.isfinite(confidence).all()
    return mx,my,confidence.astype(np.float32)


def blend_frame(video,guide,flows,frame):
    current=video[frame].astype(np.float32)
    total=current*.5
    denominator=np.full(current.shape[:2],.5,np.float32)
    stats=[]
    for lag in (1,2):
        if frame<lag:continue
        back,forward=flows[frame,lag-1]
        mx,my,confidence=maps_confidence(guide[frame],guide[frame-lag],back,forward)
        weight=SETTINGS['weights'][lag]*confidence
        old=cv2.remap(video[frame-lag].astype(np.float32),mx,my,cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT101)
        total+=weight[...,None]*old
        denominator+=weight
        stats.append(dict(lag=lag,mean_confidence=float(confidence.mean())))
    result=total/denominator[...,None]
    assert np.all(.5/denominator>=.5-1e-6)
    return result,dict(frame=frame,current_weight_mean=float((.5/denominator).mean()),
                      history_weight_max=float((1-.5/denominator).max()),lags=stats)


def guide_correction(rgb,guide):
    x=rgb.astype(np.float32)
    return x+.25*cv2.GaussianBlur(guide.astype(np.float32)-x,(0,0),1.)


def quantize(rgb):
    return np.rint(np.clip(rgb,0,255)).astype(np.uint8)

