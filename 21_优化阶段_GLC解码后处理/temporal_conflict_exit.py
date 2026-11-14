"""Causal, receiver-only temporal conflict confirmation; research profiles only."""
import cv2
import numpy as np
from receiver_structure import KEYS
from mild_reliability_gate import smooth_risk

PROFILES = {
    'P3': {'temporal_confirmation': 3, 'motion_memory': False, 'correction': 'current_raw'},
    'M3': {'temporal_confirmation': 3, 'motion_memory': True, 'correction': 'current_raw'},
    'ML3': {'temporal_confirmation': 3, 'motion_memory': True, 'correction': 'mean_sigma1_residual'},
}
SETTINGS = {
    'sigma3_thresholds': [6., 12.], 'sigma1_thresholds': [12., 24.],
    'signed_coherence_thresholds': [.75, 1.], 'motion_excess_thresholds': [2., 6.],
    'rise_per_frame': 1/3, 'fall_per_frame': 1/6,
    'memory_hold_conflict_min': .1, 'memory_release_factor': .5,
    'seed_dilation_pixels': 1, 'cycle_max': 1.5, 'margin': 2,
    'reset_at': list(KEYS),
}


class TemporalExit:
    """Keep two past frames and one transported gate state, with no lookahead."""
    def __init__(self, height, width):
        self.yy, self.xx = np.mgrid[:height, :width].astype(np.float32)
        self.height, self.width = height, width
        self.history = []
        self.key = None
        self.last_frame = -1
        self.p_weight = np.zeros((height, width), np.float32)
        self.m_weight = self.p_weight.copy()
        self.memory = self.p_weight.copy()

    @staticmethod
    def warp(value, mx, my):
        return cv2.remap(value, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    def step(self, frame, raw, q9, alpha, back, forward):
        if frame != self.last_frame+1:
            raise ValueError('Frames must be consumed exactly once in causal order')
        self.last_frame = frame
        key = KEYS[np.searchsorted(KEYS, frame, side='right')-1]
        if key != self.key:
            self.history.clear()
            self.p_weight.fill(0)
            self.m_weight.fill(0)
            self.memory.fill(0)
            self.key = key
        r, q = raw.astype(np.float32), q9.astype(np.float32)
        r3, q3 = [cv2.GaussianBlur(x, (0, 0), 3) for x in (r, q)]
        residual3 = r3-q3
        residual1 = cv2.GaussianBlur(r-q, (0, 0), 1)
        ax, ay = self.xx+back[..., 0], self.yy+back[..., 1]
        inverse = self.warp(forward, ax, ay)
        valid = ((ax >= 2) & (ay >= 2) & (ax < self.width-2) & (ay < self.height-2)
                 & (np.linalg.norm(back+inverse, axis=2) < SETTINGS['cycle_max']))
        aligned = []
        for old in self.history:
            f = self.warp(old['forward'], ax, ay)
            mx, my = ax+f[..., 0], ay+f[..., 1]
            valid &= (mx >= 2) & (my >= 2) & (mx < self.width-2) & (my < self.height-2)
            aligned.append({n: self.warp(old[n], mx, my) for n in ('residual3','residual1','r3','q3')})
            if old is self.history[-1]:
                p_previous = self.warp(self.p_weight, mx, my)
                m_previous = self.warp(self.m_weight, mx, my)
                memory_previous = self.warp(self.memory, mx, my)
        risk = np.zeros_like(alpha, dtype=np.float32)
        seed = risk.copy()
        mean1 = residual1
        if len(aligned) == 2:
            rs = [a['residual3'] for a in aligned]+[residual3]
            mean3 = np.mean(rs, axis=0)
            mean1 = np.mean([a['residual1'] for a in aligned]+[residual1], axis=0)
            magnitude = np.max(np.abs(mean3), axis=2)
            denominator = np.mean([np.max(np.abs(v), axis=2) for v in rs], axis=0)
            coherence = np.clip(magnitude/np.maximum(denominator, 1e-6), 0, 1)
            risk = (smooth_risk(magnitude, *SETTINGS['sigma3_thresholds'])
                    * smooth_risk(np.max(np.abs(mean1), axis=2), *SETTINGS['sigma1_thresholds'])
                    * smooth_risk(coherence, *SETTINGS['signed_coherence_thresholds']))
            risk *= valid
            # Excess decoded-raw change over Q9 change over two frame intervals.
            # This is a motion/innovation proxy, not semantic fish motion.
            raw_change = np.max(np.abs(r3-aligned[0]['r3']), axis=2)
            q_change = np.max(np.abs(q3-aligned[0]['q3']), axis=2)
            seed = smooth_risk(np.maximum(raw_change-q_change, 0), *SETTINGS['motion_excess_thresholds'])
            seed = cv2.dilate(seed, np.ones((3,3),np.uint8))*valid
            self.memory = np.where(risk > SETTINGS['memory_hold_conflict_min'],
                                   np.maximum(memory_previous, seed),
                                   memory_previous*SETTINGS['memory_release_factor']).astype(np.float32)
            def approach(previous, target):
                result = previous+np.clip(target-previous,
                                          -SETTINGS['fall_per_frame'], SETTINGS['rise_per_frame'])
                return np.where(valid & (alpha > 0), np.clip(result, 0, 1), 0).astype(np.float32)
            self.p_weight = approach(p_previous, risk)
            self.m_weight = approach(m_previous, risk*self.memory)
        else:
            self.p_weight.fill(0)
            self.m_weight.fill(0)
            self.memory.fill(0)
        self.history.append(dict(residual3=residual3, residual1=residual1,
                                 r3=r3, q3=q3, forward=np.array(forward, copy=True)))
        self.history = self.history[-2:]
        residual = r-q
        low_correction = np.minimum(np.maximum(mean1, np.minimum(residual,0)), np.maximum(residual,0))
        low_correction *= self.m_weight[...,None]
        results = {}
        for name, correction, weight in (
            ('P3',self.p_weight[...,None]*residual,self.p_weight),
            ('M3',self.m_weight[...,None]*residual,self.m_weight),
            ('ML3',low_correction,self.m_weight)):
            rgb = np.rint(np.clip(q+correction,0,255)).astype(np.uint8)
            results[name] = (rgb,weight.copy())
        stats = dict(frame=frame,key=int(key),history_frames=len(aligned)+1,
                     conflict_fraction=float((risk > .1).mean()),
                     seed_fraction=float((seed > .5).mean()),
                     motion_memory_fraction=float((self.memory > .5).mean()),
                     p_mean=float(self.p_weight.mean()),m_mean=float(self.m_weight.mean()),
                     p_active=float((self.p_weight > 0).mean()),m_active=float((self.m_weight > 0).mean()))
        return results,stats

