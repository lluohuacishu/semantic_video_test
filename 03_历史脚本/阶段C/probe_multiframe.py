"""4/8-frame gradient and memory checks; no optimizer, no model updates."""
import numpy as np
from PIL import Image
from experiment_common import *
from scipy import ndimage


def main():
    configure()
    video=next(v for v in manifest()["videos"] if v["video"]=="video131")
    label=next(x for x in video["labels"] if x["fish_pixels"]>0 and x["frame_index"]>=7)
    mask=np.asarray(Image.open(label["fish_mask"]))>0;ys,xs=np.nonzero(mask)
    left=int(np.clip(np.mean(xs)-128,0,1024));top=int(np.clip(np.mean(ys)-128,0,464))
    out=[]
    for count in (4,8):
        weight=RUN/"training/M_ROI16/checkpoint_512.ckpt"
        model,i_model,_=load_models(weight,PROJECT/"checkpoints/DMCI-6.0-mirror.pth.tar",entropy=False)
        initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        frames=[]
        start=label["frame_index"]-(count-1)
        for idx in range(start,label["frame_index"]+1):
            rgb=np.asarray(Image.open(Path(video["sequence_dir"])/f"{idx:05d}.jpg").convert("RGB").crop((left,top,left+256,top+256))).copy()
            frames.append(prepare(torch.from_numpy(rgb).permute(2,0,1)[None].float().cuda()/255,64))
        with torch.no_grad():reference=i_model(frames[0],q_index=24)["x_hat"].detach()
        del i_model;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        dpb={"ref_frame":reference,"ref_feature":None};model.train();losses=[];flags=[]
        roi=PrecomputedROILoss(16);ordinary=ordinary_loss()
        roi_mask=torch.from_numpy(mask[top:top+256,left:left+256].astype(np.float32))[None].cuda()
        for local,target in enumerate(frames[1:],1):
            fa=model.frame_index_map[local%len(model.frame_index_map)]
            rd=model(target,dpb,q_index=24,fa_idx=fa);dpb=rd["dpb"]
            lam=torch.tensor([768**(24/63)],device="cuda")
            stats=roi(rd,target,roi_mask,lam) if local==count-1 else ordinary(rd,target,lambdas=lam)
            losses.append(stats["loss"].mean());flags.append({k:None if v is None else v.requires_grad for k,v in dpb.items()})
        total=torch.stack(losses).mean();total.backward()
        norm=sum(p.grad.double().square().sum().item() for p in model.parameters() if p.grad is not None)**.5
        if not np.isfinite(norm) or norm==0 or any(not torch.equal(initial[n],p.detach().cpu()) for n,p in model.named_parameters()):raise AssertionError("Invalid multiframe gradient or unexpected update")
        row={"frames":count,"gradient_P_frames":count-1,"q_index":24,"crop":256,"source_indices":list(range(start,label["frame_index"]+1)),"roi_supervision_only_on_trusted_final_GT":True,
             "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),"gradient_norm":norm,"dpb_requires_grad":flags,"optimizer_updates":False}
        out.append(row);print("MULTIFRAME_OK",row,flush=True)
        del model,frames,dpb,rd,losses,total,reference;torch.cuda.empty_cache()
    save_json(RUN/"multiframe_probe.json",{"purpose":"gradient/memory feasibility only, not trained multiframe performance","rows":out})


if __name__=="__main__":main()
