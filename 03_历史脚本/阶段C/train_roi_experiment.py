"""Paired small-scale fine-tuning, frozen adjacent I reference, no inference ROI input."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from scipy import ndimage
from experiment_common import *
from src.transforms.functional import yuv_444_to_420

SEED=20261013
STEPS=512
Q_VALUES=[8,24,40]
P_WEIGHT=PROJECT/"checkpoints/mlvc-s-psnr-v1.ckpt"
I_WEIGHT=PROJECT/"checkpoints/DMCI-6.0-mirror.pth.tar"


def make_schedule():
    m=manifest()
    videos=[v for v in m["videos"] if v["experiment_role"]=="train"]
    rng=np.random.default_rng(SEED)
    schedule=[]
    for step in range(STEPS):
        video=videos[step%len(videos)]
        # Alternating video order is independent of GT status (period 4 vs 2).
        video=videos[int(rng.integers(len(videos)))]
        labelled=step%4<2
        guided=step%4==0
        if guided:
            fish_videos=[v for v in videos if any(x["fish_pixels"]>0 and x["previous_rgb"] for x in v["labels"])]
            if not fish_videos: raise ValueError("Training pool contains no positive fish GT")
            video=fish_videos[int(rng.integers(len(fish_videos)))]
        if labelled:
            choices=[x for x in video["labels"] if x["previous_rgb"] is not None and (x["fish_pixels"]>0 if guided else True)]
            if not choices: raise ValueError("No eligible fish-labelled target")
            entry=choices[int(rng.integers(len(choices)))]
            idx=entry["frame_index"]
            mask=np.asarray(Image.open(entry["fish_mask"]))>0
        else:
            gt_ids={x["frame_index"] for x in video["labels"]}
            choices=[i for i in range(video["first_index"]+1,video["last_index"]+1) if i not in gt_ids]
            idx=int(choices[int(rng.integers(len(choices)))])
            entry=None;mask=None
        width,height=1280,720
        crop=256
        kind="fish_guided" if guided else "confirmed_background" if labelled else "unannotated_random"
        if guided:
            cc,count=ndimage.label(mask,np.ones((3,3),bool))
            objects=ndimage.find_objects(cc)
            obj=objects[int(rng.integers(count))]
            center_x=(obj[1].start+obj[1].stop)//2+int(rng.integers(-48,49))
            center_y=(obj[0].start+obj[0].stop)//2+int(rng.integers(-48,49))
            left=int(np.clip(center_x-crop//2,0,width-crop));top=int(np.clip(center_y-crop//2,0,height-crop))
        else:
            for attempt in range(64):
                left=int(rng.integers(0,width-crop+1));top=int(rng.integers(0,height-crop+1))
                if not labelled or not mask[top:top+crop,left:left+crop].any(): break
            if labelled and mask[top:top+crop,left:left+crop].any(): kind="labelled_random"
        fraction=float(mask[top:top+crop,left:left+crop].mean()) if labelled else None
        schedule.append({"step":step+1,"video":video["video"],"target_index":idx,"reference_index":idx-1,
                         "reference_rgb":str(Path(video["sequence_dir"])/f"{idx-1:05d}.jpg"),"target_rgb":str(Path(video["sequence_dir"])/f"{idx:05d}.jpg"),
                         "mask":entry["fish_mask"] if entry else None,"gt_available":labelled,"crop_kind":kind,"left":left,"top":top,"crop":crop,
                         "horizontal_flip":bool(rng.integers(2)),"q_index":int(Q_VALUES[int(rng.integers(len(Q_VALUES)))]),"roi_fraction":fraction})
    RUN.mkdir(parents=True,exist_ok=True)
    protocol={"steps":STEPS,"seed":SEED,"lr":1e-5,"optimizer":"Adam","gradient_clip":1.,"crop":256,"q_values":Q_VALUES,
              "branches":{"M_RD":1,"M_ROI4":4,"M_ROI16":16},"labelled_update_fraction":.5,"fish_guided_update_fraction":.25,
              "reference":"one immediately preceding published RGB frame reconstructed by frozen DMCI, no original reference bypass at deployment",
              "loss":"native geometric YUV420 distortion with Y weight8 + bpp; lambda=768^(q/63); per-frame weights normalized to mean1",
              "checkpoint_selection":"fixed final update512, no codec checkpoint early stopping","inference_roi_input":False,"independent_seed_repetitions":1,
              "p_weight_sha256":sha256(P_WEIGHT),"i_weight_sha256":sha256(I_WEIGHT),"dataset_manifest_sha256":sha256(DATA/"manifest.json"),
              "evaluation_q_anchors":Q_VALUES,"evaluation_gop":32,"evaluation_continuous_frames":"whole available video96 (255) and official val108 (375), including every sparse GT anchor",
              "rate_match_policy":"for each video, use M_RD anchor complete file bytes; choose other branch Q only by absolute complete byte difference; GT metrics not used",
              "development_validation_is_official_validation":False,"test_used":False}
    if (RUN/"schedule.json").exists():
        if json.loads((RUN/"schedule.json").read_text("utf-8"))!=schedule: raise ValueError("Existing schedule differs")
    save_json(RUN/"schedule.json",schedule);save_json(RUN/"protocol.json",protocol)
    print("SCHEDULE_OK",len(schedule),flush=True)


def check_loss_equivalence():
    configure("cpu")
    rng_seed(SEED)
    ordinary=ordinary_loss()
    roi=PrecomputedROILoss(1)
    records=[]
    for name,mask in [("random",(torch.rand(1,64,64)>.93).float()),("empty",torch.zeros(1,64,64)),("full",torch.ones(1,64,64))]:
        target=torch.rand(1,3,64,64)
        a=(target+torch.randn_like(target)*.02).requires_grad_()
        b=a.detach().clone().requires_grad_()
        rate=torch.tensor([.05])
        lambdas=torch.tensor([768**(24/63)])
        x=ordinary({"x_hat":a,"bpp":rate},target,lambdas=lambdas)["loss"].mean()
        y=roi({"x_hat":b,"bpp":rate},target,mask,lambdas)["loss"].mean()
        ga=torch.autograd.grad(x,a)[0];gb=torch.autograd.grad(y,b)[0]
        if not torch.allclose(x,y,rtol=1e-6,atol=1e-8) or not torch.allclose(ga,gb,rtol=1e-6,atol=1e-8): raise AssertionError("ROI ratio1 is not ordinary loss")
        records.append({"case":name,"ordinary_loss":x.item(),"roi1_loss":y.item(),"loss_abs_difference":abs(x.item()-y.item()),"gradient_max_abs_difference":(ga-gb).abs().max().item()})
    RUN.mkdir(parents=True,exist_ok=True)
    save_json(RUN/"loss_equivalence.json",{"passed":True,"cases":records,"batch_size":1,"same_native_profile":True,"stage_a_arithmetic_probe_is_not_the_baseline":True,"upstream_batch2_issue":"get_roi_weights_wrapper broadcasts per-batch weights into last dimension; this experiment is locked to verified batch1, upstream left unchanged"})
    print("LOSS_EQUIVALENCE_OK",records,flush=True)


def crop_rgb(path,record):
    image=Image.open(path).convert("RGB").crop((record["left"],record["top"],record["left"]+256,record["top"]+256))
    array=np.asarray(image).copy()
    if record["horizontal_flip"]: array=array[:,::-1].copy()
    return array


def make_cache():
    configure()
    schedule=json.loads((RUN/"schedule.json").read_text("utf-8"))
    cache=RUN/"training_cache"
    cache.mkdir(parents=True,exist_ok=True)
    model,i_model,_=load_models(P_WEIGHT,I_WEIGHT,entropy=False)
    del model
    i_model.requires_grad_(False)
    with torch.no_grad():
        for record in schedule:
            target=cache/f'{record["step"]:05d}.npz'
            if target.exists(): continue
            previous=crop_rgb(record["reference_rgb"],record);current=crop_rgb(record["target_rgb"],record)
            tensors=[prepare(torch.from_numpy(x).permute(2,0,1)[None].float().cuda()/255,64) for x in (previous,current)]
            reference=i_model(tensors[0],q_index=record["q_index"])["x_hat"].detach().cpu().numpy()
            mask=np.asarray(Image.open(record["mask"]))[record["top"]:record["top"]+256,record["left"]:record["left"]+256].astype(np.float32) if record["gt_available"] else np.empty((0,),np.float32)
            if record["gt_available"] and record["horizontal_flip"]: mask=mask[:,::-1].copy()
            np.savez_compressed(target,reference=reference,target=tensors[1].cpu().numpy(),mask=mask)
            if record["step"]%64==0:print("CACHE",record["step"],flush=True)
    save_json(RUN/"cache_metadata.json",{"samples":len(schedule),"schedule_sha256":sha256(RUN/"schedule.json"),"i_weight_sha256":sha256(I_WEIGHT),"purpose":"shared frozen reconstructed references; private training cache never read by deployed cold decoder"})
    print("CACHE_OK",flush=True)


def regional_mse(xhat,target,mask,ratio):
    output={}
    terms=[]
    for name,a,b,region in zip(("Y","U420","V420"),yuv_444_to_420(xhat),yuv_444_to_420(target),(mask[:,None],F.avg_pool2d(mask[:,None],2),F.avg_pool2d(mask[:,None],2)),strict=True):
        p=region.mean();wb=1/(1+(ratio-1)*p);wr=ratio*wb
        e=(a-b).square()
        roi=(e*region*wr).mean();bg=(e*(1-region)*wb).mean()
        output[name]={"roi_weighted_mse_contribution":roi.item(),"background_weighted_mse_contribution":bg.item(),"roi_contribution_fraction":roi.item()/(roi.item()+bg.item()) if roi.item()+bg.item()>0 else None}
        if name=="Y":terms=[roi,bg]
    return output,terms


def train_branch(name,ratio):
    configure();rng_seed(SEED)
    folder=RUN/"training"/name
    folder.mkdir(parents=True,exist_ok=False)
    schedule=json.loads((RUN/"schedule.json").read_text("utf-8"))
    model,_,_=load_models(P_WEIGHT,entropy=False)
    initial={k:v.detach().cpu().clone() for k,v in model.named_parameters()}
    model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    ordinary=ordinary_loss();roi=PrecomputedROILoss(ratio)
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    records=[]
    try:
        with (folder/"steps.jsonl").open("x",encoding="utf-8") as log:
            for sample in schedule:
                rng_seed(SEED+sample["step"])
                cached=np.load(RUN/"training_cache"/f'{sample["step"]:05d}.npz')
                target=torch.from_numpy(cached["target"]).cuda();reference=torch.from_numpy(cached["reference"]).cuda()
                mask=torch.from_numpy(cached["mask"].copy())[None].cuda() if sample["gt_available"] else None
                optimizer.zero_grad(set_to_none=True)
                rd=model(target,{"ref_frame":reference,"ref_feature":None},q_index=sample["q_index"],fa_idx=1)
                lambdas=torch.tensor([768**(sample["q_index"]/63)],device="cuda")
                stats=ordinary(rd,target,lambdas=lambdas) if name=="M_RD" or mask is None else roi(rd,target,mask,lambdas)
                distortion_key="mse" if "mse" in stats else "recon_total"
                loss=stats["loss"].mean()
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
                diagnostic={}
                if mask is not None:
                    effective_ratio=1 if name=="M_RD" else ratio
                    regional,terms=regional_mse(rd["x_hat"],target,mask,effective_ratio)
                    reconstruction_gradient=torch.autograd.grad(stats[distortion_key].mean(),rd["x_hat"],retain_graph=True)[0]
                    region=mask[:,None].expand_as(reconstruction_gradient).bool()
                    diagnostic={"weighted_channel_mse":regional,"distortion_gradient_roi_abs_sum":reconstruction_gradient[region].abs().sum().item(),"distortion_gradient_background_abs_sum":reconstruction_gradient[~region].abs().sum().item(),
                                "distortion_gradient_roi_mean":reconstruction_gradient[region].abs().mean().item() if region.any() else None,"distortion_gradient_background_mean":reconstruction_gradient[~region].abs().mean().item() if (~region).any() else None}
                    if sample["step"]%32==1 and mask.any():
                        probe=model.encoder.conv1.weight
                        diagnostic["encoder_probe_Y_roi_gradient_norm"]=torch.autograd.grad(terms[0],probe,retain_graph=True)[0].norm().item()
                        diagnostic["encoder_probe_Y_background_gradient_norm"]=torch.autograd.grad(terms[1],probe,retain_graph=True)[0].norm().item()
                loss.backward()
                encoder_norm=sum(p.grad.double().square().sum().item() for n,p in model.named_parameters() if n.startswith("encoder.") and p.grad is not None)**.5
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(grad_norm) or encoder_norm==0:raise RuntimeError("Invalid encoder gradient")
                optimizer.step()
                record={**sample,"branch":name,"active_roi_ratio":ratio if name!="M_RD" and mask is not None else 1,"loss":loss.item(),"bpp_estimate":rd["bpp"].mean().item(),"distortion":stats[distortion_key].mean().item(),
                        "gradient_norm_before_clip":grad_norm.item(),"encoder_gradient_norm":encoder_norm,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),**diagnostic}
                log.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n");log.flush();records.append(record)
                if sample["step"]%64==0:print(name,sample["step"],"loss",loss.item(),"peakMiB",torch.cuda.max_memory_allocated()/2**20,flush=True)
                if sample["step"] in (256,512):
                    torch.save({"state_dict":model.state_dict(),"experiment":{"branch":name,"roi_ratio":ratio,"completed_steps":sample["step"],"schedule_sha256":sha256(RUN/"schedule.json"),"protocol_sha256":sha256(RUN/"protocol.json")}},folder/f'checkpoint_{sample["step"]}.ckpt')
        changed=sum(not torch.equal(initial[n],p.detach().cpu()) for n,p in model.named_parameters())
        if changed==0:raise RuntimeError("No actual parameter updates")
        save_json(folder/"result.json",{"success":True,"branch":name,"steps":len(records),"changed_parameter_tensors":changed,"seconds":time.perf_counter()-start,"checkpoint_sha256":sha256(folder/"checkpoint_512.ckpt"),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),"schedule_sha256":sha256(RUN/"schedule.json"),"environment":environment(),"loss_attribution_note":"weighted MSE contributions are additive separately within Y/U/V; native overall geometric distortion is nonlinear and is not falsely decomposed into a summed ROI/background loss"})
    except Exception as exc:
        save_json(folder/"failure.json",{"error":repr(exc),"completed_steps":len(records)});raise
    del model,optimizer
    torch.cuda.empty_cache()
    print("TRAINING_OK",name,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["schedule","loss_check","cache","train"]);parser.add_argument("--branch",choices=["M_RD","M_ROI4","M_ROI16","all"],default="all")
    args=parser.parse_args()
    if args.mode=="schedule":make_schedule()
    elif args.mode=="loss_check":check_loss_equivalence()
    elif args.mode=="cache":make_cache()
    else:
        for name,ratio in (("M_RD",1),("M_ROI4",4),("M_ROI16",16)):
            if args.branch in (name,"all"):train_branch(name,ratio)


if __name__=="__main__":main()
