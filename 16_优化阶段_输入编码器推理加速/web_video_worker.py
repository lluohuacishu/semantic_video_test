"""Full-video web worker: chunked frames, bounded streams, independent original decoder."""
from __future__ import annotations
import argparse
import sys
import time
from types import SimpleNamespace
from experiment_common import *
from c2_watch import write
from web_frames import FrameWriter,Frames,even_rgb
from web_transport import segment_ranges,package,unpack


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    cfg=load_json(args.config);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    stop=Path(cfg['stop_file']);started=time.perf_counter()
    def check():
        if stop.exists():raise InterruptedError('用户取消')
    def progress(phase,index=0,total=0):
        check();write(out/'progress.json',dict(phase=phase,index=index,total=total,elapsed=time.perf_counter()-started))
    try:
        from c1_io import MemoryFrames,RawPreview
        if cfg['action']=='prepare':
            from PIL import Image
            progress('读取完整视频' if cfg.get('frames') is None else '准备测试片段')
            limit=cfg.get('frames');reader=None;frames=preview=None
            stride=2 if cfg.get('fps_mode','original')=='half' else 1
            if cfg.get('sample'):
                sample=cfg['sample']
                if file_sha(sample['source'])!=sample['source_sha256']:raise ValueError('示例源文件变化')
                array=np.load(sample['source'],mmap_mode='r')[::stride];source_fps=sample['fps'];fps=source_fps/stride
                expected=min(len(array),limit) if limit else len(array);iterator=iter(array[:expected])
            else:
                import cv2
                probe=cv2.VideoCapture(cfg['input'])
                try:w,h=probe.get(cv2.CAP_PROP_FRAME_WIDTH),probe.get(cv2.CAP_PROP_FRAME_HEIGHT)
                finally:probe.release()
                if min(w,h)<=0:raise ValueError('无法识别视频尺寸')
                width=max(1,round(min(w,cfg['size']*w/max(w,h))))
                reader=MemoryFrames(SimpleNamespace(input=Path(cfg['input']),width=width,frames=limit or 2**31-1,start_frame=0,frame_stride=stride),collect_hashes=False)
                source_fps=reader.fps;fps=source_fps/stride;available=(reader.reported+stride-1)//stride
                expected=min(available,limit) if limit and available>0 else (available or limit or 0)
                iterator=iter(reader)
            try:
                for i,rgb in enumerate(iterator):
                    check();h,w=rgb.shape[:2]
                    if max(w,h)>cfg['size']:
                        scale=cfg['size']/max(w,h);rgb=np.asarray(Image.fromarray(rgb).resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.LANCZOS));h,w=rgb.shape[:2]
                    rgb=even_rgb(rgb);h,w=rgb.shape[:2]
                    if frames is None:
                        frames=FrameWriter(out/'source_chunks',w,h)
                        preview=RawPreview(load_settings()['runtime']['ffmpeg'],out/'source.mp4',w,h,fps)
                    frames.add(rgb);preview.add(rgb,check)
                    if i%16==0:progress('读取完整视频' if limit is None else '准备测试片段',i+1,expected)
                if frames is None:raise ValueError('视频没有可读取的画面')
                storage=frames.finish();preview.finish(check)
            except BaseException:
                if preview:preview.abort()
                raise
            finally:
                if reader:reader.close()
            storage.update(fps=fps,source_fps=source_fps,frame_stride=stride,duration_seconds=storage['frames']/fps,seconds=time.perf_counter()-started,
                source_video_sha256=file_sha(out/'source.mp4'),timebase='nominal constant frame order',read_until_eof=limit is None)
            write(out/'result.json',storage);return 0
        import torch
        from role_codec import RoleCodec,ref
        from web_engine_cache import build_cached
        from sender_container import PROFILE,SenderWriter,ReferenceChecksums,read_sender_stream
        role=cfg['action'];progress('加载发送端模型' if role=='sender' else '加载原接收端模型')
        tick=time.perf_counter();codec=RoleCodec(cfg['assets'],role);torch.utils.deterministic.fill_uninitialized_memory=False
        torch.cuda.synchronize();load_seconds=time.perf_counter()-tick;setup={};variant=cfg.get('variant','eager')
        source_frames=None
        if role=='sender':
            source=cfg['source'];source_frames=Frames(source);total=source['frames'];media={k:source[k] for k in ['width','height','fps']};media['q']=cfg['q']
            ranges=segment_ranges(total,cfg.get('segment_frames',4096))
            def make_header(n):
                header=dict(media,frames=n,intra_period=64,reset_interval=32,revision=ref.REVISION,entropy=ref.ENTROPY_ID,
                    table_sha256=codec.entropy.table_sha256,arithmetic=codec.profile,
                    weights={k:v['sha256'] for k,v in ref.ASSETS.items()},container_version=3,integrity_profile=PROFILE,sender_rgb_reconstructed=False)
                if variant!='eager':header['input_encoder_profile']={'amp_fp16':'pt251-amp16','trt_fp32_stream':'trt105-f32','trt_fp16_stream':'trt105-f16'}[variant]
                return header
            first=make_header(ranges[0][1]);packets=None
            if variant!='eager':
                example=input_tensor(source_frames[0]);geometry=f'{example.shape[-1]}x{example.shape[-2]}'
                with torch.inference_mode():
                    for name in ['image','video']:
                        progress('准备'+('I' if name=='image' else 'P')+'帧加速器（首次可能需要构建）')
                        target=getattr(codec,name).vqgan
                        wrapped=build_cached(target.encoder,example,variant,Path(cfg['engine_cache'])/geometry/name,out/'engine_staging'/name,check)
                        target.encoder=wrapped;setup[name]=wrapped.setup_seconds
        else:
            root=out.resolve()
            def audit(event,values):
                if event=='open' and values and isinstance(values[0],(str,bytes)):
                    f=Path(str(values[0]));name=str(f).lower()
                    if name.endswith(('.npy','.png','.jpg','.jpeg','.mp4','.mkv','.avi')) and not f.resolve().is_relative_to(root):
                        raise RuntimeError('独立接收端禁止读取源画面')
            sys.addaudithook(audit)
            paths,bundle=unpack(cfg['input'],out/'received_segments');first,packets,first_integrity=read_sender_stream(paths[0])
            total=bundle['frames'] if bundle else first['frames'];media={k:first[k] for k in ['width','height','fps','q']}
        def validate(header):
            if any(header[k]!=media[k] for k in media) or header['arithmetic']!=codec.profile or header['table_sha256']!=codec.entropy.table_sha256:
                raise ValueError('接收算术/片段参数不匹配')
            if header['revision']!=ref.REVISION or header['entropy']!=ref.ENTROPY_ID or header['weights']!={k:v['sha256'] for k,v in ref.ASSETS.items()}:
                raise ValueError('接收模型版本/权重/熵模型不匹配')
        validate(first);warm=time.perf_counter();progress('模型暖机',0,total)
        for i in range(min(8,first['frames'])):
            check()
            if role=='sender':codec.frame(first,i,x=input_tensor(source_frames[i]),render_rgb=False)
            else:codec.frame(first,i,packet=packets[i])
        torch.cuda.synchronize();warm_seconds=time.perf_counter()-warm;references=ReferenceChecksums();records=[];core=0.;last=0.;segments=[]
        preview=RawPreview(load_settings()['runtime']['ffmpeg'],out/'video.mp4',media['width'],media['height'],media['fps']) if role=='receiver' else None
        decoded=FrameWriter(out/'decoded_chunks',media['width'],media['height']) if role=='receiver' else None
        operation=time.perf_counter();offset=0
        try:
            groups=ranges if role=='sender' else paths
            for group_index,group in enumerate(groups):
                check();codec.dpb=dict(ref_latent=None,ref_y=None)
                if role=='sender':
                    begin,end=group;header=make_header(end-begin)
                    path=out/'stream.glcsnd' if len(ranges)==1 else out/'segments'/f'segment_{group_index:06d}.glcsnd'
                    path.parent.mkdir(exist_ok=True);writer=SenderWriter(path,header);local_packets=None
                else:
                    header,local_packets,integrity=(first,packets,first_integrity) if group_index==0 else read_sender_stream(group)
                    validate(header);writer=None;local_refs=ReferenceChecksums()
                try:
                    for i in range(header['frames']):
                        check();index=offset+i;x=input_tensor(source_frames[index]) if role=='sender' else None
                        torch.cuda.synchronize();tick=time.perf_counter()
                        rgb,packet=codec.frame(header,i,x=x,render_rgb=False) if role=='sender' else codec.frame(header,i,packet=local_packets[i])
                        torch.cuda.synchronize();elapsed=time.perf_counter()-tick;core+=elapsed
                        if writer:writer.add(packet)
                        else:decoded.add(rgb);preview.add(rgb,check);local_refs.add(i,packet)
                        references.add(index,packet)
                        if len(records)<128:records.append(dict(index=index,kind=packet['kind'],seconds=elapsed,latent_sha256=packet['latent_sha256']))
                        if time.monotonic()-last>.5 or index==total-1:
                            progress('编码中' if role=='sender' else '独立解码中',index+1,total);last=time.monotonic()
                    if writer:
                        writer.finish();segments.append(dict(path=str(path),frames=header['frames']))
                    else:local_refs.verify(integrity['expected_reference_checksums'])
                    offset+=header['frames']
                finally:
                    if writer:writer.close()
            if offset!=total:raise ValueError('视频帧数不完整，禁止发布')
            if role=='sender':stream=package(out,segments,dict(media,frames=total))
            else:decoded_meta=decoded.finish();preview.finish(check);stream=Path(cfg['input'])
        except BaseException:
            if preview:preview.abort()
            raise
        finally:
            if source_frames:source_frames.close()
        result=dict(action=role,variant=variant,frames=total,media=media,segments=len(groups),core_seconds=core,
            model_load_seconds=load_seconds,engine_setup_seconds=sum(setup.values()),warmup_seconds=warm_seconds,
            file_operation_seconds=time.perf_counter()-operation,worker_seconds=time.perf_counter()-started,
            fps=total/core,records=records,records_truncated=total>len(records),reference_checksums=references.values(),
            stream_sha256=file_sha(stream),stream_file=stream.name,stream_bytes=stream.stat().st_size,
            reference_verified=role=='receiver',source_media_reads_forbidden=role=='receiver')
        if role=='receiver':result.update(video_sha256=file_sha(out/'video.mp4'),decoded_storage=decoded_meta)
        write(out/'result.json',result);return 0
    except InterruptedError:
        write(out/'cancelled.json',dict(cancelled=True));return 2


if __name__=='__main__':raise SystemExit(main())
