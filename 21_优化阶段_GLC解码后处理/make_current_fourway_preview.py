"""Build the accepted I12 four-way view with a complete-file matched HEVC control."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cache_hevc_baseline import X265_PARAMS
from coral_reference_sidecar import file_sha, load_json, save_json
from experiment_common import array_sha


def match_hevc(ffmpeg, source, budget, folder):
    """Search only on file bytes; never pad or select by visual scores."""
    folder.mkdir(exist_ok=False)
    common = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin']
    pixels = source.tobytes()
    candidates = {}

    def encode(tick):
        if tick not in candidates:
            path = folder / f'crf_tick_{tick}.mp4'
            command = common + ['-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', '512x288', '-framerate', '15', '-i', 'pipe:0', '-an',
                '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
                '-color_range', 'tv', '-colorspace', 'bt709', '-color_primaries', 'bt709',
                '-color_trc', 'bt709', '-c:v', 'libx265', '-preset', 'medium',
                '-crf', str(tick/16), '-x265-params', X265_PARAMS, '-tag:v', 'hvc1',
                '-movflags', '+faststart', '-n', str(path)]
            with (folder / f'crf_tick_{tick}.log').open('wb') as log:
                subprocess.run(command, input=pixels, stdout=subprocess.DEVNULL, stderr=log, check=True)
            candidates[tick] = (path.stat().st_size, path)
            save_json(folder/'search_progress.json', {str(q/16):v[0] for q,v in sorted(candidates.items())})
            print('H265 CRF', tick/16, 'bytes', path.stat().st_size, 'target', budget, flush=True)
        return candidates[tick][0]

    low, high = 28*16, 51*16
    if encode(low) <= budget or encode(high) > budget:
        raise ValueError('Frozen HEVC CRF bounds do not bracket the byte budget')
    while high-low > 1:
        mid = (low+high)//2
        if encode(mid) <= budget:
            high = mid
        else:
            low = mid
        best = max(size for size,_ in candidates.values() if size <= budget)
        if (budget-best)/budget <= .005:
            break
    best = max(size for size,_ in candidates.values() if size <= budget)
    if (budget-best)/budget > .005:
        for tick in range(max(28*16, high-4), min(51*16, high+4)+1):
            encode(tick)
            best = max(size for size,_ in candidates.values() if size <= budget)
            if (budget-best)/budget <= .005:
                break
    tick,(size,candidate) = max(((q,v) for q,v in candidates.items() if v[0] <= budget), key=lambda x:x[1][0])
    if (budget-size)/budget > .02:
        raise ValueError('HEVC differs by more than the frozen 2% complete-file tolerance')
    destination = folder/'h265_matched.mp4'
    shutil.copy2(candidate,destination)
    decoded = subprocess.check_output(common+['-i',str(destination),'-map','0:v:0',
        '-vf','scale=in_color_matrix=bt709:in_range=tv,format=rgb24','-f','rawvideo','pipe:1'])
    rgb = np.frombuffer(decoded,np.uint8).reshape((150,288,512,3))
    np.save(folder/'rgb.npy',rgb,allow_pickle=False)
    ffprobe = ffmpeg.with_name('ffprobe'+ffmpeg.suffix)
    stream = json.loads(subprocess.check_output([str(ffprobe),'-v','error','-select_streams','v:0',
        '-count_frames','-show_entries','stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration,pix_fmt',
        '-of','json',str(destination)]))['streams'][0]
    assert stream['codec_name']=='hevc' and (stream['width'],stream['height'])==(512,288)
    assert Fraction(stream['avg_frame_rate'])==15 and int(stream['nb_read_frames'])==150 and float(stream['duration'])==10
    record = dict(target_bytes=budget,bytes=size,kbps=size*8/10000,crf=tick/16,
        relative_shortfall=(budget-size)/budget,complete_file_billing=True,padding_added=False,
        preset='medium',x265_params=X265_PARAMS,source_rgb_array_sha256=array_sha(source),
        sha256=file_sha(destination),decoded_rgb_file_sha256=file_sha(folder/'rgb.npy'),stream=stream,
        candidates={str(q/16):v[0] for q,v in sorted(candidates.items())})
    save_json(folder/'receipt.json',record)
    return rgb,record


def run(workspace,output,ffmpeg):
    output.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    profile_path=Path(__file__).resolve().with_name('当前采用配置.json')
    profile=load_json(profile_path)
    assert profile['profile_name']=='S8I + CURRENT_I12'
    assert profile['anchor_frames']==[0,15,45,75,105,135] and profile['raft']['iterations']==12
    data=workspace/'02_实验数据/优化_GLC后处理'
    batch=data/'source_dense_flow_three_clips_20261109_01/video1'
    accepted=data/'anchor_time_20261111_02'
    source_path=batch/'source.npy';raw_path=batch/'base_receive/decoded_rgb.npy'
    optimized_path=Path(profile['artifacts']['rgb']['path'])
    plan=load_json(accepted/'execution_plan.json')
    inputs={str(p):file_sha(p) for p in (source_path,raw_path,optimized_path,
        Path(profile['artifacts']['base']['path']),Path(profile['artifacts']['guide']['path']))}
    for p in (source_path,raw_path):assert inputs[str(p)]==plan['inputs_sha256'][str(p)]
    for name in ('base','guide','rgb'):
        p=Path(profile['artifacts'][name]['path'])
        assert inputs[str(p)]==profile['artifacts'][name]['sha256']
    source=np.load(source_path,mmap_mode='r',allow_pickle=False)
    raw=np.load(raw_path,mmap_mode='r',allow_pickle=False)
    optimized=np.load(optimized_path,mmap_mode='r',allow_pickle=False)
    assert all(v.shape==(150,288,512,3) and v.dtype==np.uint8 for v in (source,raw,optimized))
    base_bytes=Path(profile['artifacts']['base']['path']).stat().st_size
    aux_bytes=Path(profile['artifacts']['guide']['path']).stat().st_size
    budget=base_bytes+aux_bytes
    save_json(output/'execution_plan.json',dict(date='2026-11-11',profile='S8I + CURRENT_I12',
        requested_layout=['source','matched_HEVC','raw_GLC','CURRENT_I12'],frames=150,fps=15,
        width=512,height=288,base_bytes=base_bytes,auxiliary_bytes=aux_bytes,target_HEVC_complete_file_bytes=budget,
        hevc_preset='medium',x265_params=X265_PARAMS,rate_search='CRF 28..51, 1/16 steps; byte-only selection; aim <=0.5% shortfall, allow <=2%; no padding',
        source_used_only_for_HEVC_and_display=True,GLC_and_I12_rerun=False,inputs_sha256=inputs,
        code_sha256=file_sha(__file__)))
    hevc,matched=match_hevc(ffmpeg,source,budget,output/'matched_hevc')
    videos=[source,hevc,raw,optimized]
    base_rate,aux_rate=base_bytes*8/10000,aux_bytes*8/10000
    total_rate=budget*8/10000
    width,height,header_height=1024,688,56
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    headings=['原视频',f'H.265   {matched["kbps"]:.2f} kbps',
        f'原始 GLC   {base_rate:.2f} kbps',f'当前优化 GLC   {total_rate:.2f} kbps']
    notes=['video1 · 15 FPS · 10 秒','按完整文件匹配优化版总码率','仅本体码流',
        f'本体 {base_rate:.2f} + 辅助 {aux_rate:.2f} kbps | RAFT 12']
    canvas=Image.new('RGB',(width,height),(20,29,40));draw=ImageDraw.Draw(canvas)
    positions=[]
    for i,(title,note) in enumerate(zip(headings,notes,strict=True)):
        x,y=(i%2)*512,(i//2)*(288+header_height)
        assert draw.textbbox((0,0),title,font=font)[2]<492
        assert draw.textbbox((0,0),note,font=small)[2]<492
        draw.text((x+10,y+4),title,fill=(239,246,252),font=font)
        draw.text((x+10,y+31),note,fill=(169,198,218),font=small)
        positions.append((x,y+header_height))
    path=output/'video1_四宫格_原片_H265_原GLC_当前优化I12.mp4'
    command=[str(ffmpeg),'-hide_banner','-loglevel','error','-nostdin','-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{width}x{height}','-framerate','15','-i','pipe:0','-an',
        '-vf','scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
        '-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
        '-c:v','libx264','-preset','fast','-crf','12','-threads','4','-movflags','+faststart','-n',str(path)]
    with (output/'preview.ffmpeg.log').open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for frame in range(150):
                for video,position in zip(videos,positions,strict=True):canvas.paste(Image.fromarray(video[frame]),position)
                process.stdin.write(canvas.tobytes())
            process.stdin.close()
            if process.wait(timeout=60):raise RuntimeError('Four-way preview encoding failed')
        finally:
            if process.poll() is None:process.kill();process.wait()
    probe=json.loads(subprocess.check_output([str(ffmpeg.with_name('ffprobe'+ffmpeg.suffix)),'-v','error',
        '-select_streams','v:0','-count_frames','-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration',
        '-of','json',str(path)]))['streams'][0]
    assert int(probe['nb_read_frames'])==150 and Fraction(probe['avg_frame_rate'])==15
    assert (probe['width'],probe['height'])==(width,height) and float(probe['duration'])==10
    decoded=subprocess.check_output([str(ffmpeg),'-v','error','-i',str(path),
        '-vf','scale=in_color_matrix=bt709:in_range=tv,format=rgb24','-f','rawvideo','pipe:1'])
    frames=np.frombuffer(decoded,np.uint8).reshape(150,height,width,3)
    errors=[]
    for video,(x,y) in zip(videos,positions,strict=True):
        differences=[float(np.abs(frames[i,y:y+288,x:x+512].astype(np.float32)-video[i]).mean()) for i in (0,75,149)]
        assert max(differences)<6
        errors.append(differences)
    # Record exact pre-encode tile identities as well as post-encode visual fidelity.
    for i in (0,75,149):
        for video,position in zip(videos,positions,strict=True):canvas.paste(Image.fromarray(video[i]),position)
        assembled=np.asarray(canvas)
        for video,(x,y) in zip(videos,positions,strict=True):assert np.array_equal(assembled[y:y+288,x:x+512],video[i])
        Image.fromarray(frames[i]).save(output/f'verified_frame{i:03d}.png')
    assert all(file_sha(Path(p))==h for p,h in inputs.items())
    save_json(output/'verification.json',dict(complete=True,profile='S8I + CURRENT_I12',frames=150,fps=15,seconds=10,
        layout=['source','matched_HEVC','raw_GLC','CURRENT_I12'],headings=headings,notes=notes,probe=probe,
        base_bytes=base_bytes,auxiliary_bytes=aux_bytes,total_bytes=budget,base_kbps=base_rate,
        auxiliary_kbps=aux_rate,optimized_total_kbps=total_rate,matched_hevc=matched,
        pixel_panel_origins=positions,check_frames=[0,75,149],preencode_tiles_exact=True,panel_mae=errors,
        inputs_sha256=inputs,inputs_unchanged=True,labels_use_complete_transmission_rates=True,
        viewing_encode='libx264 fast CRF12 YUV420; preview size is not transmission rate',
        output=str(path),output_sha256=file_sha(path),code_sha256=file_sha(__file__),
        visual_layout_review='pending',user_perceptual_review='not yet requested or recorded for this new four-way view'))
    shutil.copy2(__file__,output/Path(__file__).name)
    print('COMPLETE',str(path),'HEVC kbps',matched['kbps'],'optimized kbps',total_rate,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ffmpeg',type=Path,required=True)
    args=parser.parse_args()
    run(args.workspace.resolve(),args.output.resolve(),args.ffmpeg.resolve())
