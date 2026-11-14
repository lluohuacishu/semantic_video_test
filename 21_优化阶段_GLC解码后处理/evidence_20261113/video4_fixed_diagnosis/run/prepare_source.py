import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[2]
SRC=Path(r'C:\Users\lenovo\Desktop\临时\语义\data\video4.mp4')
FF=WORK/'01_项目/运行环境/FFmpeg/ffmpeg.exe'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
assert not (ROOT/'case_registration.json').exists()
h=sha(SRC)
assert h=='1f44b3e49ef603aa69fead3b98523dc61a19bd806f301cb758c5fe0b41aca92c'
probe=json.loads(subprocess.check_output([str(FF.with_name('ffprobe.exe')),'-v','error','-select_streams','v:0',
    '-show_entries','stream=codec_name,width,height,avg_frame_rate,nb_frames,duration,pix_fmt:format=duration,size','-of','json',str(SRC)]))
save(ROOT/'source_registration_before_extraction.json',dict(date='2026-11-13',
    authorization='好了，我把video4整过去了，你可以开始下一轮了',
    source=str(SRC),sha256=h,bytes=SRC.stat().st_size,probe=probe,
    selection='Entire user-provided clip, no output-based subclip selection',
    extraction='fps=15,scale=512:288:flags=area; full duration; no padding or time stretching',
    rate_duration='Actual resampled frame count / 15, with source duration separately disclosed',
    intended_methods=['raw GLC','received guide','Q9_R25','GS25'],
    parameter_tuning=False,independent_holdout=False,content='User-selected dense-coral failure clip',
    former_source_location_question_resolved=True))
data=subprocess.check_output([str(FF),'-v','error','-i',str(SRC),'-map','0:v:0','-an',
    '-vf','fps=15,scale=512:288:flags=area','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
assert len(data)%(288*512*3)==0
arr=np.frombuffer(data,np.uint8).reshape(-1,288,512,3)
assert 100<=len(arr)<=105
assert abs(len(arr)/15-float(probe['streams'][0]['duration']))<=1/15
np.save(ROOT/'source.npy',arr,allow_pickle=False)
assert sha(SRC)==h
save(ROOT/'case_registration.json',dict(date='2026-11-13',case='video4',source_video=str(SRC),
    source_video_sha256=h,source_npy=str(ROOT/'source.npy'),source_npy_sha256=sha(ROOT/'source.npy'),
    video_stream_seconds=float(probe['streams'][0]['duration']),audio_container_seconds=float(probe['format']['duration']),
    frames=len(arr),fps=15,width=512,height=288,billing_seconds=len(arr)/15,
    selection='Entire clip, no frame padding; timestamp resampling, not slower playback',
    distinct_adjacent_frames=int(np.any(arr[1:]!=arr[:-1],axis=(1,2,3)).sum()),
    prepare_sha256=sha(Path(__file__))))
ids=sorted(set([0,len(arr)//4,len(arr)//2,3*len(arr)//4,len(arr)-1]))
canvas=Image.new('RGB',(512*len(ids),314),(24,35,48));d=ImageDraw.Draw(canvas)
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
for j,i in enumerate(ids):
    d.text((512*j+5,4),f'video4 source f{i} {i/15:.2f}s',font=font,fill='white')
    canvas.paste(Image.fromarray(arr[i]),(j*512,26))
canvas.save(ROOT/'source_contact.png')
print(json.dumps(dict(source=str(SRC),frames=len(arr),fps=15,billing_seconds=len(arr)/15,
    source_seconds=probe['streams'][0]['duration'],sha256=h),ensure_ascii=False))

