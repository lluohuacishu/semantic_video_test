"""Replace verbose research provenance by a small encoder ID; preserve every payload byte."""
from __future__ import annotations
import argparse
from experiment_common import *
from sender_container import SenderWriter,read_sender_stream
from trial_worker import payload_fingerprint

PROFILES={'amp_fp16':'pt251-amp16','trt_fp32_stream':'trt105-f32','trt_fp16_stream':'trt105-f16'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;rows=[]
    for item in load_json(out/'full_summary.json')['reports']:
        cid,variant=item['clip'],item['variant'];sender=item['sender_repeats'][0]
        source=out/'full'/cid/variant/'sender1/stream.glcsnd'
        if file_sha(source)!=sender['stream_sha256']:raise ValueError('Source stream changed')
        header,packets,integrity=read_sender_stream(source);header.pop('input_encoder_experiment',None)
        if variant!='eager':header['input_encoder_profile']=PROFILES[variant]
        folder=out/'compact'/cid/variant;folder.mkdir(parents=True,exist_ok=False)
        target=folder/'stream.glcsnd';writer=SenderWriter(target,header)
        try:
            for index,(packet,record) in enumerate(zip(packets,sender['records'])):
                if payload_fingerprint(packet)!=record['payload_sha256']:raise ValueError('Payload differs before repack')
                writer.add(dict(kind=packet['kind'],z=packet['z'],y=packet['y'],latent_sha256=record['latent_sha256']))
            writer.finish()
        finally:writer.close()
        new_header,new_packets,new_integrity=read_sender_stream(target)
        if new_header!=header or new_integrity['expected_reference_checksums']!=integrity['expected_reference_checksums']:
            raise ValueError('Header/reference repack changed unexpectedly')
        if len(new_packets)!=len(packets) or any(payload_fingerprint(a)!=payload_fingerprint(b) for a,b in zip(packets,new_packets)):
            raise ValueError('Repack changed compressed payloads')
        if variant=='eager' and file_sha(target)!=sender['stream_sha256']:raise ValueError('Identity repack not bit exact')
        base=next(r for r in load_json(out/'full_summary.json')['reports'] if r['clip']==cid and r['variant']=='eager')['sender_repeats'][0]
        rows.append(dict(clip=cid,variant=variant,profile=PROFILES.get(variant),path=str(target),
            source_sha256=sender['stream_sha256'],sha256=file_sha(target),bytes=target.stat().st_size,
            raw_research_bytes=source.stat().st_size,baseline_bytes=base['stream_bytes'],
            payload_bytes=sender['payload_bytes'],file_ratio_vs_baseline=target.stat().st_size/base['stream_bytes'],
            research_file_ratio_vs_baseline=source.stat().st_size/base['stream_bytes'],
            payload_ratio_vs_baseline=sender['payload_bytes']/base['payload_bytes'],payload_byte_exact_to_source=True,
            ordered_reference_checksums_exact_to_source=True,neural_computation_performed=False))
    save_json(out/'compact_transport.json',dict(complete=True,reports=rows,
        scope='Header-only provenance reduction; unchanged symbols, payloads and reference footer. Not a neural codec speedup.',
        profile_definitions=PROFILES,verbose_provenance_retained_in_sidecar=True))
    print('Repacked',len(rows),'streams; all payload bytes and ordered reference states preserved',flush=True)


if __name__=='__main__':main()
