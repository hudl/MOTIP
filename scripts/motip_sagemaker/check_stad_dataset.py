#!/usr/bin/env python3
"""
Validate the STAD v2 tracking dataset on S3.
Checks each clip for: seqinfo.ini with [Sequence] section, gt/gt.txt, and at least one image.

Usage (from devcontainer):
    python scripts/check_stad_dataset.py
    python scripts/check_stad_dataset.py --fix   # re-generate missing seqinfo.ini from gt.txt
"""

import argparse
import sys
from configparser import ConfigParser
from io import StringIO

import boto3

BUCKET = "hudl-experiments"
PREFIX = "touchdown/datasets/tracking_stad_v2/train/"


def list_clips(s3, bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    clips = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            clip = p["Prefix"][len(prefix):].rstrip("/")
            if clip:
                clips.append(clip)
    return sorted(clips)


def check_seqinfo(s3, bucket, prefix, clip):
    key = f"{prefix}{clip}/seqinfo.ini"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read().decode("utf-8")
        ini = ConfigParser()
        ini.read_string(content)
        if "Sequence" not in ini:
            return "seqinfo.ini missing [Sequence] section", content
        # Check required keys
        for k in ("imWidth", "imHeight", "seqLength"):
            if k.lower() not in ini["Sequence"]:
                return f"seqinfo.ini missing key {k}", content
        return None, content
    except s3.exceptions.NoSuchKey:
        return "seqinfo.ini missing", None
    except Exception as e:
        return f"seqinfo.ini error: {e}", None


def check_gt(s3, bucket, prefix, clip, seq_len=None):
    key = f"{prefix}{clip}/gt/gt.txt"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        if seq_len is None:
            return None
        # Check max frame_id doesn't exceed seqlength
        content = obj["Body"].read().decode("utf-8")
        max_frame = 0
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            try:
                max_frame = max(max_frame, int(parts[0]))
            except (ValueError, IndexError):
                return "gt/gt.txt has malformed rows"
        if max_frame > seq_len:
            return f"gt/gt.txt max frame_id {max_frame} > seqlength {seq_len}"
        if max_frame == 0:
            return "gt/gt.txt is empty"
        return None
    except s3.exceptions.NoSuchKey:
        return "gt/gt.txt missing"
    except Exception as e:
        return f"gt/gt.txt error: {e}"


def check_images(s3, bucket, prefix, clip):
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}{clip}/img1/", MaxKeys=5):
        count += len(page.get("Contents", []))
        if count > 0:
            break
    if count == 0:
        return "img1/ has no images"
    return None


def make_seqinfo(clip, seq_length, width=1920, height=1080, framerate=30):
    return (
        f"[Sequence]\n"
        f"name={clip}\n"
        f"imdir=img1\n"
        f"framerate={framerate}\n"
        f"seqlength={seq_length}\n"
        f"imwidth={width}\n"
        f"imheight={height}\n"
        f"imext=.jpg\n"
    )


def count_images(s3, bucket, prefix, clip):
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}{clip}/img1/"):
        count += len(page.get("Contents", []))
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Re-generate seqinfo.ini for bad clips")
    parser.add_argument("--prefix", default=PREFIX)
    args = parser.parse_args()

    s3 = boto3.client("s3")
    print(f"Listing clips under s3://{BUCKET}/{args.prefix} ...")
    clips = list_clips(s3, BUCKET, args.prefix)
    print(f"Found {len(clips)} clips\n")

    bad = []
    for i, clip in enumerate(clips):
        if i % 100 == 0:
            print(f"  [{i}/{len(clips)}] checking ...", flush=True)

        issues = []
        seqinfo_issue, seqinfo_content = check_seqinfo(s3, BUCKET, args.prefix, clip)
        if seqinfo_issue:
            issues.append(seqinfo_issue)
        seq_len = None
        if seqinfo_content:
            ini = ConfigParser()
            ini.read_string(seqinfo_content)
            if "Sequence" in ini:
                try:
                    seq_len = int(ini["Sequence"]["seqlength"])
                except (KeyError, ValueError):
                    pass
        gt_issue = check_gt(s3, BUCKET, args.prefix, clip, seq_len=seq_len)
        if gt_issue:
            issues.append(gt_issue)
        img_issue = check_images(s3, BUCKET, args.prefix, clip)
        if img_issue:
            issues.append(img_issue)

        if issues:
            bad.append((clip, issues))

    print(f"\n{'='*60}")
    if not bad:
        print(f"OK — all {len(clips)} clips are valid")
        return

    print(f"FOUND {len(bad)} bad clips out of {len(clips)}:\n")
    for clip, issues in bad:
        print(f"  {clip}: {', '.join(issues)}")

    if args.fix:
        print(f"\nFixing {len(bad)} clips ...")
        fixed = 0
        for clip, issues in bad:
            if any("seqinfo" in i for i in issues) and not any("img1" in i for i in issues):
                n_images = count_images(s3, BUCKET, args.prefix, clip)
                if n_images == 0:
                    print(f"  SKIP {clip} — no images, can't infer seqlength")
                    continue
                content = make_seqinfo(clip, seq_length=n_images)
                key = f"{args.prefix}{clip}/seqinfo.ini"
                s3.put_object(Bucket=BUCKET, Key=key, Body=content.encode("utf-8"))
                print(f"  FIXED {clip} (seqlength={n_images})")
                fixed += 1
        print(f"\nFixed {fixed}/{len(bad)} clips")
    else:
        print(f"\nRe-run with --fix to auto-generate seqinfo.ini for clips where images exist.")


if __name__ == "__main__":
    main()
