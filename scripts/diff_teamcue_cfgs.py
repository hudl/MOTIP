import sys, json
sys.path.insert(0, ".")
from configs.util import load_super_config
from utils.misc import yaml_to_dict
def res(p):
    c = yaml_to_dict(p); return load_super_config(c, c["SUPER_CONFIG_PATH"])
a = {n: res(f"configs/finetune_teamcue_{n}.yaml") for n in ("t0", "t1", "t2")}
ref = res("configs/finetune_crossing_rfdetr_v1.yaml")
def diff(x, y):
    return {k: (x.get(k, "<unset>"), y.get(k, "<unset>")) for k in sorted(set(x) | set(y)) if x.get(k, "<unset>") != y.get(k, "<unset>")}
print("v1 production config -> T0:"); [print("  ", k, v) for k, v in diff(ref, a["t0"]).items()]
print("T0 -> T1:"); [print("  ", k, v) for k, v in diff(a["t0"], a["t1"]).items()]
print("T1 -> T2:"); [print("  ", k, v) for k, v in diff(a["t1"], a["t2"]).items()]
for k in ("EPOCHS","LR","SAMPLE_LENGTHS","SAMPLE_INTERVALS","SCHEDULER_MILESTONES","REL_PE_LENGTH","MISS_TOLERANCE","BATCH_SIZE","DETR_FREEZE","DETR_CACHE_DIR","AUG_TRAJECTORY_SWITCH_PROB","DATA_ROOT","NUM_ID_DECODER_LAYERS","FEATURE_DIM"):
    print(f"  {k} = {a['t0'].get(k)}")
