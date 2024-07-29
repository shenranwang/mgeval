import numpy as np
import json
import os
import glob

def read_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    metrics_files = sorted(glob.glob(os.path.join(f'{os.getcwd()}/obj_metrics_results', '*.json')))
    for f in metrics_files:
        obj_metrics = read_json(f)
        model_name = f.split("/")[-1]
        if "realtime" in model_name:
            # print(model_name)
            for k in obj_metrics:
                if "empty" in k:
                    continue
                    # print(k, obj_metrics[k][:2])
                else:
                    if True: # k == "bar_used_note":
                        data = obj_metrics[k][0:6]
                        print(model_name, k, data)
                        # data = [str(round(d, 3)) for d in data]
                        # print(model_name, k, " & ".join(data))
