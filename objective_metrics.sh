#!/bin/bash

for model_size in s m # xxxs xxs xs xxxxs 
do
    for model_class in Seq2Seq; do  # Musingpro
        for test_setting in full half realtime; do
              model_save_path=$model_class\_$model_size
              echo "Save path: $model_save_path"
              sbatch evaluations.sh ../output/generated_midis/prev_samples/$test_setting/$model_class\_$model_size obj_metrics_results/$model_class\_$model_size\_$test_setting.json
        done
    done
done