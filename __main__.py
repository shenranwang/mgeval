
import json
from argparse import ArgumentParser
import glob
import copy
import os
import numpy as np
import pretty_midi
from tqdm import tqdm
from pprint import pprint
import pickle
from mgeval import core, utils
from sklearn.model_selection import LeaveOneOut

parser = ArgumentParser()
parser.add_argument('--set1dir', required=True, type=str,
                    help='Path (absolute) to the first dataset (folder)')
parser.add_argument('--set2dir', required=True, type=str,
                    help='Path (absolute) to the second dataset (folder)')
parser.add_argument('--outfile', required=True, type=str,
                    help='File (pickle) where the analysis will be stored')

parser.add_argument('--num-bar', required=False, type=int, default=None,
                    help='Number of bars to account for during processing')

args = parser.parse_args()

set1 = glob.glob(os.path.join(args.set1dir, '*.mid*'))
set2 = glob.glob(os.path.join(args.set2dir, '*.mid*'))

def get_commuid(fn):
    commuid = fn.split("/")[-1].split("_")[-1]
    print(commuid, commuid[:10], commuid[10:14])
    return commuid

slices = [["#p-5", "#p-1", "#p3."], ["#p-4", ".mid", "#p4."], ["#p-3", "#p1.", "#p5."], ["#p-2", "#p2.", "#p6."]]
transposition_dict = {get_commuid(file)[:10]: slices[i % len(slices)] for i, file in enumerate(set1)}

set1 = [file for file in set1 if get_commuid(file)[10:14] in transposition_dict[get_commuid(file)[:10]]]
set2 = [file for file in set2 if get_commuid(file)[10:14] in transposition_dict[get_commuid(file)[:10]]]

set1 = [s for s in set1 if len(pretty_midi.PrettyMIDI(s).instruments) == 2]
set2 = [s for s in set2 if len(pretty_midi.PrettyMIDI(s).instruments) == 2]

# filter specific transpositions of files

print('Evaluation sets (sample and baseline):')
print(len(set1), set1[:10])
print(len(set2), set2[:10])

if not any(set1):
  print("Error: sample set it empty")
  exit()

if not any(set2):
  print("Error: baseline set it empty")
  exit()

# Initialize Evaluation Set
num_samples = min(len(set2), len(set1))

print(num_samples)
evalset = {
    # 'avg_pitch_shift': np.zeros((num_samples, 1)), NOT USED
    # 'avg_IOI': np.zeros((num_samples, 1)), NOT USED
    # 'six_pitch_repetitions': np.zeros((num_samples, 1)), NOT USED
    # 'six_duration_repetitions': np.zeros((num_samples, 1)), NOT USED
    'empty_bars_rate': np.zeros((num_samples, 1)),  # NO NEED ANYMORE
    # 'empty_beat_rate': np.zeros((num_samples, 1)),  # NO NEED ANYMORE
    # 'total_used_pitch': np.zeros((num_samples, 1)),
    # 'total_used_note': np.zeros((num_samples, 1)),
    'bar_used_pitch': np.zeros((num_samples, args.num_bar, 1)),
    'bar_used_note': np.zeros((num_samples, args.num_bar, 1)),
    'pitch_range': np.zeros((num_samples, 1)),
    'groove_consistency': np.zeros((num_samples, 1)),
    'improvisor_ctnctr': np.zeros((num_samples, 1)),
    'total_pitch_class_histogram': np.zeros((num_samples, 12)),
    # 'bar_pitch_class_histogram': np.zeros((num_samples, args.num_bar, 12)),
    'note_length_hist': np.zeros((num_samples, 12)),  # TO SEE
    'pitch_class_transition_matrix': np.zeros((num_samples, 12, 12)),
    'note_length_transition_matrix': np.zeros((num_samples, 12, 12)),
}

bar_metrics = [ 'bar_used_pitch', 'bar_used_note', 'bar_pitch_class_histogram' ]

for metric in bar_metrics:
  if not args.num_bar:
    evalset.pop(metric)

# print(evalset)

metrics_list = list(evalset.keys())

single_arg_metrics = (['groove_consistency', 'empty_bars_rate', 'empty_beat_rate', 'improvisor_ctnctr', 'six_pitch_repetitions', 'six_duration_repetitions', 'total_used_pitch', 'avg_IOI', 'total_pitch_class_histogram', 'pitch_range'])

set1_eval = copy.deepcopy(evalset)
set2_eval = copy.deepcopy(evalset)

sets = [ (set1, set1_eval), (set2, set2_eval) ]


# Extract Fetures
for _set, _set_eval in sets:
  for i in tqdm(range(0, num_samples)):
      if len(pretty_midi.PrettyMIDI(_set[i]).instruments) < 2:
          print("File with less than two instruments! Skipping")
          continue
      feature = core.extract_feature(_set[i])
      for metric in metrics_list:
          evaluator = getattr(core.metrics(), metric)
          if metric in single_arg_metrics:
              tmp = evaluator(feature)
          elif metric in bar_metrics:
              # print(metric)
              tmp = evaluator(feature, 0, args.num_bar)
              # print(tmp.shape)
          else:
              tmp = evaluator(feature, 0)
          _set_eval[metric][i] = tmp

loo = LeaveOneOut()
loo.get_n_splits(np.arange(num_samples))
set1_intra = np.zeros((num_samples, len(metrics_list), num_samples - 1))
set2_intra = np.zeros((num_samples, len(metrics_list), num_samples - 1))

print(metrics_list)

# Calculate Intra-set Metrics
print("Calculating Intra-set Metrics", flush=True)
for i, metric in enumerate(metrics_list):
    print(metric, i, flush=True)
    for train_index, test_index in loo.split(np.arange(num_samples)):
        set1_intra[test_index[0]][i] = utils.c_dist(
            set1_eval[metrics_list[i]][test_index], set1_eval[metrics_list[i]][train_index])
        set2_intra[test_index[0]][i] = utils.c_dist(
            set2_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]][train_index])

loo = LeaveOneOut()
loo.get_n_splits(np.arange(num_samples))
sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))

# Calculate Inter-set Metrics
print("Calculating Inter-set Metrics", flush=True)
for i, metric in enumerate(metrics_list):
    print(metric, i, flush=True)
    for train_index, test_index in loo.split(np.arange(num_samples)):
        sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metric][test_index], set2_eval[metric])


plot_set1_intra = np.transpose(
    set1_intra, (1, 0, 2)).reshape(len(metrics_list), -1)
plot_set2_intra = np.transpose(
    set2_intra, (1, 0, 2)).reshape(len(metrics_list), -1)
plot_sets_inter = np.transpose(
    sets_inter, (1, 0, 2)).reshape(len(metrics_list), -1)

print("intra1", plot_set1_intra.shape, set1_intra.shape)
print("intra2", plot_set2_intra.shape, set2_intra.shape)

output = {}
for i, metric in enumerate(metrics_list):
    print('calculating kl of: {}'.format(metric), flush=True)

    mean = np.mean(set1_eval[metric], axis=0).tolist()
    std = np.std(set1_eval[metric], axis=0).tolist()

    if metric == "empty_bars_rate" or metric == "empty_beat_rate" or metric == "total_used_note" or metric == "total_used_pitch":
        print(mean, std)
        output[metric] = [mean, std]
        continue

    intra_mean = np.mean(plot_set1_intra[i])
    intra_std = np.std(plot_set1_intra[i])

    kl1 = utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i])
    ol1 = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
    kl2 = utils.kl_dist(plot_set2_intra[i], plot_sets_inter[i])
    ol2 = utils.overlap_area(plot_set2_intra[i], plot_sets_inter[i])
    print(mean, std)
    print(intra_mean, intra_std)
    print(kl1, ol1)
    print(kl2, ol2)
    output[metric] = [mean, std, intra_mean, intra_std, kl1, ol1, kl2, ol2]


# Save output
if os.path.exists(args.outfile):
    os.remove(args.outfile)

output_file = open(args.outfile, 'w')
json.dump(output, output_file)
output_file.close()

print('Saved output to file: ' + args.outfile, flush=True)
