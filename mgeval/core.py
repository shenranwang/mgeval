# coding:utf-8
"""core.py
Include feature extractor and musically informed objective measures.
"""
import pretty_midi
import numpy as np
import sys
import os
import symusic
# import midi
import glob
import pandas as pd
import math
import muspy
from mgeval.utils import count_n_consecutive_values, find_closest_value, categorize_tone


data_path = f'{os.getcwd()}/../data/raw'
def get_commu_metadata():
    commu_path = f'{data_path}/ComMU'
    commu_meta = pd.read_csv(f'{commu_path}/commu_meta.csv', index_col="id")
    commu_meta = commu_meta.drop(columns=['Unnamed: 0'])
    return commu_meta
commu_meta = get_commu_metadata()


# feature extractor
def extract_feature(_file):
    """
    This function extracts two midi feature:
    pretty_midi object: https://github.com/craffel/pretty-midi
    midi_pattern: https://github.com/vishnubob/python-midi

    Returns:
        dict(pretty_midi: pretty_midi object,
             midi_pattern: midi pattern contains a list of tracks)
    """
    commu_id = _file.split("/")[-1].split("_")[-1][:10]
    pm = pretty_midi.PrettyMIDI(_file)
    mp = muspy.from_pretty_midi(pm, resolution=24)
    del mp[1]
    feature = {
        'pretty_midi': pm,
        'muspy': mp,
        'muspy_chords': muspy.from_pretty_midi(pm, resolution=24),
        'commu_meta': commu_meta.loc[commu_id],
        'filename': _file.split("/")[-1],
    }
    return feature


# musically informed objective measures.
class metrics(object):
    def improvisor_ctnctr(self, feature):
        pm_object = feature['pretty_midi']
        music = feature['muspy_chords']
        numer = pm_object.time_signature_changes[-1].numerator
        deno = pm_object.time_signature_changes[-1].denominator
        bar_res = int(music.resolution * 4 * numer / deno)
        onset_to_pitch_list_dict = {}
        for note in music[1].notes:
            if note.start not in onset_to_pitch_list_dict:
                onset_to_pitch_list_dict[note.start] = []
            onset_to_pitch_list_dict[note.start].append(note.pitch)
        chord_onsets = list(onset_to_pitch_list_dict.keys())
        curr_chord_idx = 0
        score = 0
        for note in music[0].notes:
            curr_chord_onset = chord_onsets[min(len(chord_onsets)-1, curr_chord_idx)]
            next_chord_onset = chord_onsets[min(len(chord_onsets)-1, curr_chord_idx+1)]
            if note.start >= next_chord_onset:
                while note.start >= next_chord_onset and len(chord_onsets)-1 > curr_chord_idx+1:
                    curr_chord_idx += 1
                    next_chord_onset = chord_onsets[min(len(chord_onsets)-1, curr_chord_idx+1)]
                chord = onset_to_pitch_list_dict[next_chord_onset]
            else:
                chord = onset_to_pitch_list_dict[curr_chord_onset]
            score += categorize_tone(note.start % 12, [p % 12 for p in chord])
        return score / float(len(music[0].notes))
        
    def groove_consistency(self, feature):
        """
        groove_consistency: mean hamming distance of neighboring measures in a sample.
        """
        pm_object = feature['pretty_midi']
        music = feature['muspy']
        numer = pm_object.time_signature_changes[-1].numerator
        deno = pm_object.time_signature_changes[-1].denominator
        bar_res = int(music.resolution * 4 * numer / deno)
        consistency = muspy.groove_consistency(music, bar_res)
        if np.isnan(consistency):
            return 0
        return consistency
    
    def empty_bars_rate(self, feature):
        """
        empty_bars_rate (Pitch count): The ratio of bars that are empty in a sample.
        """
        pm_object = feature['pretty_midi']
        music = feature['muspy']
        numer = pm_object.time_signature_changes[-1].numerator
        deno = pm_object.time_signature_changes[-1].denominator
        bar_res = int(music.resolution * 4 * numer / deno)
        return muspy.empty_measure_rate(music, bar_res)
    
    def empty_beat_rate(self, feature):
        """
        empty_beat_rate (Pitch count): The ratio of beats that are empty in a sample.
        """
        pm_object = feature['pretty_midi']
        music = feature['muspy']
        numer = pm_object.time_signature_changes[-1].numerator
        deno = pm_object.time_signature_changes[-1].denominator
        return muspy.empty_beat_rate(music)
        
    def total_used_pitch(self, feature):
        """
        total_used_pitch (Pitch count): The number of different pitches within a sample.

        Returns:
        'used_pitch': pitch count, scalar for each sample.
        """
        piano_roll = feature['pretty_midi'].instruments[0].get_piano_roll(fs=100)
        sum_notes = np.sum(piano_roll, axis=1)
        used_pitch = np.sum(sum_notes > 0)
        return used_pitch

    def bar_used_pitch(self, feature, track_num=0, num_bar=None):
        """
        bar_used_pitch (Pitch count per bar)

        Args:
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).
        'num_bar': specify the number of bars in the midi pattern, if set as None, round to the number of complete bar.

        Returns:
        'used_pitch': with shape of [num_bar,1]
        """
        pm_object = feature['pretty_midi']
        music = feature['muspy']
        numer = pm_object.time_signature_changes[-1].numerator
        deno = pm_object.time_signature_changes[-1].denominator
        bar_res = int(music.resolution * 4 * numer / deno)
        used_pitch_dict = {i: set() for i in range(num_bar)}
        for note in music[track_num].notes:
            measure, position = divmod(note.time, bar_res)
            if measure >= num_bar:
                break
            used_pitch_dict[measure].add(note.pitch)

        used_pitch = np.zeros((num_bar, 1))
        for i in range(num_bar):
            used_pitch[i] = len(used_pitch_dict[i])

        return used_pitch

    def six_pitch_repetitions(self, feature):
        """
        six_pitch_repetitions (Note count): The number of six repetitions of same pitch.
        """
        music = feature['muspy']
        arr = np.array([note.pitch for note in music[0].notes])
        return count_n_consecutive_values(arr, 6)

    def total_used_note(self, feature, track_num=0):
        """
        total_used_note (Note count): The number of used notes.
        As opposed to the pitch count, the note count does not contain pitch information but is a rhythm-related feature.

        Args:
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).

        Returns:
        'used_notes': a scalar for each sample.
        """
        mp = feature['muspy']
        used_notes = len(mp[track_num].notes)
        return used_notes

    def bar_used_note(self, feature, track_num=0, num_bar=None):
        """
        bar_used_note (Note count per bar).

        Args:
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).
        'num_bar': specify the number of bars in the midi pattern, if set as None, round to the number of complete bar.

        Returns:
        'used_notes': with shape of [num_bar, 1]
        """
        pm_object = feature['pretty_midi']
        music = feature['muspy']
        numer = pm_object.time_signature_changes[-1].numerator
        deno = pm_object.time_signature_changes[-1].denominator
        bar_res = int(music.resolution * 4 * numer / deno)
        used_notes = np.zeros((num_bar, 1))
        for note in music[track_num].notes:
            measure, position = divmod(note.time, bar_res)
            if measure >= num_bar:
                break
            used_notes[measure] += 1
        return used_notes

    def six_duration_repetitions(self, feature):
        """
        six_duration_repetitions (Note count): The number of six repetitions of same duration.
        """
        music = feature['muspy']
        arr = np.array([note.duration for note in music[0].notes])
        return count_n_consecutive_values(arr, 6)

    def total_pitch_class_histogram(self, feature):
        """
        total_pitch_class_histogram (Pitch class histogram):
        The pitch class histogram is an octave-independent representation of the pitch content with a dimensionality of 12 for a chromatic scale.
        In our case, it represents to the octave-independent chromatic quantization of the frequency continuum.

        Returns:
        'histogram': histrogram of 12 pitch, with weighted duration shape 12
        """
        piano_roll = feature['pretty_midi'].instruments[0].get_piano_roll(fs=100)
        histogram = np.zeros(12)
        for i in range(0, 128):
            pitch_class = i % 12
            histogram[pitch_class] += np.sum(piano_roll, axis=1)[i]
        histogram = histogram / sum(histogram)
        return histogram

    def bar_pitch_class_histogram(self, feature, track_num=0, num_bar=None, bpm=120):
        """
        bar_pitch_class_histogram (Pitch class histogram per bar):

        Args:
        'bpm' : specify the assigned speed in bpm, default is 120 bpm.
        'num_bar': specify the number of bars in the midi pattern, if set as None, round to the number of complete bar.
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).

        Returns:
        'histogram': with shape of [num_bar, 12]
        """

        # todo: deal with more than one time signature cases
        pm_object = feature['pretty_midi']
        md_object = feature['commu_meta']
        filename = feature['filename']
        if num_bar is None:
            numer = pm_object.time_signature_changes[-1].numerator
            deno = pm_object.time_signature_changes[-1].denominator
            bar_length = 60. / pm_object.estimate_tempo() * numer * 100
            piano_roll = pm_object.instruments[track_num].get_piano_roll(fs=100)
            piano_roll = np.transpose(piano_roll, (1, 0))
            actual_bar = round(len(piano_roll) / bar_length)
            num_bar = int(round(actual_bar))
            bar_length = int(round(bar_length))
        else:
            numer = pm_object.time_signature_changes[-1].numerator
            deno = pm_object.time_signature_changes[-1].denominator
            bpm_est = round(60. * num_bar * numer * 4 / deno / pm_object.get_end_time())
            bpm_choice = 120 if len(filename.split("_")) == 2 else md_object["bpm"]
            bar_length = 60. / bpm_choice * numer * 100
            piano_roll = pm_object.instruments[track_num].get_piano_roll(fs=100)
            piano_roll = np.transpose(piano_roll, (1, 0))
            actual_bar = len(piano_roll) / bar_length
            bar_length = int(math.ceil(bar_length))

        if actual_bar > num_bar:
            mod = -np.mod(len(piano_roll), bar_length)
            # print(md_object["bpm"], bpm_est, len(pm_object.get_downbeats()), pm_object.get_downbeats(), f"{numer}/{deno}", bar_length, len(piano_roll) / bar_length, actual_bar, piano_roll.shape)
            # print(mod, len(piano_roll) - mod)
            # print(piano_roll[:-np.mod(len(piano_roll), bar_length)].shape, piano_roll.shape)
            piano_roll = piano_roll[:-np.mod(len(piano_roll), bar_length)].reshape((num_bar, -1, 128))  # make exact bar
        elif actual_bar == num_bar:
            piano_roll = piano_roll.reshape((num_bar, -1, 128))
        else:
            piano_roll = np.pad(piano_roll, ((0, int(num_bar * bar_length - len(piano_roll))), (0, 0)), mode='constant', constant_values=0)
            piano_roll = piano_roll.reshape((num_bar, -1, 128))

        bar_histogram = np.zeros((num_bar, 12))
        for i in range(0, num_bar):
            histogram = np.zeros(12)
            for j in range(0, 128):
                pitch_class = j % 12
                histogram[pitch_class] += np.sum(piano_roll[i], axis=0)[j]
            if sum(histogram) != 0:
                bar_histogram[i] = histogram / sum(histogram)
            else:
                bar_histogram[i] = np.zeros(12)
        return bar_histogram

    def pitch_class_transition_matrix(self, feature, normalize=0):
        """
        pitch_class_transition_matrix (Pitch class transition matrix):
        The transition of pitch classes contains useful information for tasks such as key detection, chord recognition, or genre pattern recognition.
        The two-dimensional pitch class transition matrix is a histogram-like representation computed by counting the pitch transitions for each (ordered) pair of notes.

        Args:
        'normalize' : If set to 0, return transition without normalization.
                      If set to 1, normalize by row.
                      If set to 2, normalize by entire matrix sum.
        Returns:
        'transition_matrix': shape of [12, 12], transition_matrix of 12 x 12.
        """
        pm_object = feature['pretty_midi'].instruments[0]
        transition_matrix = pm_object.get_pitch_class_transition_matrix(normalize=True)
        
        if normalize == 0:
            # print(transition_matrix)
            return transition_matrix

        elif normalize == 1:
            sums = np.sum(transition_matrix, axis=1)
            sums[sums == 0] = 1
            return transition_matrix / sums.reshape(-1, 1)

        elif normalize == 2:
            return transition_matrix / sum(sum(transition_matrix))

        else:
            print("invalid normalization mode, return unnormalized matrix")
            return transition_matrix

    def pitch_range(self, feature):
        """
        pitch_range (Pitch range):
        The pitch range is calculated by subtraction of the highest and lowest used pitch in semitones.

        Returns:
        'p_range': a scalar for each sample.
        """
        piano_roll = feature['pretty_midi'].instruments[0].get_piano_roll(fs=100)
        pitch_index = np.where(np.sum(piano_roll, axis=1) > 0)
        p_range = np.max(pitch_index) - np.min(pitch_index)
        return p_range

    '''
    def avg_pitch_shift(self, feature, track_num=1):
        """
        avg_pitch_shift (Average pitch interval):
        Average value of the interval between two consecutive pitches in semitones.

        Args:
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).

        Returns:
        'pitch_shift': a scalar for each sample.
        """
        pattern = feature['midi_pattern']
        pattern.make_ticks_abs()
        resolution = pattern.resolution
        total_used_note = self.total_used_note(feature, track_num=track_num)
        d_note = np.zeros((max(total_used_note - 1, 0)))
        # if total_used_note == 0:
          # return 0
        # d_note = np.zeros((total_used_note - 1))
        current_note = 0
        counter = 0
        for i in range(0, len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                if counter != 0:
                    d_note[counter - 1] = current_note - pattern[track_num][i].data[0]
                    current_note = pattern[track_num][i].data[0]
                    counter += 1
                else:
                    current_note = pattern[track_num][i].data[0]
                    counter += 1
        pitch_shift = np.mean(abs(d_note))
        return pitch_shift
    '''

    def avg_IOI(self, feature):
        """
        avg_IOI (Average inter-onset-interval):
        To calculate the inter-onset-interval in the symbolic music domain, we find the time between two consecutive notes.

        Returns:
        'avg_ioi': a scalar for each sample.
        """

        pm_object = feature['pretty_midi']
        onset = pm_object.get_onsets()
        ioi = np.diff(onset)
        avg_ioi = np.mean(ioi)
        return avg_ioi

    def note_length_hist(self, feature, track_num=0, normalize=True, pause_event=False):
        """
        note_length_hist (Note length histogram):
        To extract the note length histogram, we first define a set of allowable beat length classes:
        [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet].
        The pause_event option, when activated, will double the vector size to represent the same lengths for rests.
        The classification of each event is performed by dividing the basic unit into the length of (barlength)/96, and each note length is quantized to the closest length category.

        Args:
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).
        'normalize' : If true, normalize by vector sum.
        'pause_event' : when activated, will double the vector size to represent the same lengths for rests.

        Returns:
        'note_length_hist': The output vector has a length of either 12 (or 24 when pause_event is True).
        """
        pm_object = feature['pretty_midi']
        music = feature['muspy']
        if pause_event is False:
            note_length_hist = np.zeros((12))
            numer = pm_object.time_signature_changes[-1].numerator
            deno = pm_object.time_signature_changes[-1].denominator
            bar_res = int(music.resolution * 4 * numer / deno)
            unit = bar_res / music.resolution
            hist_list = [
                music.resolution * 4, music.resolution * 2, music.resolution, music.resolution / 2, music.resolution / 4,
                music.resolution * 3, music.resolution * 3 / 2, music.resolution * 3 / 4, music.resolution * 3 / 8,
                music.resolution * 4 / 3, music.resolution * 2 / 3, music.resolution / 3
            ]
            for note in music[track_num].notes:    
                _, idx = find_closest_value(hist_list, note.duration)
                note_length_hist[idx] += 1
        else:
            assert False

        if normalize is False:
            return note_length_hist

        elif normalize is True:
            return note_length_hist / np.sum(note_length_hist)

    def note_length_transition_matrix(self, feature, track_num=0, normalize=0, pause_event=False):
        """
        note_length_transition_matrix (Note length transition matrix):
        Similar to the pitch class transition matrix, the note length tran- sition matrix provides useful information for rhythm description.

        Args:
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).
        'normalize' : If true, normalize by vector sum.
        'pause_event' : when activated, will double the vector size to represent the same lengths for rests.

        'normalize' : If set to 0, return transition without normalization.
                      If set to 1, normalizae by row.
                      If set to 2, normalize by entire matrix sum.

        Returns:
        'transition_matrix': The output feature dimension is 12 × 12 (or 24 x 24 when pause_event is True).
        """
        pm_object = feature['pretty_midi']
        music = feature['muspy']
        if pause_event is False:
            transition_matrix = np.zeros((12, 12))
            numer = pm_object.time_signature_changes[-1].numerator
            deno = pm_object.time_signature_changes[-1].denominator
            bar_res = int(music.resolution * 4 * numer / deno)
            unit = bar_res / music.resolution
            hist_list = [
                music.resolution * 4, music.resolution * 2, music.resolution, music.resolution / 2, music.resolution / 4,
                music.resolution * 3, music.resolution * 3 / 2, music.resolution * 3 / 4, music.resolution * 3 / 8,
                music.resolution * 4 / 3, music.resolution * 2 / 3, music.resolution / 3
            ]
            prev_note = music[track_num].notes[0]
            for note in music[track_num].notes[1:]:
                if prev_note.start != note.start:
                    if note.start - prev_note.end < music.resolution / 4:
                        _, prev_idx = find_closest_value(hist_list, prev_note.duration)
                        _, idx = find_closest_value(hist_list, note.duration)
                        transition_matrix[prev_idx][idx] += 1
        else:
            assert False

        if normalize == 0:
            return transition_matrix
        elif normalize == 1:
            sums = np.sum(transition_matrix, axis=1)
            sums[sums == 0] = 1
            return transition_matrix / sums.reshape(-1, 1)
        elif normalize == 2:
            return transition_matrix / sum(sum(transition_matrix))
        else:
            print("invalid normalization mode, return unnormalized matrix")
            return transition_matrix

    # def chord_dependency(self, feature, bar_chord, bpm=120, num_bar=None, track_num=1):
    #     pm_object = feature['pretty_midi']
    #     # compare bar chroma with chord chroma. calculate the ecludian
    #     bar_pitch_class_histogram = self.bar_pitch_class_histogram(pm_object, bpm=bpm, num_bar=num_bar, track_num=track_num)
    #     dist = np.zeros((len(bar_pitch_class_histogram)))
    #     for i in range((len(bar_pitch_class_histogram))):
    #         dist[i] = np.linalg.norm(bar_pitch_class_histogram[i] - bar_chord[i])
    #     average_dist = np.mean(dist)
    #     return average_dist
