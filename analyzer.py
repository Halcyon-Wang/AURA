import librosa
import numpy as np
import scipy.signal
import json
import argparse
import sys
import os
import math

def extract_valence(chroma_matrix, rms_vector=None):
    n_frames = chroma_matrix.shape[1]
    valence = np.zeros(n_frames)
    
    # 0=root, 1=m2, 2=M2, 3=m3, 4=M3, 5=P4, 6=TT, 7=P5, 8=m6, 9=M6, 10=m7, 11=M7
    interval_weights = {
        0: 0.0, 1: -1.5, 2: 0.0, 3: 0.5, 4: 1.5, 5: 0.5,
        6: -1.5, 7: 1.5, 8: 0.5, 9: 1.5, 10: 0.0, 11: -1.5
    }
    
    for i in range(n_frames):
        c = np.nan_to_num(chroma_matrix[:, i])
        max_c = np.max(c)
        
        current_rms = rms_vector[i] if rms_vector is not None else 0.5
        # If signal is very quiet, drag valence into the abyss
        base_fallback = 0.15 if current_rms < 0.05 else 0.5
        
        if max_c < 1e-6:
            valence[i] = 0.15
            continue
            
        c = c / max_c
        
        consonance = 0
        pairs = 0
        for c1 in range(12):
            if c[c1] < 0.2: continue
            for c2 in range(c1 + 1, 12):
                if c[c2] < 0.2: continue
                interval = (c2 - c1) % 12
                weight = c[c1] * c[c2]
                consonance += weight * interval_weights[interval]
                pairs += 1
                
        if pairs > 0:
            val = 0.5 + (consonance / pairs) * 0.25
            valence[i] = np.clip(val, 0.0, 1.0)
        else:
            valence[i] = base_fallback
            
    return valence

def smooth(x, window_length=21, polyorder=3):
    if len(x) < window_length:
        return x
    return scipy.signal.savgol_filter(x, window_length, polyorder)

def calculate_lfo(frames, fps, variance):
    if frames <= 0: return np.array([])
    freq = 0.2 
    time_arr = np.arange(frames) / fps
    base_lfo = np.sin(2 * np.pi * freq * time_arr)
    # LFO amplitude inversely scaled to temporal envelope variance
    amp = 0.02 * np.exp(-np.clip(variance, 0, 100) * 500.0) 
    return base_lfo * amp

def analyze_audio(audio_path):
    print(f"Loading audio: {audio_path}")
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit(1)
        
    if len(y) == 0:
        print("Error: Audio file is empty.")
        sys.exit(1)
        
    fps = 10
    hop_length = int(sr // fps) # Explicit int cast to avoid librosa TypeError
    
    print("Separating frequency bands...")
    try:
        nyq = sr * 0.5
        cutoff = 400 / nyq
        if cutoff >= 1.0 or cutoff <= 0.0:
            y_bottom, y_top = y, y # Fallback for bizarre sample rates
        else:
            b_low, a_low = scipy.signal.butter(4, cutoff, btype='low')
            y_bottom = scipy.signal.filtfilt(b_low, a_low, y)
            
            b_high, a_high = scipy.signal.butter(4, cutoff, btype='high')
            y_top = scipy.signal.filtfilt(b_high, a_high, y)
    except Exception as e:
        print(f"Warning: Band splitting failed ({e}), using raw signal.")
        y_bottom, y_top = y, y
        
    print("Extracting RMS Energy & Onset Strength...")
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    print("Extracting Spectral Flatness (Roughness)...")
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]

    print("Extracting Chroma (Top & Bottom)...")
    try:
        chroma_top = librosa.feature.chroma_cqt(y=y_top, sr=sr, hop_length=hop_length)
        chroma_bottom = librosa.feature.chroma_cqt(y=y_bottom, sr=sr, hop_length=hop_length)
    except Exception as e:
        print(f"Error in chroma_cqt: {e}")
        sys.exit(1)
        
    v_top_raw = extract_valence(chroma_top, rms)
    v_bottom_raw = extract_valence(chroma_bottom, rms)
    
    print("Extracting HNR (Harmonic-to-Noise Ratio)...")
    try:
        D = librosa.stft(y, hop_length=hop_length)
        H, P = librosa.decompose.hpss(D, margin=1.2)
        e_harm = np.sum(np.abs(H)**2, axis=0)
        e_perc = np.sum(np.abs(P)**2, axis=0)
        hnr = e_harm / (e_perc + 1e-8)
    except Exception as e:
        print(f"Warning: HPSS failed ({e}), setting hnr to strictly 0.")
        hnr = np.zeros(len(rms))
        
    # Align lengths
    min_len = min(len(v_top_raw), len(v_bottom_raw), len(rms), len(onset_env), len(flatness), len(hnr))
    if min_len == 0:
        print("Error: Signal too short to produce meaningful trajectory frames.")
        sys.exit(1)
        
    v_top_raw = v_top_raw[:min_len]
    v_bottom_raw = v_bottom_raw[:min_len]
    rms = rms[:min_len]
    onset_env = onset_env[:min_len]
    flatness = flatness[:min_len]
    hnr = hnr[:min_len]

    # Smoothing and Normalization
    window_length = int(10 * fps)
    v_top = smooth(v_top_raw, window_length, 3)
    v_bottom = smooth(v_bottom_raw, window_length, 3)
    
    a_raw = np.clip(rms / 0.35, 0.0, 1.0)
    global_a_mean = float(np.mean(a_raw))
    a = smooth(a_raw, window_length, 3)
    
    # PASS-0 GLOBAL PROFILING
    global_energy_mean = global_a_mean
    global_energy_max = float(np.max(a))
    global_valence_mean = float(np.mean((v_top + v_bottom) / 2.0))
    
    if global_energy_mean < 0.2 or global_energy_max < 0.5:
        global_mood = "DEEP_MEDITATIVE"
    elif global_energy_max > 0.8 and global_energy_mean > 0.4:
        global_mood = "HIGH_KINETIC"
    else:
        global_mood = "VOLATILE_CINEMATIC"
        
    print(f"Global Trajectory Arc: {global_mood} (Energy Mean: {global_energy_mean:.2f})")
    
    # Global Bias dampening for meditative tracks
    if global_a_mean < 0.25:
        v_top = v_top * 0.6 + 0.1
        v_bottom = v_bottom * 0.6 + 0.1
        
    if global_mood == "DEEP_MEDITATIVE":
        # Darkness Bias: Force arousal/energy into the abyss
        a = a * 0.4
        v_top = np.clip(v_top - 0.2, 0.0, 1.0)
        v_bottom = np.clip(v_bottom - 0.2, 0.0, 1.0)
        
    # Baseline Roughness calculations
    onset_norm = smooth(onset_env / (np.max(onset_env) + 1e-6), 11, 3)
    
    r = smooth(flatness, 15, 3)
    r = np.clip(r * 2.0, 0.0, 1.0)
    
    hnr_norm = smooth(hnr / (np.max(np.nan_to_num(hnr)) + 1e-8), 15, 3)
    
    print("Calculating Temporal Context & Lux...")
    window_size_10s = int(10 * fps)
    window_size_15s = int(15 * fps)
    
    lux = np.ones(min_len)
    sat = np.ones(min_len)
    var_array = np.zeros(min_len)
    
    for i in range(min_len):
        start_10 = max(0, i - window_size_10s)
        window_10 = a[start_10:i+1]
        mean_10 = np.mean(window_10) if len(window_10) > 0 else 0
        var_array[i] = np.var(window_10) if len(window_10) > 0 else 0
        
        # 'Drop' detection
        if a[i] > mean_10 * 1.5 + 0.1:
            lux[i] = 1.0 + (a[i] - mean_10) * 2.0
            
        # Ambient/Silence detection
        start_15 = max(0, i - window_size_15s)
        window_15 = a[start_15:i+1]
        mean_15 = np.mean(window_15) if len(window_15) > 0 else 0
        
        if mean_15 < 0.05 and len(window_15) >= window_size_15s:
            lux[i] = max(0.2, lux[i-1] * 0.99) if i > 0 else 0.2
            sat[i] = max(0.1, sat[i-1] * 0.99) if i > 0 else 0.1
        else:
            sat_target = 0.5 + 0.5 * hnr_norm[i]
            sat[i] = sat[i-1] * 0.95 + sat_target * 0.05 if i > 0 else sat_target

    print("Generating Breathing LFO...")
    lfo = calculate_lfo(min_len, fps, var_array)
    
    # Base adjustments according to logic requests
    v_top_final = np.clip(v_top + lfo, 0.0, 1.0)
    v_bottom_final = np.clip(v_bottom + lfo, 0.0, 1.0)
    a_final = np.clip(a + lfo, 0.0, 1.0)
    
    # Fuzzy logic modification: high onset/syncopation pushes arousal spikes
    a_final = np.clip(a_final + (onset_norm * 0.2), 0.0, 1.0)
    
    # High HNR (pure vocals/piano) increases sat_mod
    sat_final = np.clip(sat + (hnr_norm * 0.1), 0.0, 1.0)
    
    # Glacier Macro-Smoothing (5-10 Second Windows)
    # Using window length of 10s (approx 10 * fps)
    window_length = int(5 * fps)
    if window_length % 2 == 0: window_length += 1
    
    if min_len > window_length:
        a_final = scipy.signal.savgol_filter(a_final, window_length, 3)
        v_top_final = scipy.signal.savgol_filter(v_top_final, window_length, 3)
        v_bottom_final = scipy.signal.savgol_filter(v_bottom_final, window_length, 3)
        # Re-clip after polynomial smoothing
        a_final = np.clip(a_final, 0.0, 1.0)
        v_top_final = np.clip(v_top_final, 0.0, 1.0)
        v_bottom_final = np.clip(v_bottom_final, 0.0, 1.0)
    
    def safe_float(val):
        """Ensure strict native Python float, fallback to 0.0 for NaNs."""
        try:
            f = float(np.nan_to_num(val))
            if math.isnan(f) or math.isinf(f):
                return 0.0
            return f
        except:
            return 0.0

    print("Formatting Output...")
    output_data = []
    for i in range(min_len):
        frame_data = {
            "t": round(i / fps, 3),
            "v_top": round(safe_float(v_top_final[i]), 3),
            "v_bottom": round(safe_float(v_bottom_final[i]), 3),
            "a": round(safe_float(a_final[i]), 3),
            "r": round(safe_float(r[i]), 3),
            "lux": round(safe_float(lux[i]), 3),
            "sat": round(safe_float(sat_final[i]), 3)
        }
        output_data.append(frame_data)
        
    return {
        "metadata": {
            "mood": global_mood,
            "energy_mean": round(safe_float(global_energy_mean), 3),
            "energy_max": round(safe_float(global_energy_max), 3),
            "valence_mean": round(safe_float(global_valence_mean), 3)
        },
        "frames": output_data
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AURA Cinematic Audio Analyzer")
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("output", help="Path to output JSON file", nargs="?", default="trajectory.json")
    
    args = parser.parse_args()
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        sys.exit(1)
    trajectory = analyze_audio(args.input)
    
    try:
        with open(args.output, 'w') as f:
            json.dump(trajectory, f, separators=(',', ':'))
        print(f"Exported {len(trajectory)} frames to {args.output}")
    except Exception as e:
        print(f"Error during JSON serialization: {e}")
        sys.exit(1)
