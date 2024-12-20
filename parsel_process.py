import parselmouth
from parselmouth.praat import call
import librosa
import numpy as np
import os
import speech_recognition as sr
import syllables
from scipy import signal
from pydub import AudioSegment
from pydub.silence import detect_silence

SAMPLE_RATE = 22050
DURATION = 6 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def analyse_pitch(filepath, sample_rate=21000):
    '''
    Pitch is the quality of sound governed by the rate of vibrations. Degree of highness and lowness of a tone.
    F0 is the lowest point in a periodic waveform. WARNING: this may not be applicable to current dataset 
    Input:
        row of dataset
    Output:
        mean of the fundamental frequency found  
    '''
    sound = parselmouth.Sound(filepath).convert_to_mono()
    F0 = sound.to_pitch().selected_array['frequency']
    F0[F0 == 0] = np.nan
    return np.nanmedian(F0)

def analyse_pitch_range(filepath, sample_rate=21000):
    '''
    Pitch is the quality of sound governed by the rate of vibrations. Degree of highness and lowness of a tone.
    F0 is the lowest point in a periodic waveform. WARNING: this may not be applicable to current dataset 
    Input:
        row of dataset
    Output:
        range of the fundamental frequency found  
    '''
    sound = parselmouth.Sound(filepath).convert_to_mono()
    F0 = sound.to_pitch().selected_array['frequency']
    F0[F0 == 0] = np.nan
    minval = np.nanmin(F0)
    maxval = np.nanmax(F0)
    return maxval - minval

def analyse_formants(f, filepath):
    '''
    "A formant is acoustic energy around a frequency"
    CURRENTLY: Measures formants ONLY at glottal pulses
    Input:
        row of dataset
        f: formant we want
    Output:
        mean of the given formant (e.g., f1, f2, f3, f4)
    '''
    sound = parselmouth.Sound(filepath).convert_to_mono()
    pitch = call(sound, "To Pitch", 0.0, 75, 500)  # check pitch to set formant settings
    meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")  # get mean pitch
    maxFormant = 5500 if meanF0 > 150 else 5000
    formants = sound.to_formant_burg(time_step=0.010, maximum_formant=maxFormant)
    f_list = [parselmouth.praat.call(formants, "Get value at time", f, t, 'Hertz', 'Linear') for t in formants.ts()]
    f_list = [f_val if str(f_val) != 'nan' else 0 for f_val in f_list]
    return np.mean(f_list)

def analyse_mfcc(filepath, outputpath, sample_rate=21000):
    '''
    Creates MFCC 
    Input:
        row of dataset
    Output:
        outputs 20 by # array containing the mfcc 
    '''
    x, sr = librosa.load(filepath, sr=sample_rate)
    x = librosa.to_mono(x)
    mfcc = librosa.feature.mfcc(y=x, sr=sr)
    filname, ext = os.path.splitext(os.path.basename(filepath))
    return np.mean(mfcc)

def get_energy(filepath, sample_rate=21000):
    '''
    Energy of a signal corresponds to the total magnitude of the signal. 
    For audio signals, that roughly corresponds to how loud the signal is. 
    Input:
        row of dataset
    Output:
        energy of the signal. 
    '''
    sound = parselmouth.Sound(filepath).convert_to_mono()
    energy = sound.get_energy()
    return energy

def analyse_intensity(filepath, sample_rate=21000):
    '''
    Intensity represents the power that the sound waves produce
    Input:
        row of dataset
    Output:
        Returns mean intensity or loudness of sound extracted from Praat. 
    '''
    average_intensity = parselmouth.Sound(filepath).convert_to_mono().to_intensity()
    return average_intensity.get_average() #the duration will weight the average down for longer clips

def get_max_intensity(filepath, sample_rate=21000):
    '''
    Intensity represents the power that the sound waves produce
    references: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C1/C1S3_Dynamics.html
    Input:
        row of dataset
    Output:
        Returns max intensity or power in dB. 
    '''
    y, s = librosa.load(filepath, mono=True, sr=sample_rate)
    win_len_sec = 0.2
    power_ref = 10**(-12)
    win_len = round(win_len_sec * s)
    win = np.ones(win_len) / win_len
    if len(win) == 0 or len(y) == 0:
        return "NA"
    power = 10 * np.log10(np.convolve(y**2, win, mode='same') / power_ref)
    return np.max(power)

def analyse_zero_crossing(filepath, sample_rate=21000):
    '''
    Zero crossing tells us where the voice and unvoice speech occurs. 
    "Large number of zero crossings tells us there is no dominant low frequency oscillation"
    https://towardsdatascience.com/how-i-understood-what-features-to-consider-while-training-audio-files-eedfb6e9002b#:~:text=Zero%2Dcrossing%20rate%20is%20a,feature%20to%20classify%20percussive%20sounds.
    Input:
        row of dataset
    Output:
        Returns the number of zero crossing points that occur 
    '''
    x, s = librosa.load(filepath)
    x = librosa.to_mono(x)
    zero_crossings = librosa.feature.zero_crossing_rate(x)
    return np.mean(zero_crossings)

def clean_audio(input_file, clean_file): 
    noise_file = input_file[0:-4] + '_noise.wav'
    os.system('sox %s %s trim 0 1.000' % (input_file, noise_file))
    os.system('sox %s -n noiseprof noise.prof' % (noise_file))
    os.system('sox %s %s noisered noise.prof 0.3' % (input_file, clean_file))
    os.remove(noise_file)
    os.remove('noise.prof')
    return clean_file

def cleaning(filepath):
    '''
    Cleaning function to remove noise using sox.  
    Input:
        row of dataset
    Output:
        returns a clean audio file with noise removed. 
    '''
    wav_audio = AudioSegment.from_file(filepath, format="m4a")
    wav_audio.export(filepath[0:-4] + ".wav", format="wav")
    input_file = filepath[0:-4] + ".wav"
    clean_file = input_file[0:-4] + '_cleaned.wav'
    return clean_audio(filepath, clean_file)

def get_number_sylls(filepath, sample_rate=21000):
    '''
    Rate of speech using number of syllables per second. 
    Input:
        row of dataset
    Output:
        syllables/second. 
    '''
    r = sr.Recognizer()
    with sr.AudioFile(filepath) as source:              
        audio = r.record(source)                        
    try:
        transcript = r.recognize_google(audio, key=None, language='en-IN')     
    except (sr.UnknownValueError, sr.RequestError):
        return 'NA'
    syll = syllables.estimate(transcript)
    y, s = librosa.load(filepath, sr=sample_rate)
    y = librosa.to_mono(y)
    duration = librosa.get_duration(y=y, sr=s)
    return syll / duration

def get_number_words(filepath, sample_rate=21000):
    '''
    Rate of speech using number of words per second. 
    Input:
        row of dataset
    Output:
        words/second. 
    '''
    r = sr.Recognizer()
    with sr.AudioFile(filepath) as source:              
        audio = r.record(source)                        
    try:
        transcript = r.recognize_google(audio, key=None, language='en-IN')     
    except (sr.UnknownValueError, sr.RequestError):
        return 'NA'
    y, s = librosa.load(filepath, sr=sample_rate)
    y = librosa.to_mono(y)
    duration = librosa.get_duration(y=y, sr=s)
    return len(transcript.split()) / duration

def spectral_slope(filepath, sample_rate=21000):
    '''
    A spectral slope function that uses the mel spectrogram as input. 
    #reference: https://www.audiocontentanalysis.org/code/audio-features/spectral-slope-2/
    Input:
        row of dataset
    Output:
        mean spectral slope
    '''
    y, s = librosa.load(filepath, sr=sample_rate)
    y = librosa.to_mono(y)
    melspec = librosa.feature.melspectrogram(y=y, sr=s)

def pauses(filepath, sample_rate=21000):
    '''
    Average pause length in seconds which is an indicant of rate of speech. This detects silences and as such
    is sensitive to background noise.
    Input:
        row of dataset
    Output:
        pause rate
    '''
    file = AudioSegment.from_wav(filepath)
    chunks = detect_silence(file,
        min_silence_len = 50,
        silence_thresh = -30
    )
    y, s = librosa.load(filepath, sr=sample_rate)
    y = librosa.to_mono(y)
    t =round(librosa.get_duration(y=y, sr=s)*1000, 0)
    if t == 0:
        return "NA"
    pause_length = 0
    try:
        pause_lengths = []
        for pause in chunks:
          if pause[0] == 0 or pause[1] == t:
            continue
          pause_lengths.append(pause[1] - pause[0])
          pause_length=sum(pause_lengths)/len(pause_lengths)*0.001
    except:
        chunks = detect_silence(file,
        min_silence_len = 20,
        silence_thresh = -30
        )
        pause_lengths = []
        for pause in chunks:
          pause_lengths.append(pause[1] - pause[0])
          pause_length=sum(pause_lengths)/len(pause_lengths)*0.001

    return pause_length

def analyse_harmonics(filepath, sample_rate=21000):
    '''
    Harmonics to noise which is the ratio of noise to harmonics in the audio signal.  
    Input:
        row of dataset
    Output:
        hnr
    '''
    sound = parselmouth.Sound(filepath).convert_to_mono()
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    return hnr

def mean_spectral_rolloff(filepath, sample_rate=21000):
    '''
    The spectral roll-off, which indicates liveliness of audio signal.  
    Input:
        row of dataset
    Output:
        mean spectral roll-off
    '''
    y, s = librosa.load(filepath, sr=sample_rate)
    y = librosa.to_mono(y)
    spec_rf = librosa.feature.spectral_rolloff(y=y, sr=s, roll_percent=0.50)[0] #from paper about education style
    return np.mean(spec_rf)

def get_envelope(filepath, sample_rate=21000):
    '''
    Returns spectral envelope.
    Input:
        filepath: Path to the audio file
    Output:
        spectral envelope
    '''
    y, s = librosa.load(filepath, sr=sample_rate)
    y = librosa.to_mono(y)
    envelope = np.abs(np.fft.fft(y))**2
    return envelope

def analyse_jitter(filepath, sample_rate=21000):
    '''
    Deviations in individual consecutive F0 period lengths
    Input:
        filepath: Path to the audio file
    Output:
        mean local jitter
    '''
    y, s = librosa.load(filepath, sr=sample_rate)
    sound = parselmouth.Sound(y).convert_to_mono()
    pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 400)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    return localJitter

def analyse_shimmer(filepath, sample_rate=21000):
    '''
    Difference of the peak amplitudes of consecutive F0 periods.
    Input:
        filepath: Path to the audio file
    Output:
        mean local shimmer
    '''
    y, s = librosa.load(filepath, sr=sample_rate)
    sound = parselmouth.Sound(y).convert_to_mono()
    pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 400)
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return localShimmer


   
