import torch, glob
from maha_tts import load_models,infer_tts,config
from scipy.io.wavfile import write
from IPython.display import Audio,display

speaker =['./infer_ref_wavs/infer_ref_wavs/2272_152282_000019_000001/',
          '/infer_ref_wavs/infer_ref_wavs/2971_4275_000049_000000/',
          '/infer_ref_wavs/infer_ref_wavs/4807_26852_000062_000000/',
          '/infer_ref_wavs/infer_ref_wavs/6518_66470_000014_000002/']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diff_model,ts_model,vocoder,diffuser = load_models('Smolie-en',device)
print('Using:',device)

speaker_num = 0 # @param ["0", "1", "2", "3"] {type:"raw"}
# text = "all the world's a stage, and all the men and women merely players" # @param {type:"string"}
text = input("Enter the text to synthesize: ").strip()
print("_",text,"_")
ref_clips = glob.glob(speaker[speaker_num]+'*.wav')
audio,sr = infer_tts(text,ref_clips,diffuser,diff_model,ts_model,vocoder)

write('./generated-audio/test_english.wav',sr,audio)