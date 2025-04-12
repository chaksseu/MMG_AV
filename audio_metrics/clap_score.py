
import torchaudio
import numpy as np
from scipy.spatial.distance import cosine
import re  


excluded_words = ['clip', 'test', 'sparse', 'vggsound','batch', 'proc', 'sample', 'audio', 'video']
pattern_words = re.compile(r'^(?:' + '|'.join(excluded_words) + r')\d*$', re.IGNORECASE)
pattern_numbers = re.compile(r'^\d+$')

def clean_sentence(filename):
    if not isinstance(filename, str):
        raise ValueError("filename must be a string")
    
    sentence = filename.replace('_', ' ').replace('.wav', '')
    words = sentence.split()
    filtered_words = [
        word for word in words 
        if not pattern_words.match(word) and not pattern_numbers.match(word)
    ]
    cleaned_sentence = ' '.join(filtered_words)
    return cleaned_sentence




def calculate_clap(model_clap, preds_audio, filename, freq):
    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=freq)
    preds_audio_clap = resampler(preds_audio)

    preds_audio_clap  = preds_audio_clap.squeeze().numpy()

    # Get audio embeddings from audio data
    audio_data = preds_audio_clap.reshape(1, -1) # Make it (1,T) or (N,T)
    audio_embed = model_clap.get_audio_embedding_from_data(x=audio_data, use_tensor=False)

    # Get text embeddings from texts
    sentence_clean = clean_sentence(filename)
        
    text_data = [sentence_clean]
    
    # print(text_data)

    text_embed = model_clap.get_text_embedding(text_data)

    E_cap = np.array(text_embed[0])
    E_aud = np.squeeze(np.array(audio_embed))

    E_aud = E_aud / np.linalg.norm(E_aud)
    E_cap = E_cap / np.linalg.norm(E_cap)

    # Compute the cosine similarity
    similarity = 1 - cosine(E_aud, E_cap)

    # Scale the similarity score and bound it between 0 and 100
    score = max(100 * similarity, 0)

    return score