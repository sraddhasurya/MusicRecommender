import os
import re
import nltk
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import lyricsgenius as lyrics
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import torch
import torch.nn.functional as F




class MusicRecommender:
    def __init__(self):

        #loads the environment variables from .env file
        from dotenv import load_dotenv
        load_dotenv()

        #initalizes the spotify client with oauth, uses env variables, os allows us to get access to environment variables 
        self.sp= spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=os.getenv('SPOTIFY_CLIENT_ID'), client_secret=
                                                           os.getenv('SPOTIFY_CLIENT_SECRET)'), 
                                                           redirect_uri="https://example.com/callback",
                                                           scope="user-library-read"))
        
        #initalizes the genius client using the access token(positional arguement so no = sign)
        self.genius= lyrics.Genius(os.getenv('GENIUS_ACCESS_TOKEN'))

        #turns text into input for the model
        self.tokenizer= AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        #puts the input through the model and outputs a numerical vector
        self.model= AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        #downloads a model that turns text into sentences
        nltk.download('punkt')
        #downloads a list of common words that don't carry a lot of meaning
        nltk.download('stopwords')
        #converts the list of words into a set for faster look up 
        self.stop_words = set(nltk.corpus.stopwords.words('english'))

        #list of difference audio variblaes 
        self.feature_columns = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness','acousticness', 'instrumentalness', 
            'valence', 'tempo'
        ]

        #how much audio and lyrics are weighed when finding similarity between two songs
        self.audio_weight = 0.6
        self.lyrics_weight = 0.4

#Finds the track id of a song using the song name and the artist name
    def get_track_id(self, song_name, artist_name):
        #searches spotify for the track ID
        q=f"track:{song_name} artists:{artist_name}"
        result= self.sp.search(q=q, type='track', limit=1)
        tracks= result.get('tracks',{}).get('items',[])

        #if the track is found
        if tracks:
            return tracks[0]['id']
        #if the track is not found
        else:
            print(f"track not found: {song_name} by {artist_name}")
            return None
        
#Gets the audio features of a song using the songs id, input: track id of the song, output: list of values of the various audio 
#features if the song is found
    def get_song_audio_features(self, track_id):
        features =self.sp.audio_features([track_id])[0]
        if features is None:
            print('Song is not found')
            return None
        else:
            feature_removed={}
            for feature in self.feature_columns:
                feature_removed[feature]=features[feature]
            return feature_removed
        
#Can't use track ID because track id comes from spotify and we must use genius to get the lyrics 
#Returns the song lyrics using genius API
    def get_song_lyrics(self, song_name, artist_name):
        try:
            song=self.genius.search_song(title=song_name, artist=artist_name)
            if song and song.lyrics:
                return song.lyrics
            else:
                print(f"Lyrics not found for {song_name} by {artist_name}")
                return None
        except Exception as e:
            print(f"Error getting the lyrics for {song_name} by {artist_name}")
            return None

#takes the raw lyrics and cleans them, only keeping the words that are actually part of the song, returns a string
    def clean_lyrics(self, raw_lyrics):
        #if the lyrics is empty or not valid
        if not raw_lyrics:
            return ""

        #re.sub is a substitution function where it looks for a pattern and replaces them
        #removes sections headers, anything in [] are deleted
        cleaned=re.sub(r"\[.*?\]","",raw_lyrics)
        #removes any of those words and is case-insensitive
        cleaned=re.sub(r"(?:\n)?(Translations|Contributors|You might like|Embed|Read More)[^\n]*","",cleaned, flags=re.IGNORECASE)

        #splits line into individual lines
        lines=cleaned.split("\n")
        stripped_lines=[]
        #goes through each line and remove leading and trailing whitespace and if not empty adds to list
        for line in lines:
            stripped_line=line.strip()
            if stripped_line:
                stripped_lines.append(stripped_line)
        #adds the cleaned lines back together with a newline everytime
        cleaned="\n".join(stripped_lines)
        return cleaned


#Tokenizes and removes the stopwords in the lyrics to be able to put through BERT, returns a list
    def preprocess(self, cleaned_lyrics):
        if not cleaned_lyrics:
            return[]
        
        #splits up the lyrics into words
        tokens=wordpunct_tokenize(cleaned_lyrics)
        #reomves all the non words and stopwords
        nonstopwords=[]
        for word in tokens:
            lower_word=word.lower()
            if lower_word.isalpha():
                if lower_word not in self.stop_words:
                    nonstopwords.append(lower_word)
        
        return nonstopwords

#Look at model card to understand better
#Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        #First element of model_output contains all token embeddings
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#look at model card to understand better
#puts the cleaned and processed lyrics through a bert model
    def bert_processor(self, process_lyrics):
        if not process_lyrics:
            return None
        #joins the list of words into a string with a space seperating them 
        lyrics_text= " ".join(process_lyrics)
        model_inputs= self.tokenizer(lyrics_text, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output=self.model(**model_inputs)

        sentence_embeddings=mean_pooling(model_output, model_inputs['attention_mask'])
        sentence_embeddings=F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

#Normalizes the audio features of the song, scaled to [0,1], output is pytorch tensor
    def get_audio_feature(self, track_id):
        if not track_id: 
            return None
        features=self.get_song_audio_features(track_id)
        if features is None:
            return None
        
        feature_ranges={
            'danceability':(0.0, 1.0), 
            'energy':(0.0, 1.0), 
            'key':(-1,11), 
            'loudness':(-60.0,0.0), 
            'mode':(0,1), 
            'speechiness':(0.0,1.0),
            'acousticness': (0.0,1.0), 
            'instrumentalness': (0.0, 1.0), 
            'valence':(0.0,1.0), 
            'tempo':(50.0,250.0)
        }
        normalized_features=[]
        for feature in self.feature_columns:
            value= features[feature]
            min_val, max_val= feature_ranges[feature]
            if max_val!=min_val:
                scaled=(value-min_val)/(max_val-min_val)
            else:
                scaled = 0.0
            normalized_features.append(scaled)
        return torch.tensor(normalized_features, dtype=torch.float)
    
    






