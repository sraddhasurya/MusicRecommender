import unittest
from unittest.mock import patch, MagicMock
from recommender import MusicRecommender  # adjust based on your file name
import torch

class TestMusicRecommender(unittest.TestCase):

    @patch('recommender.SpotifyOAuth')
    @patch('recommender.spotipy.Spotify')
    @patch('recommender.lyrics.Genius')
    def setUp(self, mock_genius, mock_spotify, mock_auth):
        self.recommender = MusicRecommender()
        self.mock_sp = mock_spotify.return_value
        self.mock_genius = mock_genius.return_value

    def test_get_track_id_success(self):
        self.mock_sp.search.return_value = {
            'tracks': {
                'items': [{'id': 'test123'}]
            }
        }
        result = self.recommender.get_track_id('Test Song', 'Test Artist')
        self.assertEqual(result, 'test123')

    def test_get_track_id_not_found(self):
        self.mock_sp.search.return_value = {'tracks': {'items': []}}
        result = self.recommender.get_track_id('Bad Song', 'No Artist')
        self.assertIsNone(result)

    def test_get_song_audio_features_success(self):
        self.mock_sp.audio_features.return_value = [{
            'danceability': 0.5, 'energy': 0.7, 'key': 5, 'loudness': -5.0,
            'mode': 1, 'speechiness': 0.05, 'acousticness': 0.1,
            'instrumentalness': 0.0, 'liveness': 0.1, 'valence': 0.9,
            'tempo': 120.0
        }]
        result = self.recommender.get_song_audio_features('test123')
        self.assertEqual(result['danceability'], 0.5)
        self.assertIn('tempo', result)

    def test_get_song_audio_features_none(self):
        self.mock_sp.audio_features.return_value = [None]
        result = self.recommender.get_song_audio_features('fake123')
        self.assertIsNone(result)

    def test_get_song_lyrics_success(self):
        mock_song = MagicMock()
        mock_song.lyrics = "These are lyrics"
        self.mock_genius.search_song.return_value = mock_song
        lyrics = self.recommender.get_song_lyrics("Test", "Artist")
        self.assertEqual(lyrics, "These are lyrics")

    def test_get_song_lyrics_not_found(self):
        self.mock_genius.search_song.return_value = None
        lyrics = self.recommender.get_song_lyrics("None", "None")
        self.assertIsNone(lyrics)

    def test_clean_lyrics_removes_sections_and_blank_lines(self):
        raw = "[Chorus]\nHello\n\n[Verse]\nWorld\n\nRead More on Genius"
        cleaned = self.recommender.clean_lyrics(raw)
        self.assertNotIn("Chorus", cleaned)
        self.assertNotIn("Read More", cleaned)
        self.assertEqual(cleaned, "Hello\nWorld")

    def test_clean_lyrics_empty(self):
        self.assertEqual(self.recommender.clean_lyrics(""), "")

    def test_preprocess_removes_stopwords_and_nonalpha(self):
        text = "Hello the world! This is 2024."
        result = self.recommender.preprocess(text)
        self.assertIn("hello", result)
        self.assertIn("world", result)
        self.assertNotIn("the", result)
        self.assertNotIn("2024", result)

    def test_preprocess_empty(self):
        result = self.recommender.preprocess("")
        self.assertEqual(result, [])

    def test_get_normalized_audio_features(self, mock_audio_features, mock_track_id):
        # Mock responses
        mock_track_id.return_value = "mock_track_id"
        mock_audio_features.return_value = {
            'danceability': 0.8,
            'energy': 0.7,
            'key': 5,
            'loudness': -5.0,
            'mode': 1,
            'speechiness': 0.05,
            'acousticness': 0.1,
            'instrumentalness': 0.2,
            'liveness': 0.15,
            'valence': 0.6,
            'tempo': 120.0
        }

        output = self.recommender.get_normalized_audio_features("Mock Song", "Mock Artist")
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(len(output), len(self.recommender.feature_columns))
        self.assertTrue(torch.all(output >= 0.0) and torch.all(output <= 1.0))

    def test_bert_processor(self):
        # Simple mock lyrics input
        lyrics_list = ["love love love", "dancing all night", "heartbreak hotel"]

        # Preprocess each one
        processed = [' '.join(self.recommender.preprocess(text)) for text in lyrics_list]
        embeddings = self.recommender.bert_processor(processed)

        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.shape[0], len(processed))  # num rows == num sentences
        self.assertEqual(embeddings.shape[1], 768)  # sentence-transformers model output

    if __name__ == '__main__':
        unittest.main()
