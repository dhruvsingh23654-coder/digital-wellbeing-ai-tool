import unittest
from app import app

class TestMindSyncAI(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'MindSync', response.data)

    def test_predict_page_loads(self):
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Predict Wellbeing', response.data)

    def test_predict_post_valid(self):
        response = self.client.post('/predict', data={
            'screen_time': '4', 'social_media_usage': '2',
            'sleep_hours': '7', 'study_hours': '3',
            'exercise_time': '1', 'notifications': '50',
            'app_unlocks': '40', 'late_night_usage': '0.1',
            'mood_level': '7'
        })
        self.assertEqual(response.status_code, 200)

    def test_predict_post_exceeds_24hrs(self):
        response = self.client.post('/predict', data={
            'screen_time': '10', 'social_media_usage': '8',
            'sleep_hours': '9', 'study_hours': '6',
            'exercise_time': '5', 'notifications': '50',
            'app_unlocks': '40', 'late_night_usage': '0.1',
            'mood_level': '7'
        })
        self.assertIn(b'exceed 24', response.data)

if __name__ == '__main__':
    unittest.main()