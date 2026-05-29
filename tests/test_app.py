from app import create_app
import unittest

class TestMindSyncAI(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        self.app_context.pop()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome to MindSync AI', response.data)

    def test_results_page(self):
        response = self.client.get('/results')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Results of your analysis', response.data)

    def test_dashboard_page(self):
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Analytics Dashboard', response.data)

if __name__ == '__main__':
    unittest.main()