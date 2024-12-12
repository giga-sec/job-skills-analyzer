import unittest
import pandas as pd
from streamlit_app_offline import start_scrape_jobs, start_ai_generate_skills, start_main_function_analysis

class TestStreamlitAppOffline(unittest.TestCase):

    def setUp(self):
        self.job_skills = ['content writing', 'data analysis', 'data science', 'machine learning']
        self.df_job_skills = pd.DataFrame({'skill': self.job_skills})
        self.job_title = 'Data Analyst'

    def test_start_scrape_jobs(self):
        job_postings = start_scrape_jobs(self.job_title)
        self.assertIsInstance(job_postings, pd.DataFrame)
        self.assertGreater(len(job_postings), 0)

    def test_start_ai_generate_skills(self):
        skills = start_ai_generate_skills(self.job_title)
        self.assertIsInstance(skills, str)
        self.assertGreater(len(skills), 0)
        self.assertIn(',', skills)  # check if skills are comma-separated

    def test_start_main_function_analysis(self):
        ORIGINAL_DF = pd.DataFrame({
            'description': ['This is a job description with skills like content writing and data analysis.',
                            'Another job description with skills like data science and machine learning.']
        })
        JOBSKILLS_DF = self.df_job_skills
        start_main_function_analysis(ORIGINAL_DF)
        self.assertIsInstance(JOBSKILLS_DF, pd.DataFrame)
        self.assertGreater(len(JOBSKILLS_DF), 0)

if __name__ == '__main__':
    unittest.main()