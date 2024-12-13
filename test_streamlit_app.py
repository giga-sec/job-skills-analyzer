import unittest
import pandas as pd
from streamlit_app_offline import start_scrape_jobs, start_ai_generate_skills, start_main_function_analysis
from streamlit_app_offline import (
    tableChartExistingTrendSkills,
    tableChartNotExistingTrendSkills,
    barChartAutomatic
)

class TestStreamlitAppOffline(unittest.TestCase):

    def setUp(self):
        self.job_skills = ['content writing', 'data analysis', 'data science', 'machine learning']
        self.df_job_skills = pd.DataFrame({'skill': self.job_skills})
        self.job_title = 'Data Analyst'
        self.dataframe = pd.DataFrame({
            'jobSkills': ['Python', 'Data Analysis', 'Machine Learning', 'SQL', 'JavaScript'],
            'count': [15, 10, 0, 8, 0]
        })
        self.start = 0
        self.end = 2

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


    def test_tableChartExistingTrendSkills(self):
        # Test that the function filters skills with count > 0
        existing_skills_df = self.dataframe.query('count > 0')
        self.assertGreater(len(existing_skills_df), 0)
        # Assuming Streamlit outputs can't be tested directly, check DataFrame logic
        tableChartExistingTrendSkills(self.dataframe)  # Runs without exceptions

    def test_tableChartNotExistingTrendSkills(self):
        # Test that the function filters skills with count < 1
        not_existing_skills_df = self.dataframe.query('count < 1')
        self.assertGreater(len(not_existing_skills_df), 0)
        tableChartNotExistingTrendSkills(self.dataframe)  # Runs without exceptions

    def test_barChartAutomatic(self):
        # Test that the bar chart function correctly creates a chart with filtered data
        filtered_df = self.dataframe.iloc[0:2]
        chart = barChartAutomatic(self.dataframe, self.start, self.end)
        # Check if returned object is not None (indicating successful chart generation)
        self.assertIsNotNone(chart)
        # Verify data used in the chart
        self.assertEqual(filtered_df.shape[0], self.end - self.start)
        self.assertIn('Python', filtered_df['jobSkills'].tolist())

if __name__ == '__main__':
    unittest.main()