import streamlit as st
from pandas import DataFrame
import pandas as pd
from plotly.express import line, bar
import boto3
import io

import nltk
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('punkt_tab')

def tokenize_lemmatize(text):
    from re import sub
    from string import punctuation
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    # 1. Tokenize the text into individual words
    text = sub(r'[-/]', ' ', text)  # Replace special characters with spaces
    stop_words = set(stopwords.words("english"))
    nltk = WordNetLemmatizer()
    tokens = word_tokenize(str(text).lower())

    # 2. Remove punctuation and stop words
    # 3. Perform Lemmatization on each word, this reduces words to their base form.
    tokens = [nltk.lemmatize(token) for token in tokens if token not in stop_words
                  and token not in punctuation]

    # 4. Join the tokens back into a string
    processed_text = " ".join(tokens)
    return processed_text  # <- str

# Function to count occurrences of each skill in the job description
def count_skill_occurrences(job_description, skills):
    if isinstance(skills, str):
        skills = [skills]  # Convert single skill string to a list

    skill_counts = {skill: 0 for skill in skills}
    job_description_tokens = set(job_description.split())

    for skill in skills:
        skill_tokens = set(skill.lower().split())  # Tokenize and convert skill to a set
        if skill_tokens.issubset(job_description_tokens):
            skill_counts[skill] = 1  # Count as 1 if skill is mentioned

    return skill_counts


def lemmatize_skills(skills_list):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_skills = [lemmatizer.lemmatize(skill.lower()) for skill in skills_list]
    return lemmatized_skills

def start_main_function_analysis(ORIGINAL_DF):
  from csv import QUOTE_NONNUMERIC
  global JOBSKILLS_DF, SIGNAL
  if (skills_list_txtarea != "") and (job_title != ""):
    skills_list = [skill.strip() for skill in skills_list_txtarea.split(",")]
    skills_list = lemmatize_skills(skills_list)

    # START - Bigram Job Skills Scanning
    # Tokenize and Lemmatize First
    temp_df = ORIGINAL_DF
    temp_df["lemmatized_text"] = ORIGINAL_DF["description"].apply(tokenize_lemmatize)
    # temp_df["lemmatized_text"].to_csv(f"csv\\lemmatized_{job_title}.csv", encoding='utf-8', quoting=QUOTE_NONNUMERIC, escapechar="\\", index=False) # to_xlsx

    # Initialize skill occurrences dictionary
    skill_occurrences: list[tuple[str, int]] = [(skill, 0) for skill in skills_list]
    # ^-- Result Example: dict["Content Writer": 20]

    # Iterate over each job description
    for index, row in temp_df.iterrows():
        job_description = row['lemmatized_text']
        
        # Count occurrences of each skill
        skill_counts = count_skill_occurrences(job_description, skills_list)
        for skill, count in skill_counts.items():
            for i, (s, c) in enumerate(skill_occurrences):
                if s == skill:
                    skill_occurrences[i] = (skill, c + count)
                    break
            else:
                skill_occurrences.append((skill, count))

    # Saving data
    df_jobTitles_count = DataFrame(skill_occurrences, columns=['jobSkills', 'count'], index=None)
    JOBSKILLS_DF = df_jobTitles_count.sort_values(by='count', ascending=False)
    SIGNAL = "Skills Analyzed Done"

# END OF FUNCTIONS FOR NLP ANALYSIS #
########################################
########################################


#################################
# START OF FUNCTIONS FOR CHARTS #

def tableChartExistingTrendSkills(dataframe):
  existing_skillTrend = dataframe.query('count > 0')
  st.dataframe(existing_skillTrend, hide_index=True, width=300)

def tableChartNotExistingTrendSkills(dataframe):
  existing_skillTrend = dataframe.query('count < 1')
  st.dataframe(existing_skillTrend, hide_index=True, width=300)

def barChartAutomatic(dataframe, index_start, index_end):
  df2_top10 = dataframe.iloc[start:end]
  job_skills = df2_top10['jobSkills'].tolist()[::-1]  #[::-1] is used to indirectly display charts top to bottom
  counts = df2_top10['count'].tolist()[::-1]  #[::-1] is used to indirectly display charts top to bottom
  ## Create a horizontal bar chart using Plotly
  # chart = Figure()
  # chart.add_trace(Bar(x=counts, y=job_skills, orientation='h'))
  chart = bar(
    df2_top10,
    x=counts,
    y=job_skills,
    title=f"<b>Top {index_start+1}-{index_end} Job Trend Skills</b>",
    template="plotly_white",
    text=counts,
    color_discrete_sequence=["#0083B8"],
  )
  ## Display the chart using Streamlit
  return chart
  


# END OF FUNCTIONS FOR CHARTS #
###############################
###############################






def start_ai_generate_skills(job_title):
  from openai import OpenAI
  client = OpenAI(api_key = st.secrets["API_KEY"])
  completion = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
          {"role": "system", "content": "You are a Job Skills Generator."},
          {
              "role": "user",
              "content": f"""
                Give me strictly 100 skills needed for {job_title}
                No explanation.
                Separate skills with commas.
                One to Three Words only. Acronyms are allowed 
                Strictly separate acronym and definition
                Correct: Content Management System, CMS
                Incorrect: Content Management System (CMS)"""
          }
      ]
  )
  print(completion.choices[0].message)
  return completion.choices[0].message.content
  # return "Java, Python, C++, JavaScript, TypeScript, SQL, NoSQL, Git, GitHub, GitLab, Agile, Scrum, Kanban, AWS, Azure, GCP, Docker, Kubernetes, CI/CD, Jenkins, REST API, GraphQL, HTML, CSS, React, Angular, Vue.js, Node.js, Express.js, Spring Boot, Hibernate, ORM, TDD, Unit Testing, Integration Testing, Data Structures, Algorithms, Design Patterns, OOP, Functional Programming, Microservices, Serverless, Linux, Shell Scripting, Bash, PowerShell, Performance Optimization, Refactoring, Code Review, Debugging, Troubleshooting, Scalability, Security, Authentication, Authorization, Encryption, HTTPS, SSL, TLS, Database Design, Caching, Load Balancing, Networking, TCP/IP, HTTP, WebSockets, Responsive Design, UX, UI, Accessibility, Internationalization, Localization"


def start_scrape_jobs(job):
  # from csv import QUOTE_NONNUMERIC
  from jobspy import scrape_jobs

  filename = f"original_{job}"
  print(job)
  # Check if File exists
  # if os.path.exists(f"csv\\{filename}.csv"):
  #     print(f"File '{filename}' already exists. Exiting scrape.py...")
  #     sys.exit()

  # Scrape Jobs
  six_months = 183
  jobs: DataFrame = scrape_jobs(
      site_name=["indeed"],
