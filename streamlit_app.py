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
      search_term=job,
      location="Mandaue City",
      distance=2, # miles
      # is_remote= 
      results_wanted=500,
      # hours_old = six_months, # (only linkedin is hour specific, others round up to days old)
      country_indeed='philippines',  # only needed for indeed / glassdoor
      linkedin_fetch_description=True
  )
  print(f"Found {len(jobs)} jobs")
  print(jobs.head())

  columns_to_remove = ["company_description", "logo_photo_url", "banner_photo_url", "ceo_name", "ceo_photo_url", "company_num_employees", "company_industry", "company_addresses", "company_url_direct", "emails", "currency", "interval", "min_amount", "max_amount", "job_type", "company_revenue", "site", "job_url_direct", "salary_source", "company_url", "job_level", "job_function", "listing_type", "company_logo"]

  jobs_filtered = jobs.drop(columns_to_remove, axis=1, errors='ignore')
  upload_csv_to_s3(jobs_filtered, BUCKET_NAME, filename)
  return jobs_filtered



def file_exists_in_s3(bucket_name, filename):
    from botocore.exceptions import ClientError
    try:
        s3_client.head_object(Bucket=bucket_name, Key=filename)
        print("file_exists_in_s3: True")
        return True
    except ClientError as e:
        # If a 404 error is raised, the file does not exist
        if e.response['Error']['Code'] == '404':
            print("file_exists_in_s3: False")
            return False
        # For other errors, you might want to raise an exception
        else:
            raise e

def download_csv_from_s3(bucket_name, filename):
    # response = s3_client.get_object(Bucket='csvfilesforjobs', Key=filename)
    # return response['Body'].read()    
    response = s3_client.get_object(Bucket=bucket_name, Key=filename)
    csv_content = response['Body'].read().decode('utf-8') 
    return pd.read_csv(io.StringIO(csv_content))


def upload_csv_to_s3(dataframe, bucket_name, filename):
    from io import StringIO
    csv_buffer = StringIO()
    dataframe.to_csv(csv_buffer, index=False)  # Save DataFrame to CSV format in the buffer
    csv_buffer.seek(0)

    s3_client.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=f"{filename}.csv")



# Start of BOTO3
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"] 
BUCKET_NAME = st.secrets["BUCKET_NAME"]

# Initialize S3 client

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)
# End of BOTO3



##############################
##############################
##############################
### START OF THE MAIN CODE ###
st.set_page_config(layout="wide")
SIGNAL = "Skills not Analyzed"
LENGTH_SKILLS = 0
ORIGINAL_DF = DataFrame()
JOBSKILLS_DF = DataFrame() 
response = ""

print("\n\nRELOAD!!  RELOAD!!")



#############################
# >-- START OF SIDE BAR <-- #
with st.sidebar:
  st.title("Sidebar Menu")
  st.write("This is some content in the sidebar")

  # Initial state (skills hidden)
  show_skills = False

  # Get user input for job title
  job_title = st.text_input("Enter your job title:")
  
  is_disabled = True  # Initial state
  # Button to enable/disable text area of Skills
  
  
  file_exists = file_exists_in_s3(BUCKET_NAME, f"original_{job_title}.csv")
    # print(f"File original_'{job_title}' already exists. Skipping...")
    # st.write("Scraped!")
  if (job_title != ""):
    is_disabled = False
    if file_exists == False:
      # subprocess.run(["python", "scrape.py", job_title])
      st.write("File doesn't exist in our database. Will be scraping job postings from indeed.com for a moment...")
      ORIGINAL_DF = start_scrape_jobs(job_title)
      if ORIGINAL_DF.empty:
        st.write("No jobs found")
        return
    elif file_exists == True:
      # ORIGINAL_DF = read_csv(f"csv\\original_{job_title}.csv")
      ORIGINAL_DF = download_csv_from_s3(BUCKET_NAME, f"original_{job_title}.csv")
    st.write(f"{ORIGINAL_DF['title'].count()} jobs were scraped from Indeed.com")

  #--> Start of Job Skill TextArea
  if (st.session_state.get('enable_ai_generate_skills')) and (job_title != ""):
    st.session_state['name'] = start_ai_generate_skills(job_title)
  st.button('Auto Skills Generate', key='enable_ai_generate_skills', help="AI Powered")
  st.button('Generate Data/Chart', key='enable_generate_data', disabled=is_disabled,)
  skills_list_txtarea = st.text_area("Skills:", height=500, key='name', disabled=is_disabled,
                                        help="Input your skills here \nor click 'AI GENERATE SKILLS' to automatically generate skills for you")
  #--> End of Job Skill TextArea/


  #--> START OF BIGRAM ANALYSIS
  # This should only be enabled if the "st.button" Generate Data is clicked
  narrow_search_exists = st.session_state.get('narrow_search_input')
  if st.session_state.get('enable_generate_data') or narrow_search_exists:
    start_main_function_analysis(ORIGINAL_DF)
  #--> END OF BIGRAM ANALYSIS
  
  
# ^-- End of Sidebar --^ #
##########################
##########################



###########################
# Main Menu Section Below #
st.markdown("""# Mandaue City Job-Skills Real Time Analysis
            """)

tabs = st.tabs(["Table Chart", "Bar Chart", "Line Chart"])
if SIGNAL == "Skills Analyzed Done":
  # df2 = pd.read_csv(f"csv\\jobSkills_{job_title}.csv")
  # df = pd.read_csv(f"csv\\original_{job_title}.csv")
  LENGTH_JOBS = ORIGINAL_DF['title'].count()
  LENGTH_SKILLS = JOBSKILLS_DF['jobSkills'].count()

  # Display content based on selected tab
  # THE CODES OF CHARTS ARE HERE
  if JOBSKILLS_DF.empty:
    print("Empty Values")
  else:
    # Table Chart
    with tabs[0]:
      left, right = st.columns(2)
      with left:
        st.markdown("Skills in-demand")
        tableChartExistingTrendSkills(JOBSKILLS_DF)
      with right:
        st.markdown("Zero demand of skills")
        tableChartNotExistingTrendSkills(JOBSKILLS_DF)
      
      top_skill = JOBSKILLS_DF['jobSkills'].iloc[0]
      top_skill2 = JOBSKILLS_DF['jobSkills'].iloc[1]
      top_skill3 = JOBSKILLS_DF['jobSkills'].iloc[2]


      st.header("Narrow down your search here: ")
      st.write("Search the jobs that was mentioned as top skills ")
      narrow_search_job_desc = \
        st.text_input(f"""
          \nTry searching one of the top skill of {job_title}: "{top_skill}" or "{top_skill2}" or "{top_skill3}
          \nExample: 
          \nIf you search for {top_skill}, 
          \nthen the table below will only show job links that has {top_skill} as a skill needed for that job"
""", placeholder=f"Try typing {top_skill}", key="narrow_search_input")
      st.write("\n\n\n")
      df_narrowed = ORIGINAL_DF
      # Below is filtering of job search based on skill inputted by user 
      if (narrow_search_job_desc != ""):
          # Initialize counts dictionary
          skill_counts = {narrow_search_job_desc.lower(): 0}
          matched_jobs = []  # List to keep track of matched jobs

          # Iterate through the lemmatized_text column to count skill occurrences
          for job_description in df_narrowed['lemmatized_text']:
              counts = count_skill_occurrences(job_description, narrow_search_job_desc.lower())
              skill_counts[narrow_search_job_desc.lower()] += counts[narrow_search_job_desc.lower()]

              # Check if the count for the skill is greater than 0
              matched_jobs.append(counts[narrow_search_job_desc.lower()] > 0)
          
          df_narrowed = df_narrowed[matched_jobs]

          st.write(f"{skill_counts[narrow_search_job_desc.lower()]} jobs found that contain '{narrow_search_job_desc}' as a skill for {job_title}")
      st.dataframe(df_narrowed, hide_index=True, width=1500, height=300)

    # Bar Chart
    with tabs[1]: 
      chartLeft, chartRight, selectedpoints1, selectedpoints2 = "", "", "", ""
      left, right = st.columns(2)
      start, end = 0, 10
      inc_length_skills = LENGTH_SKILLS
      
      lastPosition_jobSkillsCount_value = JOBSKILLS_DF['count'].iloc[end] 
      # ^-- If the condition JOBSKILLS_DF['count'].iloc[end] != 0 evaluates to True, it means there's at least one skill in the DataFrame with a count greater than zero, indicating there are skills present in the data.
      # ^-- If the condition evaluates to False, it suggests that all the values in the 'count' column are zero, possibly implying no skills or empty data in that column.
      while (inc_length_skills > 10) and (lastPosition_jobSkillsCount_value != 0):
        with left:
          chartLeft = barChartAutomatic(JOBSKILLS_DF, start, end)
          st.plotly_chart(chartLeft, use_container_width=True, key=f"chartLeft_{start}")
        start += 10
        end += 10
        with right:
          chartRight = barChartAutomatic(JOBSKILLS_DF, start, end)
          st.plotly_chart(chartRight, use_container_width=True, key=f"chartRight_{start}")
        inc_length_skills -= 10
      if (inc_length_skills >= 10):
         chartLeft = barChartAutomatic(JOBSKILLS_DF, start, end)
         st.plotly_chart(chartLeft, use_container_width=True, key=f"chartLeft_final")
    
    # Line Chart
    with tabs[2]:
      df2_top20 = JOBSKILLS_DF.iloc[0:20]
      inc_length_skills = LENGTH_SKILLS
      fig = line(df2_top20, x="jobSkills", y="count", 
                    title=f"Line Chart - Top 20 Skills for {job_title} in Mandaue City", 
                    color="jobSkills", markers=True)
      # fig.update_traces(textposition="middle center", text=df2_top20["count"].astype(str))
      st.plotly_chart(fig, use_container_width=True)


# END OF Main Menu Section #
############################
############################