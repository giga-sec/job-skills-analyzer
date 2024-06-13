import streamlit as st
from pandas import DataFrame, read_csv
from plotly.express import line, bar
import boto3
from os import path
from nltk.data import find

# Define a function to check and download NLTK data
def check_and_download_nltk_data():
    datasets = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
    }

    for dataset_name, dataset_path in datasets.items():
        try:
            find(dataset_path)
            st.write(f"{dataset_name} is already downloaded.")
        except LookupError:
            st.write(f"{dataset_name} not found. Downloading...")
            nltk.download(dataset_name)

# Ensure necessary NLTK data is downloaded
check_and_download_nltk_data()


def tokenize_lemmatize(text):
    
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from string import punctuation
    from nltk.tokenize import word_tokenize
    # Tokenize the text
    stop_words = set(stopwords.words("english"))
    nltk = WordNetLemmatizer()
    tokens = word_tokenize(str(text).lower())

    # Remove punctuation and stop words then lemmatize
    # Lemmatization helps to reduce words to their base or dictionary form.
    tokens = [nltk.lemmatize(token) for token in tokens if token not in stop_words
                  and token not in punctuation]

    # Convert the list to a set to remove duplicates, then back to a list
    # tokens = list(set(tokens))

    # Join the tokens back into a string
    processed_text = " ".join(tokens)
    return processed_text  # <- str

# Function to tokenize text into bigrams
def generate_bigrams(text):
    from nltk.util import ngrams
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(str(text).lower())
    filtered_tokens = [word for word in tokens if word.isalnum()]
    bigrams = list(ngrams(filtered_tokens, 2))
    return bigrams

# Function to count occurrences of each skill in the job description
def count_skill_occurrences(job_description, skills):
    # from re import findall, escape, IGNORECASE
  #     Steps:
    # Initialize a dictionary to store counts for each skill.
    # For each skill, use regular expressions to find and count occurrences in the job description.
    # Return the dictionary of skill counts.
    skill_counts = {skill: 0 for skill in skills}
    job_description_tokens = set(job_description.split())

    # for skill in skills:
    #     occurrences = len(re.findall(r'\b' + re.escape(skill) + r'\b', str(job_description), re.IGNORECASE))
    #     skill_counts[skill] = occurrences
    # return skill_counts
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


# Function to count occurrences of bigrams in the text
def count_bigram_occurrences(text, bigram):
    return text.count(bigram)


def start_bigram_analysis():
  from csv import QUOTE_NONNUMERIC
  global JOBSKILLS_DF, SIGNAL
  if skills_list_txtarea != "":
    skills_list = [skill.strip() for skill in skills_list_txtarea.split(",")]
    skills_list = lemmatize_skills(skills_list)

    # START - Bigram Job Skills Scanning
    # Tokenize and Lemmatize First
    temp_df = ORIGINAL_DF
    temp_df["lemmatized_text"] = ORIGINAL_DF["description"].apply(tokenize_lemmatize)
    temp_df["lemmatized_text"].to_csv(f"csv\\lemmatized_{job_title}.csv", encoding='utf-8', quoting=QUOTE_NONNUMERIC, escapechar="\\", index=False) # to_xlsx

    # Initialize skill occurrences dictionary
    # skill_occurrences: dict[str, int] = {skill: 0 for skill in skills_list}
    skill_occurrences: list[tuple[str, int]] = [(skill, 0) for skill in skills_list]
    # ^-- Example: dict["Content Writer": 20]

    # Initialize bigram occurrences dictionary
    bigram_occurrences = {}

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
        
        # Count occurrences of each bigram
        # bigrams = generate_bigrams(job_description)
        # for bigram in bigrams:
        #     if bigram in bigram_occurrences:
        #         bigram_occurrences[bigram] += 1
        #     else:
        #         bigram_occurrences[bigram] = 1


    # Saving data
    df_jobTitles_count = DataFrame(skill_occurrences, columns=['jobSkills', 'count'], index=None)
    JOBSKILLS_DF = df_jobTitles_count.sort_values(by='count', ascending=False)
    SIGNAL = "Skills Analyzed Done"


####### ^-- FUNCTIONS FOR BIGRAM ANALYSIS ########### 


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


def start_ai_generate_skills():
  from openai import OpenAI
  client = OpenAI(
  #   organization='YOUR_ORG_ID',
  #   project='$PROJECT_ID',
    api_key = st.secrets["API_KEY"]
  )
  prompt = f"""
          Give me strictly 70 top skills needed for {job_title}
          No explanation. Don't use numbers to categorize skills
          Separate skills with commas.
          One to Three Words only. Acronyms are allowed.
          Strictly separate acronym and definition
          Correct: Content Management System, CMS
          Incorrect: Content Management System (CMS)
          ```Format Example
          CMS, Content Management System, Japanese, Communication
          ```
            """
  response = client.chat.completions.create(
    model="gpt-4-0125-preview",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.5,
    max_tokens=500
  )
  return response.choices[0].message.content


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
      location="Cebu City",
      distance=50, # miles
      # is_remote= 
      results_wanted=500,
      # hours_old = six_months, # (only linkedin is hour specific, others round up to days old)
      country_indeed='philippines',  # only needed for indeed / glassdoor
      linkedin_fetch_description=True
  )
  print(f"Found {len(jobs)} jobs")
  print(jobs.head())

  columns_to_remove = [
      "company_description",
      "logo_photo_url",
      "banner_photo_url",
      "ceo_name",
      "ceo_photo_url",
      "company_num_employees",
      "company_industry",
      "company_addresses",
      "company_url_direct",
      "emails",
      "currency",
      "interval",
      "min_amount",
      "max_amount",
      "job_type",
      "company_revenue",
      "site",
      "job_url_direct",
      "company_url",
  ]

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
    return response['Body'].read()

def upload_csv_to_s3(dataframe, bucket_name, filename):
    from io import StringIO
    csv_buffer = StringIO()
    dataframe.to_csv(csv_buffer, index=False)  # Save DataFrame to CSV format in the buffer
    csv_buffer.seek(0)

    s3_client.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=filename)



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



### START OF THE CODE
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
  # file_exists = path.exists(f"csv\\original_{job_title}.csv")
  # file_exists = download_csv_from_s3(BUCKET_NAME, f"original_{job_title}.csv")
  file_exists = file_exists_in_s3(BUCKET_NAME, f"original_{job_title}.csv")
    # print(f"File original_'{job_title}' already exists. Skipping...")
    # st.write("Scraped!")
  if (job_title != ""):
    is_disabled = False
    if file_exists == False:
      # subprocess.run(["python", "scrape.py", job_title])
      st.write("File doesn't exist in our database. Will be scraping job postings from indeed.com for a moment...")
      ORIGINAL_DF = start_scrape_jobs(job_title)
    elif file_exists == True:
      # ORIGINAL_DF = read_csv(f"csv\\original_{job_title}.csv")
      ORIGINAL_DF = download_csv_from_s3(BUCKET_NAME, f"original_{job_title}.csv")
    st.write(f"{ORIGINAL_DF['title'].count()} jobs were scraped from Indeed.com")

  #--> Start of Job Skill TextArea
  if st.session_state.get('generate'):
    st.session_state['name'] = start_ai_generate_skills()
  st.button('Generate Data', key='generate')
  skills_list_txtarea = st.text_area("Skills:", height=500, key='name', disabled=is_disabled,
                                        help="Input your skills here \nor click 'AI GENERATE SKILLS' to automatically generate skills for you")
  #--> End of Job Skill TextArea


  #--> START OF BIGRAM ANALYSIS  
  start_bigram_analysis()
  #--> END OF BIGRAM ANALYSIS

# ^-- End of Sidebar --^ #
##########################



###########################
# Main Menu Section Below #
st.markdown("""# Cebu City Job-Skills Real Time Analysis
            """)

tabs = st.tabs(["Table Chart", "Bar Chart", "Line Chart"])
if SIGNAL == "Skills Analyzed Done":
  # df2 = pd.read_csv(f"csv\\jobSkills_{job_title}.csv")
  # df = pd.read_csv(f"csv\\original_{job_title}.csv")
  LENGTH_JOBS = ORIGINAL_DF['title'].count()
  LENGTH_SKILLS = JOBSKILLS_DF['jobSkills'].count()

  # Display content based on selected tab
  if JOBSKILLS_DF.empty:
    print("Empty Values")
  else:
    with tabs[0]:
      left, right = st.columns(2)
      with left:
        tableChartExistingTrendSkills(JOBSKILLS_DF)
      with right:
        tableChartNotExistingTrendSkills(JOBSKILLS_DF)
      
      top_skill = JOBSKILLS_DF['jobSkills'].iloc[0]
      top_skill2 = JOBSKILLS_DF['jobSkills'].iloc[1]
      top_skill3 = JOBSKILLS_DF['jobSkills'].iloc[2]
      st.header("Narrow down your search here: ")
      st.write("Search the jobs that was mentioned as top skills ")
      narrow_search_job_desc = \
        st.text_input(f"""
          \nTry searching one of the top skill of {job_title}: "{top_skill}" or "{top_skill2}" or "{top_skill3}
          \nExample: If you search for {top_skill}, then the table below will only show job links that has {top_skill} as a skill needed for that job"
""", placeholder=f"Try typing: {top_skill}")
      st.write("\n\n\n")
      df_narrowed = ORIGINAL_DF
      if narrow_search_job_desc != "":
        df_narrowed = df_narrowed[df_narrowed['lemmatized_text'].str.contains(narrow_search_job_desc, case=False, na=False)]
        st.write(f"{df_narrowed['title'].count()} jobs found that contains '{narrow_search_job_desc}' as a skill for {job_title}")
      st.dataframe(df_narrowed, hide_index=True, width=1500, height=200)

    with tabs[1]: # Bar Chart
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
          st.plotly_chart(chartLeft, use_container_width=True)
        start += 10
        end += 10
        with right:
          chartRight = barChartAutomatic(JOBSKILLS_DF, start, end)
          st.plotly_chart(chartRight, use_container_width=True)
        inc_length_skills -= 10
      if (inc_length_skills >= 10):
         chartLeft = barChartAutomatic(JOBSKILLS_DF, start, end)
         st.plotly_chart(chartLeft, use_container_width=True)
    
    with tabs[2]:
      df2_top20 = JOBSKILLS_DF.iloc[0:20]
      inc_length_skills = LENGTH_SKILLS
      fig = line(df2_top20, x="jobSkills", y="count", 
                    title=f"Line Chart - Top 20 Skills for {job_title} in Cebu City", 
                    color="jobSkills", markers=True)
      # fig.update_traces(textposition="middle center", text=df2_top20["count"].astype(str))
      st.plotly_chart(fig, use_container_width=True)
