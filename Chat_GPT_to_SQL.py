import streamlit as st
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain.agents import create_sql_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
import pandas as pd
import pypyodbc as podbc
import os
from openai import OpenAI
from PIL import Image
import psycopg2 as pg
import mysql.connector
from openai import AzureOpenAI
import json
import SchemaAnalyzer as sa
import uuid
from typing import List
from urllib.parse import quote
from langchain_openai  import AzureChatOpenAI

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.messages import (
    convert_to_openai_messages,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.chains import create_sql_query_chain
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.sql.vector_sql import VectorSQLOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
endpoint = "https://kosar-ma5rg5wy-swedencentral.cognitiveservices.azure.com/"
model_name = "gpt-35-turbo-instruct"
deployment = "gpt-35-turbo-instruc"

subscription_key = "F1P2mXSV6mwmIGWj2TzJWnjT084w8wH74aRy1dJLznZMIfI2H4krJQQJ99BEACfhMk5XJ3w3AAAAACOGyb2X"
os.environ['OPENAI_API_KEY'] = subscription_key
api_version = "2024-12-01-preview"

mydb = mysql.connector.connect(
  host="127.0.0.1",
  user="root",
  password="J@ck2468",
  database="tax"
)

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)
# Set Streamlit layout to wide and customize page background color
st.set_page_config(layout="wide")

# Set CSS for custom styling (background color, card borders, etc.)
st.markdown("""
    <style>
    .reportview-container {
        background-color: #f0f0f0;
    }
    .sidebar .sidebar-content {
        padding-top: 0px;
    }
    .block-container {
        padding-top: 0rem;
    }
    .custom-card {
        background-color: white;
        padding: 5px;  /* Reduced padding for smaller height */
        border-radius: 10px;
        border: 2px solid #d3d3d3;  /* Slightly reduced border size */
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);  /* Slightly reduced shadow */
        text-align: center;
        margin-bottom: 10px;  /* Reduced space between cards */
    }
    .custom-card h3 {
        margin-bottom: 2px;  /* Reduced margin below the title */
    }
    .custom-card p {
        color: #003366;  /* Dark Blue Color */
        font-size: 20px;  /* Slightly smaller font size for compactness */
        font-weight: bold;
    }
    .card-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
        gap: 10px;  /* Reduced gap between cards */
    }
    .centered-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .stDataFrame div[data-testid="stHorizontalBlock"] {
        width: auto !important;
        min-width: 150px !important;  /* Adjusting column width for better visibility */
    }
    </style>
    """, unsafe_allow_html=True)

# Display banner image at the top of the app with reduced height
#image = Image.open('AI Pic2.png')
#st.image(image, use_column_width=True, caption="Tax Banner", output_format="PNG")

# Centered main title
st.markdown('<h1 class="centered-title">Tax Dashboard</h1>', unsafe_allow_html=True)

# Sidebar with an image
image2 = Image.open('AI Pic3.png')
st.sidebar.image(image2, use_column_width=True)
st.sidebar.header("Filter Options")

# Connect to MYSQL Server
def get_sql_data(query):
    mydb = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="J@ck2468",
    database="data"
    )
    mycursor = mydb.cursor()
    mycursor.execute(query)
    records = mycursor.fetchall()
    return records

# Fetch distinct options for filters
def get_filter_options(column_name):
    query = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS where table_name='tbl_income'"
    fetch_results=get_sql_data(query)
    results = []
    for row in fetch_results:
       results.append(row[0])
    return results


# Fetch min and max dates for order_date range
def get_date_range():
    query = "SELECT MIN(date_payment) as min_date, MAX(date_payment) as max_date FROM tax.tbl_income"
    result = get_sql_data(query)
    for y in result:
      mind=y[0]
      maxd=y[1] 
    return mind,maxd

# Fetch min and max dates for order_date range
def get_model_constant(constant_id):
    query="SELECT distinct constant_value from tax.tbl_model_constants where constant_id ='"+constant_id
    query=query+"' and activeind='1'"
    print (query)
    cv=''
    result = get_sql_data(query)
    for y in result:
      cv=y[0]
    return cv

 
# Define the function to query OpenAI
def query_data(data_json, question, model_name):
    prompt = f"Based on the following relational data, answer the question:\n{data_json}\nQuestion: {question}"
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "you are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        model=model_name,
    )
    return chat_completion.choices[0].message.content

# Order Date Range Filter
min_date, max_date = get_date_range()
order_date_range = st.sidebar.date_input("Order Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Segment Filter (Multiselect Drop-down)
segment_options = get_filter_options("symbol")
segment = st.sidebar.multiselect("symbol", segment_options, default=segment_options)

# City Filter (Single Select Drop-down)
city_options = get_filter_options("city")
city = st.sidebar.selectbox("City", ["All"] + city_options)

# Category Filter (Multiselect Drop-down)
category_options = get_filter_options("category")
category = st.sidebar.multiselect("Category", category_options, default=category_options)

# Product Name Filter (Single Select Drop-down)
product_name_options = get_filter_options("product_name")
product_name = st.sidebar.selectbox("Product Name", ["All"] + product_name_options)

# Build SQL Query with Filters
query = "SELECT * FROM tax.tbl_income limit 200"

# Load and display filtered data
fetch_results=get_sql_data(query)
results = []
for row in fetch_results:
   results.append(row)
data = results

# Add slider for row limit
row_limit = st.sidebar.slider('Limit number of rows:', min_value=0, max_value=1000, value=200)

# Apply row limit to the data
data_limited = data

# Calculate Metrics
num_rows = len(data_limited)
num_tokens = num_rows * 30
#num_products = data_limited['symbol'].nunique()
num_products=0
#sum_sales = data_limited['amount'].sum()
sum_sales=0

# Display cards with metrics
st.markdown('<div class="card-container">', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f'<div class="custom-card"><h3>Number of Rows</h3><p>{num_rows:,}</p></div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="custom-card"><h3>Number of Tokens</h3><p>{num_tokens:,}</p></div>', unsafe_allow_html=True)

with col3:
    st.markdown(f'<div class="custom-card"><h3>Number of Products</h3><p>{num_products:,}</p></div>', unsafe_allow_html=True)

with col4:
    st.markdown(f'<div class="custom-card"><h3>Sum of Sales</h3><p>${sum_sales:,.2f}</p></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Split page layout for Question section (50%) and SQL table (50%)
#col_left, col_right = st.columns(2)

# below is all langchin stuff

#schm="tax"
#analyzer = sa.SchemaAnalyzer("127.0.0.1","root","J@ck2468",schm,3306
#)

#analyzer.analyze_schema()
#schema = analyzer.get_schema()

#table_name_list = list(schema.keys())

class Table(BaseModel):
    """List tables name in SQL database."""
    name: List[str] = Field(description="List of tables name in SQL database.")

from typing import List

#def get_tables(table):
#    print('get_tables', type(table), table)
#    tables_results = []
#    response_text = ""
#    for table_name in table.name:
#        tables_results.append(table_name)
#        response_text += f"Table: {table_name} "
#        response_text += f" Columns:{', '.join(schema[table_name]['columns'])}"

#    return response_text

@tool
def extract_list_tables_relavance(query: str):
    """ Return the names of ALL the SQL tables that MIGHT be relevant to the user question. """
    print("call tool:extract_list_tables_relavance", query)
    system = f"""
        Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
        The tables are:
        {table_name_list}
        Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed.
        Output:
        "table_name1", "table_name2"
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{input}"),
        ]
    )

    prompt = prompt | model.with_structured_output(Table)

    prompt_value = prompt.invoke({"input": query})
    response_text = get_tables(prompt_value)
    print("prompt_value", response_text)
    return response_text

endpoint = "https://kosar-ma5rg5wy-swedencentral.openai.azure.com"
model_name = "gpt-4.1"
deployment = "gpt-4.1"

subscription_key = "F1P2mXSV6mwmIGWj2TzJWnjT084w8wH74aRy1dJLznZMIfI2H4krJQQJ99BEACfhMk5XJ3w3AAAAACOGyb2X"
api_version = "2024-12-01-preview"


memory = MemorySaver()
model = AzureChatOpenAI(
    azure_endpoint=endpoint,
    openai_api_version="2023-03-15-preview",
    deployment_name=deployment,
    openai_api_key=subscription_key,
    openai_api_type="azure",
)

schema_name='tax'
dbstring="mysql://root:%s@127.0.0.1/tax" % quote("J@ck2468")
#engine = create_engine(dbstring)

#db = SQLDatabase(engine,schema=schema_name)
db = SQLDatabase.from_uri(dbstring,schema="tax")

db_chain = SQLDatabaseChain.from_llm(model, db, verbose=True,use_query_checker=True)
#db_chain =create_sql_query_chain(model, db)


def state_modifier(state) -> list[BaseMessage]:
    """Given the agent state, return a list of messages for the chat model."""
    # We're using the message processor defined above.
    return trim_messages(
        state["messages"],
        token_counter=len,  # <-- len will simply count the number of messages rather than tokens
        max_tokens=5,  # <-- allow up to 5 messages.
        strategy="last",
        # Most chat models expect that chat history starts with either:
        # (1) a HumanMessage or
        # (2) a SystemMessage followed by a HumanMessage
        # start_on="human" makes sure we produce a valid chat history
        start_on="human",
        # Usually, we want to keep the SystemMessage
        # if it's present in the original history.
        # The SystemMessage has special instructions for the model.
        include_system=True,
        allow_partial=False,
    )


app = create_react_agent(
    model,
    tools=[extract_list_tables_relavance],
    checkpointer=memory,
    state_modifier=state_modifier
    
)

# The thread id is a unique key that identifies
# this particular conversation.
# We'll just generate a random uuid here.
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}


# Question section on the left
#with col_left:
st.subheader("Ask a Question About the Data")
user_question = st.text_input("Enter your question:")

# Button to ask question
if st.button("Ask ChatGPT"):
        if user_question:
            constant_value=get_model_constant('TABLE_HELP')
            print ("constant value:",constant_value)
            user_question=user_question  + ' ' + constant_value
            toolkit = SQLDatabaseToolkit(db=db, llm=model)

            agent_executor = create_sql_agent(
            llm=model,
            toolkit=toolkit,
            verbose=True
            )
            response=agent_executor.run(user_question)
            print(response) 
            st.subheader("ChatGPT Question")
            st.write(user_question)
            st.subheader("ChatGPT Answer")
            #st.write(chat_completion.choices[0].message.content) 
            st.write(response)
            
        else:
            st.warning("Please enter a question.")

# SQL table section on the right with increased column width for "profit"
#with col_right:
 #   st.subheader("Filtered SQL Data")
 #   st.dataframe(data_limited.style.set_properties(**{'symbol': '150px'}))