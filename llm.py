import pandas as pd
from llama_index.llms.gemini import Gemini
import re

def converting_date(df):
    
    llm = Gemini(
        model="models/gemini-1.5-flash",
        api_key="AIzaSyDQqajAGpdTkAuM1DCzQsCcbcUL7Mz__h0",  # uses GOOGLE_API_KEY env var by default
    )
    columns = df.columns    
    Prompt = f"""
    Role:
    You are a machine learning engineer who has experience in developing time series algorithim for past 10 years in python.  
    You provide the most optimized python codes based on the Instructions
    
    Instructions:
    1. Create a new column named as Date with format yyyy-mm-dd
    2. Just give the python code and nothing else
    
    Columns : {columns}
    
    Output Format:
    Code:
    data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str))
    """
    resp = llm.complete(Prompt)
    resp = str(resp)
    cleaned_string = re.sub(r'```|python|\n', '', resp).strip()
    formatted_string = cleaned_string.replace('import pandas as pd', 'import pandas as pd\n')
    data = df
    exec(formatted_string)
    return data


def converting_prediction_columns(df):
    
    llm = Gemini(
        model="models/gemini-1.5-flash",
        api_key="AIzaSyDQqajAGpdTkAuM1DCzQsCcbcUL7Mz__h0",  # uses GOOGLE_API_KEY env var by default
    )
    df_head = df.head(5)  
    Prompt = f"""
    Role:
    You are a machine learning engineer who has experience in developing time series algorithim for past 10 years in python.  
    You provide the most optimized python codes based on the Instructions
    
    Instructions:
    1. Analyze the column that can be predicted from the dataframe given like in the give . 
    2. Rename that column to "value"
    3. Just give the python code and nothing else

    
    dataframe : {df_head}

    """
    resp = llm.complete(Prompt)
    print(resp)
    # resp = str(resp)
    # cleaned_string = re.sub(r'```|python|\n', '', resp).strip()
    # print(cleaned_string)
    # formatted_string = cleaned_string.replace('import pandas as pd', 'import pandas as pd\n')
    # data = df
    # print(formatted_string)
    # exec(formatted_string)