import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
from pre_defined import normalise_and_label_encode

openai.api_key = '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'

# check if key exists in a list of dictionaries
def key_exists(k1, k2, list_of_dicts):
    for d in list_of_dicts:
        if k1 == d['Column 1'] and k2 == d['Column 2'] or k1 == d['Column 2'] and k2 == d['Column 1']:
            return True
    return False

# find columns with correlation greater than threshold value
def get_good_correlation(df, threshold=0.15):
    cor = df.corr()
    good_cor = []
    for i in cor.columns:
        for j in cor.columns:
            if i != j and (cor[i][j] > threshold or cor[i][j] < -threshold) and key_exists(i, j, good_cor) == False:
                good_cor.append(
                    {
                        'Column 1': i,
                        'Column 2': j,
                        'cor': cor[i][j]
                    }
                )
    return cor, good_cor


#openai api function for data modeling
def get_model(**kwargs):
    response = openai.Completion.create(
        engine="code-davinci-001",
        prompt=f"""Perform EDA on the "datasets/file.csv" file.""",
        temperature=0.2,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0.4,
        presence_penalty=0,
        n=1
    )
    return response['choices'][0]['text']

def main():
    st.set_page_config(page_title='Automated EDA and ML',
                   layout="wide")
    st.title("Automated EDA and ML")
    uploaded_file = st.file_uploader("Upload your file of structured data", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # drop index column that is not needed
        if 'Unnamed: 0' in df.columns:
            df = df.drop(['Unnamed: 0'], axis=1)
        
        df.to_csv('datasets/file.csv')
        columns = list(df.columns)


        st.subheader("Columns in the dataframe")
        # output all columns in dataframe
        st.write(columns, scrolling=True)

        df = normalise_and_label_encode(df)

        # get column to be predicted in our model
        st.subheader("What column do you want to predict?", columns)
        data_y = st.selectbox('', options=columns)
        
        # ask user for custom threshold values
        threshold = st.slider("threshold", 0.0, 1.0, 0.4)

        #get correlation and values for good correlation
        cor, good_cor = get_good_correlation(df, threshold)

        # plot correlation heatmap
        fig, ax = plt.subplots()
        sns.heatmap(cor, annot=True, ax=ax)

        # print the columns with an acceptable correlation
        st.markdown("## We reccomend using the following columns to create a model: ")
        st.table(good_cor)
        
        if data_y:
            already_plotted = []
            for i in good_cor:
                if i['Column 1'] != data_y and i['Column 1'] not in already_plotted:
                    fig, ax = plt.subplots()
                    sns.scatterplot(x = df[i['Column 1']], y = df[data_y], ax=ax)
                    already_plotted.append(i['Column 1'])
                    st.pyplot(fig)
                if i['Column 2'] != data_y and i['Column 2'] not in already_plotted:
                    fig, ax = plt.subplots()
                    sns.scatterplot(x = df[i['Column 2']], y = df[data_y], ax=ax)
                    already_plotted.append(i['Column 2'])
                    st.pyplot(fig)

        options = st.multiselect('Select columns to model', columns)
        st.write('You selected:', options)

        if options:
            st.subheader("Model from OpenAI inference")
            if st.button('Get Code'):
                with st.spinner('Wait for it...'):
                    code = get_model(data_x=options, data_y=data_y)
                st.success('Done!')
                st.code(code, language='python')
                # run code
                if st.button('Run Code'):
                    st.write(exec(code))
if __name__ == '__main__':
    main()