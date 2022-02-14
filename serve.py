import base64
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dtreeviz.trees import dtreeviz
import openai
from pre_defined import normalise_and_encode, get_good_correlation, decision_tree_classifier, random_forest_classifier

openai.api_key = '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'

def general_eda(df, data_y):
    # ask user for custom threshold values
    threshold = st.slider("threshold", 0.0, 1.0, 0.4)

    #get correlation and values for good correlation
    cor, good_cor = get_good_correlation(df, threshold)

    # plot correlation heatmap
    fig, ax = plt.subplots()
    # sns.heatmap(cor, annot=True, ax=ax)
    st.pyplot(sns.pairplot(cor))

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

#decision tree visulaization
def svg_write(svg, center=True):
    """
    Disable center to left-margin align like other objects.
    """
    # Encode as base 64
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")

    # Add some CSS on top
    css_justify = "center" if center else "left"
    css = f'<p style="text-align:center; display: flex; justify-content: {css_justify};">'
    html = f'{css}<img src="data:image/svg+xml;base64,{b64}"/>'

    # Write the HTML
    st.write(html, unsafe_allow_html=True)

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

        st.write(df.describe())

        st.dataframe(df.head())

        st.subheader("What is your variable of interest?", columns)
        data_y = st.selectbox('', options=columns)

        radio_option = st.radio('What do you want to do?', ('EDA', 'Model Data'))

        if radio_option == 'EDA':
            general_eda(df, data_y)
        elif radio_option == 'Model Data':
            radio_option = st.radio('What model do you want to use?', ('Decision Tree', 'Linear Regression', 'Random Forest', 'Logistic Regression'))
            if radio_option == 'Decision Tree':
                encoding = st.radio('What encoding do you want to perform on your data?', ('One Hot Encoding', 'Label Encoding', 'No Encoding'))
                if st.button('Create model'):
                    df = normalise_and_encode(
                        df, 
                        data_y, 
                        dct=True, 
                        encoding=encoding
                    )
                    model, predicted, score, report = decision_tree_classifier(df, data_y=data_y)
                    st.write(f'The model score is {score}')
                    st.subheader('Model Report')
                    st.table(report)   
                    viz = dtreeviz(
                            model,
                            x_data = df.drop(data_y, axis=1),
                            y_data = df[data_y],
                            target_name = data_y,
                            feature_names = df.drop(data_y, axis=1).columns,
                            class_names = ['Churned', 'Not Churned'],
                        )
                    svg_write(viz.svg())
                       
            elif radio_option == 'Linear Regression':
                options = st.multiselect('Select columns to model', columns)
                st.write('You selected:', options)

            elif radio_option == 'Random Forest':
                encoding = st.radio('What encoding do you want to perform on your data?', ('One Hot Encoding', 'Label Encoding', 'No Encoding'))
                if st.button('Create model'):
                    df = normalise_and_encode(df, data_y, dct=True, encoding=encoding)
                    model, predicted, score, report = random_forest_classifier(df, data_y=data_y)
                    st.write(f'The model score is {score}')
                    st.subheader('Model Report')
                    st.table(report)
    
            elif radio_option == 'Logistic Regression':
                pass
        

        

        # if options:
        #     st.subheader("Model from OpenAI inference")
        #     if st.button('Get Code'):
        #         with st.spinner('Wait for it...'):
        #             code = get_model(data_x=options, data_y=data_y)
        #         st.success('Done!')
        #         st.code(code, language='python')
        #         # run code
        #         if st.button('Run Code'):
        #             st.write(exec(code))
if __name__ == '__main__':
    main()