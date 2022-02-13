import streamlit as st
import pandas as pd
import openai as gpt
import seaborn as sns
import matplotlib.pyplot as plt


# check if key exists in a list of dictionaries
def key_exists(k1, k2, list_of_dicts):
    for d in list_of_dicts:
        if k1 == d['Column 1'] and k2 == d['Column 2'] or k1 == d['Column 2'] and k2 == d['Column 1']:
            return True
    return False

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


def main():
    uploaded_file = st.file_uploader("Upload your file of structured data", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        columns = list(df.columns)

        # drop index column that is not needed
        if 'Unnamed: 0' in columns:
            df = df.drop(['Unnamed: 0'], axis=1)

        # output all columns in dataframe
        st.write(f"avaliable columns: {columns}")
        
        # ask user for custom threshold values
        threshold = st.slider("threshold", 0.0, 1.0, 0.15)

        #get correlation and values for good correlation
        cor, good_cor = get_good_correlation(df, threshold)

        # plot correlation heatmap
        fig, ax = plt.subplots()
        sns.heatmap(cor, annot=True, ax=ax)

        # get column to be predicted in our model
        st.subheader("What column do you want to predict?", columns)
        data_y = st.selectbox('', options=columns)
        
        # print the columns with an acceptable correlation
        st.markdown("## We reccomend using the following columns to create a model: ")
        st.table(good_cor)

        for i in good_cor:
            fig, ax = plt.subplots()
            sns.scatterplot(df[i['Column 1']], df[i['Column 2']], ax=ax)
            st.pyplot(fig)

        options = st.multiselect('Select columns to model', columns)

        st.write('You selected:', options)

if __name__ == '__main__':
    main()