import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
st.set_page_config(page_title='LEUKEMIA EDA Dashboard',layout='wide')
COLORS={'AML': '#e63946','Bone_Marrow':'#457b9d','PB': '#2a9d8f','PBSC_CD34':'#e9c46a','Bone_Marrow_CD34':'#f4a262'}
df=pd.read_csv(r'Leukemia.csv.csv')
gene_data=df.drop(columns=['samples','type'])
gene_variance=gene_data.var()
with st.sidebar:
    st.title("🧬 Leukemia EDA")
    st.markdown("---")
    st.markdown("Filter Data")
    all_types=df['type'].unique().tolist()
    selected_types=st.multiselect('Select Patient types',
                                  options=all_types,
                                  default=all_types)
    st.markdown('---')
    st.markdown("About")
    st.markdown("Dataset: GSE9476")
    st.markdown('Patients:64')
    st.markdown('Genes:22,283')
    st.markdown('Source:Kaggle')
    st.markdown('---')
    st.markdown('Bioinformatics Dashboard . Streamlit')
df_filtered=df[df['type'].isin(selected_types)]
st.title("🧬 Leukemia Gene Expression Dashboard")
st.markdown("Exploring the **GSE9476** dataset — 64 patients, 22,283 gene-expression features including AML and healthy reference groups.")
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔬 Gene Explorer", "🧪 Advanced analysis"])
with tab1:
    st.markdown('Patient Type distribution')
    st.markdown('This chart shows how many patients belong to each group. AML is the leukemia group — the rest are healthy reference groups.')
    fig,ax=plt.subplots(figsize=(8,4))
    type_counts=df_filtered['type'].value_counts()
    colors=[COLORS.get(t,'#888888')for t in type_counts.index]
    type_counts.plot(kind='bar',color=colors,edgecolor='black',ax=ax)
    ax.set_xlabel('Patient type')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x',rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown("---")
    st.markdown(" Dataset Preview")
    st.markdown("A quick look at the first 10 patients their sample ID and type.")
    st.dataframe(df_filtered[['samples', 'type']].head(10), use_container_width=True)
with tab2:
    st.markdown('Gene Expression Explorer')
    st.markdown('Select a gene from the dropdown to see how it is expressed accross all patients and patient types.High variance genes are shown first as they are  the most informative.')
    top50_genes=gene_variance.sort_values(ascending=False).head(50).index
    gene=st.selectbox('Select a gene to explore ',top50_genes)
    col1,col2=st.columns(2)
    with col1:
        st.markdown('Overall Distribution')
        st.markdown("How this gene's expression is spread across all 64 patients")
        fig,ax=plt.subplots()
        sns.histplot(df_filtered[gene],bins=20,kde=True,color='#e63946',ax=ax)
        ax.set_xlabel('Expression value')
        ax.set_ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with  col2:
        st.markdown('Expression by Patient Type')
        st.markdown('Does this gene behave differently in AML vs healthy groups?')
        fig,ax=plt.subplots()
        sns.boxplot(x='type',y=gene,data=df_filtered,palette=COLORS,ax=ax)
        ax.set_xlabel('')
        ax.tick_params(axis='x',rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
with tab3:
    st.markdown('Top 10 Most Variable genes')
    st.markdown('These genes vary the most accross all 64 patients they are the most likely to be biologically linked to leukemia and the msot useful for distiguishing patient types')
    top10=gene_variance.sort_values(ascending=False).head(10)
    fig,ax=plt.subplots(figsize=(10,5))
    top10.plot(kind='bar',color='#e63946',edgecolor='black',ax=ax)
    ax.set_xlabel('Gene')
    ax.set_ylabel('Variance')
    ax.tick_params(axis='x',rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('---')
    st.markdown('Correlation Heatmap - Top 10 variable genes')
    st.markdown('This show how related the top 10 genes are to each other')
    top10_genes=gene_variance.sort_values(ascending=False).head(10).index
    corr_matrix=df[top10_genes].corr()
    fig,ax=plt.subplots(figsize=(10,8))
    sns.heatmap(corr_matrix,annot=True,fmt='.2f',cmap='RdBu_r',center=0,ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    import plotly.express as px
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X=df.drop(columns=['type'])
    y=df['type']
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    pca=PCA(n_components=2)
    components=pca.fit_transform(X_scaled)
    pca_df=pd.DataFrame({'PC1':components[:,0],
                     'PC2':components[:,1],
                     'Patient Type':y})
    fig=px.scatter(pca_df,x='PC1',y='PC2',color='Patient Type',title=f'PCA Plot(explains{pca.explained_variance_ratio_.sum()*100:.1f}% of variance)')
    st.plotly_chart(fig,use_container_width=True)
st.markdown("""
    <style>
    [data-testid="stStatusWidget"] {display: none;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
