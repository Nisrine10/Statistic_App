# data_exploration.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

def perform_data_exploration():
    st.header("Data Exploration")

    # File upload section
    data_file = st.file_uploader("Upload Data File", type=["csv", "txt", "xlsx", "xls", "html"])

    if data_file is not None:
        try:
            if data_file.type == 'application/vnd.ms-excel':
                df = pd.read_excel(data_file)
            else:
                df = pd.read_csv(data_file)

            # Display DataFrame
            st.write("### Uploaded DataFrame:")
            st.write(df)

            # Sidebar controls
            st.sidebar.subheader("Data Options")

            # Display DataFrame columns
            if st.sidebar.checkbox("Show Columns", key="show_columns"):
                st.write("### DataFrame Columns:")
                st.write(df.columns.tolist())

            # Display specific line using df.loc
            if st.sidebar.checkbox("Show Specific Line", key="show_specific_line"):
                line_number = st.sidebar.slider("Enter specific line number:", 0, len(df) - 1, 0)
                st.write("### Loc (Specific Line) Result:")
                st.write(df.loc[line_number])

            # Display specific range using df.loc
            if st.sidebar.checkbox("Show Specific Range", key="show_specific_range"):
                start_line = st.sidebar.slider("Enter start line for range:", 0, len(df) - 1, 0)
                end_line = st.sidebar.slider("Enter end line for range:", start_line, len(df) - 1, len(df) - 1)
                st.write("### Loc (Specific Range) Result:")
                st.write(df.loc[start_line:end_line])

            # Calculations
            if st.sidebar.checkbox("Calculations", key="perform_calculations"):
                st.subheader("Calculations")

                # Descriptive Statistics
                st.write("### Descriptive Statistics:")
                desc_stats = pd.DataFrame({
                    'Mean': df.mean(),
                    'Variance': df.var(),
                    'Expectation': np.mean(df.values),
                    'Range': df.max() - df.min(),
                    'Mode': df.mode().iloc[0],
                    'Standard Deviation': df.std()
                })
                st.write(desc_stats)

            # Group by specific column and display mean
            if st.sidebar.checkbox("Group by and Display Mean", key="mean"):
                groupby_column = st.sidebar.selectbox("Select column for grouping:", df.columns.tolist())
                grouped_df = df.groupby(groupby_column)[[groupby_column]].mean()
                st.write("### Grouped DataFrame (Mean):")
                st.write(grouped_df)

            # Heatmap based on user input
            if st.sidebar.checkbox("Heatmap", key="heatmap"):
                st.subheader("Heatmap")
                selected_columns = st.sidebar.multiselect("Select columns for Heatmap:", df.columns.tolist())
                heatmap(df, selected_columns)

            # Line Plot based on user input
            if st.sidebar.checkbox("Line Plot", key="lineplot"):
                x_column_line = st.sidebar.selectbox("Select x-axis column for Line Plot:", df.columns.tolist())
                y_columns_line = st.sidebar.multiselect("Select y-axis column(s) for Line Plot:", df.columns.tolist())
                line_plot(df, x_column_line, y_columns_line)

           # Boxplot based on user input
            if st.sidebar.checkbox("Boxplot", key="boxplot"):
                try:
                    st.subheader("Boxplot")
                    x_column_boxplot = st.sidebar.selectbox("Select x-axis column for Boxplot:", df.columns.tolist())
                    y_column_boxplot = st.sidebar.selectbox("Select y-axis column for Boxplot:", df.columns.tolist())

                    # Check the validity of selected columns for Boxplot
                    if not (is_categorical_dtype(df[x_column_boxplot]) and is_numeric_dtype(df[y_column_boxplot])):
                        raise ValueError("For Boxplot, the x-axis column should be categorical and the y-axis column should be numeric.")

                    boxplot(df, x_column_boxplot, y_column_boxplot)
                except Exception as e:
                    st.error(f"An error occurred while generating the Boxplot: {e}")


                # Scatter Plot based on user input
                if st.sidebar.checkbox("Scatter Plot", key="scatterplot"):
                    x_column_scatter = st.sidebar.selectbox("Select x-axis column for Scatter Plot:", df.columns.tolist())
                    y_column_scatter = st.sidebar.selectbox("Select y-axis column for Scatter Plot:", df.columns.tolist())
                    
                    # Check the validity of selected columns for Scatter Plot
                    if not (is_numeric_dtype(df[x_column_scatter]) and is_numeric_dtype(df[y_column_scatter])):
                        raise ValueError("Both selected columns for Scatter Plot should be numeric.")

                    scatter_plot(df, x_column_scatter, y_column_scatter)

                # Histogram based on user input
                if st.sidebar.checkbox("Histogram", key="histog"):
                    x_column_hist = st.sidebar.selectbox("Select column for Histogram:", df.columns.tolist())
                    
                    # Check the validity of selected column for Histogram
                    if not is_numeric_dtype(df[x_column_hist]):
                        raise ValueError("Selected column for Histogram should be numeric.")

                        histogram(df, x_column_hist)
                # KDE Plot based on user input
                if st.sidebar.checkbox("KDE Plot", key="kde"):
                    x_column_kde = st.sidebar.selectbox("Select column for KDE Plot:", df.columns.tolist())
                    kde_plot(df, x_column_kde)

            # Violin Plot based on user input
                if st.sidebar.checkbox("Violin Plot", key="violin"):
                    x_column_violin = st.sidebar.selectbox("Select x-axis column for Violin Plot:", df.columns.tolist())
                    y_column_violin = st.sidebar.selectbox("Select y-axis column for Violin Plot:", df.columns.tolist())
                    
                # Check the validity of selected columns for Violin Plot
                if not (is_numeric_dtype(df[x_column_violin]) and is_numeric_dtype(df[y_column_violin])):
                    raise ValueError("Both selected columns for Violin Plot should be numeric.")

                violin_plot(df, x_column_violin, y_column_violin)

            # Bar Plot based on user input
            if st.sidebar.checkbox("Bar Plot", key="bar"):
                x_column_bar = st.sidebar.selectbox("Select x-axis column for Bar Plot:", df.columns.tolist())
                y_column_bar = st.sidebar.selectbox("Select y-axis column for Bar Plot:", df.columns.tolist())
                
                # Check the validity of selected columns for Bar Plot
                if not (is_categorical_dtype(df[x_column_bar]) and is_numeric_dtype(df[y_column_bar])):
                    raise ValueError("For Bar Plot, the x-axis column should be categorical and the y-axis column should be numeric.")

                bar_plot(df, x_column_bar, y_column_bar)

            # Pie Chart based on user input
            if st.sidebar.checkbox("Pie Chart", key="pie"):
                pie_column = st.sidebar.selectbox("Select column for Pie Chart:", df.columns.tolist())
                
                pie_chart(df, pie_column)


        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty. Please upload a file with data.")
        except pd.errors.ParserError:
            st.error("Error parsing the file. Please make sure it is a valid CSV, Excel, HTML, or TXT file.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

def heatmap(data, columns):
    st.subheader("Heatmap")

    # Set Seaborn style
    sns.set()

    # Create a heatmap
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(data[columns].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)

        # Add title
        ax.set_title("Heatmap")

        # Show the plot
        st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred while generating the heatmap: {e}")



def line_plot(data, x_column, y_columns):
    st.subheader("Line Plot")

    # Set Seaborn style
    sns.set()

    # Create a line plot with hover
    fig, ax = plt.subplots(figsize=(10, 6))
    for y_column in y_columns:
        sns.lineplot(x=data[x_column], y=data[y_column], marker='o', label=f'{y_column}', ax=ax)

    # Add labels and title
    ax.set(xlabel=x_column, ylabel="Values")  # You can customize the y-axis label
    ax.set_title(f'Line Plot: {", ".join(y_columns)} vs {x_column}')

    # Show legend
    ax.legend()

    # Display details on hover
    hover = st.sidebar.checkbox("Display Details on Hover")
    if hover:
        tooltips = [(col, f"@{col}") for col in [x_column] + y_columns]
        mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text('\n'.join([f"{label}: {sel.artist.get_array()[sel.index]:.2f}" for label, sel in zip(*sel.target)])), target=ax, tooltips=tooltips)

    # Show the plot
    st.pyplot(fig)

def scatter_plot(data, x_column, y_column):
    st.subheader("Scatter Plot")

    # Set Seaborn style
    sns.set()

    # Create a scatter plot with hover
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=data[x_column], y=data[y_column], marker='o', ax=ax)

    # Add labels and title
    ax.set(xlabel=x_column, ylabel=y_column)
    ax.set_title(f'Scatter Plot: {y_column} vs {x_column}')

    # Display details on hover
    hover = st.sidebar.checkbox("Display Details on Hover")
    if hover:
        tooltips = [(col, f"@{col}") for col in [x_column, y_column]]
        mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text('\n'.join([f"{label}: {sel.artist.get_array()[sel.index]:.2f}" for label, sel in zip(*sel.target)])), target=ax, tooltips=tooltips)

    # Show the plot
    st.pyplot(fig)

def boxplot(data, x_column, y_column):
    st.subheader("Boxplot")

    # Set Seaborn style
    sns.set()

    # Create a boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=data[x_column], y=data[y_column], ax=ax)

    # Add labels and title
    ax.set(xlabel=x_column, ylabel=y_column)
    ax.set_title(f'Boxplot: {y_column} vs {x_column}')

    # Show the plot
    st.pyplot(fig)

def histogram(data, x_column):
    st.subheader("Histogram")

    # Set Seaborn style
    sns.set()

    # Create a histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[x_column], bins=30, kde=True, ax=ax)

    # Add labels and title
    ax.set(xlabel=x_column, ylabel="Frequency")
    ax.set_title(f'Histogram: {x_column}')

    # Show the plot
    st.pyplot(fig)

def kde_plot(data, x_column):
    st.subheader("KDE Plot")

    # Set Seaborn style
    sns.set()

    # Create a KDE plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data[x_column], fill=True, ax=ax)

    # Add labels and title
    ax.set(xlabel=x_column, ylabel="Density")
    ax.set_title(f'KDE Plot: {x_column}')

    # Show the plot
    st.pyplot(fig)

def violin_plot(data, x_column, y_column):
    st.subheader("Violin Plot")

    # Set Seaborn style
    sns.set()

    # Create a violin plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x=data[x_column], y=data[y_column], ax=ax)

    # Add labels and title
    ax.set(xlabel=x_column, ylabel=y_column)
    ax.set_title(f'Violin Plot: {y_column} vs {x_column}')

    # Show the plot
    st.pyplot(fig)

def bar_plot(data, x_column, y_column):
    st.subheader("Bar Plot")

    # Set Seaborn style
    sns.set()

    # Create a bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=data[x_column], y=data[y_column], ax=ax)

    # Add labels and title
    ax.set(xlabel=x_column, ylabel=y_column)
    ax.set_title(f'Bar Plot: {y_column} vs {x_column}')

    # Show the plot
    st.pyplot(fig)

def pie_chart(data, column):
    st.subheader("Pie Chart")

    # Set Seaborn style
    sns.set()

    # Create a pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    data[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)

    # Add title
    ax.set_title(f"Pie Chart: {column}")

    # Show the plot
    st.pyplot(fig)

if __name__ == "__main__":
    perform_data_exploration()