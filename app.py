import streamlit as st
from data_exploration import perform_data_exploration
from statistics_laws import perform_statistics_laws
from tests import perform_tests

def main():
    # Set page title and favicon
    st.set_page_config(
        page_title="Statistics Final Project",
        page_icon="📊",
        
    )

    # Navigation bar
    menu = ["Home", "Data Exploration", "Statistics Laws", "Statistical Test"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # Page content based on the selected choice
    if choice == "Home":
        st.title("Statistics Final Project")
        st.image("https://thelogisticsworld.com/wp-content/uploads/2022/02/concepto-de-big-data-700x476.jpg", caption="Data Management in Python", use_column_width=True)
        st.write("Welcome to the Home App!")

    elif choice == "Data Exploration":
        perform_data_exploration()
        
    elif choice == "Statistics Laws":
        perform_statistics_laws()
        
    elif choice == "Statistical Test":
        perform_tests()

# Run the Streamlit app
if __name__ == "__main__":
    main()
