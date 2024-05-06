"""
Name:       Alex Manes
CS 230:     Section 5
Data:       Starbucks Locations
URL:        TBD

Description: This is a mixture of visual, numerical, and comparative summaries
of Starbucks locations around the world. Numerous packages were used, including some not before used
in class. URLs for teaching of the use of these packages are provided below. Note:
other resources were used, but were not substantial enough in the aiding of small parts
of this project.

https://www.w3schools.com/python/matplotlib_pie_charts.asp
https://www.w3schools.com/python/matplotlib_bars.asp
https://www.geeksforgeeks.org/donut-chart-using-matplotlib-in-python/
https://www.youtube.com/watch?v=8G4cD7ofgCM

Code for haversine distance included in Word (ChatGPT use)
"""




import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pydeck as pdk

st.set_option('deprecation.showPyplotGlobalUse', False)

streamlit_path = "C:/Users/alexu/PycharmProjects/pythonProject/venv/pythonprojectshw/FinalProject/FinalProjectCode.py"
path = "C:/Users/alexu/PycharmProjects/pythonProject/venv/pythonprojectshw/FinalProject/stores.csv"
df_store = pd.read_csv(path, nrows= 10000)
df_store.dropna(subset=['Street1'], inplace=True)
def count_country_occurrences(df):
    country_counts = df['CountryCode'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Count']
    top_10_countries = country_counts.head(10)
    other_countries = country_counts['Count'][10:].sum()
    other_countries_row = pd.DataFrame({'Country': ['Other'], 'Count': [other_countries]})
    df_final = pd.concat([top_10_countries, other_countries_row])
    return df_final

def plot_bar_chart(df):
    plt.figure(figsize=(10, 6))
    plt.bar(df['Country'], df['Count'])
    plt.xlabel('Country')
    plt.ylabel('Number of Starbucks Locations')
    plt.title('Starbucks Locations by Country')
    plt.tight_layout()
    st.pyplot()

def plot_pie_chart(df):
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0',
              '#ffb3e6', '#c2f0c2', '#ffcc00', '#ff6666', '#c2d6d6', '#ffc2b3']
    plt.figure(figsize=(8, 8))
    plt.pie(df['Count'], labels=df['Country'], autopct='%1.1f%%', colors=colors, pctdistance=0.85)
    plt.title('Starbucks Locations by Country')
    plt.ylabel('')
    plt.tight_layout()
    plt.legend(df['Country'], loc='upper right')
    st.pyplot()

def plot_donut_chart(df):
    us_count = df[df['Country'] == 'US']['Count'].sum()
    non_us_count = df[df['Country'] != 'US']['Count'].sum()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie([us_count, non_us_count], labels=['United States', 'Foreign'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ffcc99'])
    ax.axis('equal')

    center_circle = plt.Circle((0, 0), 0.6, color='white', linewidth=0)
    fig.gca().add_artist(center_circle)

    plt.title('Distribution of Starbucks Locations', fontsize=16)

    st.pyplot(fig)

def filter_dataframe(df):
    country_code_1 = st.sidebar.selectbox("Select Country Code 1", df["CountryCode"].unique())
    filtered_df_1 = df[df["CountryCode"] == country_code_1]
    if country_code_1 == "US":
        state_code_1 = st.sidebar.selectbox("Select State Code 1", filtered_df_1["CountrySubdivisionCode"].unique())
        filtered_df_1 = filtered_df_1[filtered_df_1["CountrySubdivisionCode"] == state_code_1]

    country_code_2 = st.sidebar.selectbox("Select Country Code 2", df["CountryCode"].unique())
    filtered_df_2 = df[df["CountryCode"] == country_code_2]
    if country_code_2 == "US":
        state_code_2 = st.sidebar.selectbox("Select State Code 2", filtered_df_2["CountrySubdivisionCode"].unique())
        filtered_df_2 = filtered_df_2[filtered_df_2["CountrySubdivisionCode"] == state_code_2]

    return filtered_df_1, filtered_df_2

def filter_dataframe2(df):
    df = df[df["CountryCode"] == "US"]

    state_code = st.selectbox("Select State Code", df["CountrySubdivisionCode"].unique())

    filtered_df = df[df["CountrySubdivisionCode"] == state_code]

    city = st.selectbox("Select City", filtered_df["City"].unique())

    filtered_df = filtered_df[filtered_df["City"] == city]

    location = st.selectbox("Select Location", filtered_df["Name"])

    location_lat = filtered_df[filtered_df["Name"] == location]["Latitude"].values[0]
    location_lon = filtered_df[filtered_df["Name"] == location]["Longitude"].values[0]

    num_locations = st.slider("How many of the closest locations would you like to see?")

    closest_locations = find_closest_locations(df, location_lat, location_lon, int(num_locations))
    return closest_locations

# Following function based on code from ChatGPT, see Word document for explanation.
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of earth in kilometers is 6371
    distance_km = 6371 * c

    # Convert distance from kilometers to miles
    distance_miles = distance_km * 0.621371

    return distance_miles


def find_closest_locations(df, location_lat, location_lon, num_locations):
    df["Distance (Miles)"] = haversine_distance(location_lat, location_lon, df["Latitude"], df["Longitude"])

    closest_locations = df.sort_values(by="Distance (Miles)").head(num_locations)
    return closest_locations


def create_starbucks_scatterplot(df):
    map_df = df.filter(['Street1', 'Latitude', 'Longitude'])
    scatterplot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["Longitude", "Latitude"],
        get_radius=10000,
        get_fill_color=[0, 255, 255],
        pickable=True,
    )

    # code from the youtube video provided with HTML syntax to get formatting for the tool tip
    tool_tip = {'html': 'Location:<br/> <b>{Street1}</b>', 'style': {'backgroundColor': 'steelblue', 'color': 'white'}}
    view_state = pdk.ViewState(latitude=0, longitude=0, zoom=1)
    scatterplot_map = pdk.Deck(
        layers=[scatterplot_layer],
        initial_view_state=view_state,
        tooltip= tool_tip
    )

    st.pydeck_chart(scatterplot_map)

def main():
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Tabs", ["Main", "World", "United States", "Comparison"])
    if tab == "Main":
        st.title("Data Visualization with Python")
        st.write("Welcome! Please use the sidebar to choose a page.")
        image_path = "https://upload.wikimedia.org/wikipedia/en/thumb/d/d3/Starbucks_Corporation_Logo_2011.svg/300px-Starbucks_Corporation_Logo_2011.svg.png"
        st.image(image_path, caption="Starbucks Location "
                            "Finder", use_column_width=True)
    elif tab == "World":
        st.title("Visual Summary of the Distribution of Starbucks Locations")

        selected_visualizations = st.multiselect("Select Visualization Options",
                                                 ["Donut Chart", "Bar Chart", "Pie Chart", "Scatterplot"])

        if "Donut Chart" in selected_visualizations:
            country_counts = count_country_occurrences(df_store)
            plot_donut_chart(country_counts)
        if "Bar Chart" in selected_visualizations:
            country_counts = count_country_occurrences(df_store)
            plot_bar_chart(country_counts)
        if "Pie Chart" in selected_visualizations:
            country_counts = count_country_occurrences(df_store)
            plot_pie_chart(country_counts)
        if "Scatterplot" in selected_visualizations:
            create_starbucks_scatterplot(df_store)



    elif tab == "United States":

        st.title("Starbucks Locations in the United States")
        st.write("Find Closest Starbucks Locations by Selection")

        closest_locations = filter_dataframe2(df_store)

        st.write(closest_locations)

        second_closest = closest_locations.iloc[1]
        st.write(f"The closest Starbucks location that is not your own is {second_closest['Distance (Miles)']:.2f} miles away.")

        # Finding the information of the "second" closest store (which is really the 1st closest unique store) and stopping
        # after we find it
        for index, row in df_store.iterrows():
            if (row['Latitude'] == second_closest['Latitude']) and (row['Longitude'] == second_closest['Longitude']):
                st.write("Information of the closest location:")
                st.write(row)
                break
    elif tab == "Comparison":
        st.title("Comparison")
        st.write("Select Country and State for Comparison")

        # Selecting the first country / country-state
        country_code_1 = st.selectbox("Select Country Code 1", df_store["CountryCode"].unique())
        filtered_df_1 = df_store[df_store["CountryCode"] == country_code_1]
        if country_code_1 == "US":
            state_code_1 = st.selectbox("Select State Code 1", filtered_df_1["CountrySubdivisionCode"].unique())
            filtered_df_1 = filtered_df_1[filtered_df_1["CountrySubdivisionCode"] == state_code_1]

        # Selecting the second country / country-state
        country_code_2 = st.selectbox("Select Country Code 2", df_store["CountryCode"].unique())
        filtered_df_2 = df_store[df_store["CountryCode"] == country_code_2]
        if country_code_2 == "US":
            state_code_2 = st.selectbox("Select State Code 2", filtered_df_2["CountrySubdivisionCode"].unique())
            filtered_df_2 = filtered_df_2[filtered_df_2["CountrySubdivisionCode"] == state_code_2]

        num_locations_1 = len(filtered_df_1)
        num_locations_2 = len(filtered_df_2)
        difference = abs(num_locations_1 - num_locations_2)
        st.write(f"The difference in the number of locations between the two selected places is: {difference}")
        percent_change = abs((num_locations_1/num_locations_2)-1) * 100
        st.write(f"The percent difference in the number of locations between the two selected places is: {percent_change:.2f}%")


main()