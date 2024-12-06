import streamlit as st
import pandas as pd
import numpy as np
import ast
import json
import requests
import plotly.express as px
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib import cm
from streamlit_extras.grid import grid

# Set the page layout to wide
st.set_page_config(layout="wide")

st.sidebar.title("Navigation")

with st.sidebar:
    with st.expander("Background"):
        st.markdown("""
        The **Voting Rights Act**, federal legislation from 1965, provides protections for racial and language minorities. Section 203 specifies conditions under which local jurisdictions must offer language assistance during elections. Each jurisdiction is evaluated for each of 72 minority languages. If criteria are met, they must provide all election materials in the minority language.
        
        **To demonstrate** how differential privacy can be used, a synthetic dataset was created based off of the Census Public use files. Your task is to experiment with different implementation decisions to determine how differential privacy affects the accuracy of the release dataset.

        """)

    with st.expander("Dataset Description"):
        st.markdown("""
        The U.S. Census Bureau has released public use files that detail the data used to determine these requirements. A synthetic dataset was created based on these files to illustrate how differential privacy can be applied.
        This dataset contains individuals' answers to:
        1. **Do they speak primarily a language other than English?**
        2. **What languages other than English do they speak?**
        3. **Do they have above a 5th grade education level?**
        
        These factors determine if a county receives minority language assistance. For a detailed description, please examine the linked paper or the public use file record.
        
        """)

    with st.expander("Accuracy Metrics"):
        st.markdown("""
        **How Coverage is Determined:**
        - A county is covered under the Voting Rights Act if:
          - The percentage of non-English speaking individuals exceeds 5% of the total population, OR
          - The number of non-English speaking individuals exceeds 10,000, AND
          - The illiteracy rate among these individuals is greater than 1.31%.

        **Metrics:**
        - **Drop Rate:** Counties incorrectly not classified as needing assistance.
        - **Spurious Rate:** Counties incorrectly classified as needing assistance.
        - **Average Relative Error for Non English Speaking Individuals:** The average percentage difference between actual and DP-estimated counts.
        - **Average Relative Error for Illiterate Individuals:** The average percentage difference between actual and DP-estimated counts of illiterate individuals.
        """)

    with st.expander("Implementation Decisions"):
        st.markdown("""
        - **Epsilon:** Privacy budget where lower values increase privacy but decrease accuracy.
        - **Mechanism:** Noise application method (Laplace or Gaussian).
        - **Max Languages Spoken:** Limits the maximum contribution for number of languages a user can speak. 
        """)

# Define URLs for the datasets
alaska_synthetic_URL = "https://raw.githubusercontent.com/lpanavas/DP_Fairness_Datasets/refs/heads/main/alaska_synthetic_better_proportions_10_languages%20(4).csv"
alaska_real_URL = "https://raw.githubusercontent.com/lpanavas/DP_Fairness_Datasets/refs/heads/main/alaska_counties_no_dp.csv"

# Cache the data loading function
@st.cache_data
def load_data():
    with st.spinner('Loading data...'):
        df = pd.read_csv(alaska_synthetic_URL)
        df_counties_no_dp = pd.read_csv(alaska_real_URL)
    return df, df_counties_no_dp

# Load vacit_map from JSON file
with open('vacit_map.json', 'r') as f:
    vacit_map = json.load(f)
# Load county mapping from JSON file
with open('alaska_county_mapping.json', 'r') as f:
    county_mapping = json.load(f)

percentage_threshold = 0.05  # 5%
absolute_threshold = 10000   # Absolute threshold for LEP population
illiteracy_threshold = 0.0131 

# Convert strings to lists safely
def convert_to_list(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
    return value

# Define DP mechanism functions
def laplace_mechanism_vector(vector, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noise_vector = np.random.laplace(0, scale, size=len(vector))
    noisy_vector = np.round(vector + noise_vector).astype(int)  # Round to the nearest whole number
    noisy_vector = np.maximum(noisy_vector, 0)  # Ensure no values fall below 0
    return noisy_vector

def gaussian_mechanism_vector(vector, sensitivity, epsilon, delta):
    sigma = np.sqrt(2 * (sensitivity**2) * np.log(1.25 / delta)) / epsilon
    noise_vector = np.random.normal(0, sigma, size=len(vector))
    noisy_vector = np.round(vector + noise_vector).astype(int)  # Round to the nearest whole number
    noisy_vector = np.maximum(noisy_vector, 0)  # Ensure no values fall below 0
    return noisy_vector

# DP Noise Application
def apply_dp_noise_vector(row, epsilon, sensitivity, delta, mechanism):
    vector = np.array([row['ILLIT'], row['VACLEP'] - row['ILLIT'], row['total_population_vacit'] - row['VACLEP']])
    if mechanism == "Laplace":
        dp_vector = laplace_mechanism_vector(vector, sensitivity, epsilon)
    elif mechanism == "Gaussian":
        dp_vector = gaussian_mechanism_vector(vector, sensitivity, epsilon, delta)
    else:
        raise ValueError("Unsupported mechanism. Choose either 'Laplace' or 'Gaussian'.")

    dp_illit = dp_vector[0]
    dp_vaclep = dp_vector[0] + dp_vector[1]
    dp_vaclite = dp_vector[2]
    return dp_vaclep, dp_vaclite, dp_illit

@st.cache_data
def calculate_dp_covered(df, mechanism, epsilon, max_contribution, delta=None):
    mask_vaclep = df.groupby('PersonID')['VACLEP'].cumsum() <= max_contribution
    mask_illit = df.groupby('PersonID')['ILLIT'].cumsum() <= max_contribution
    df_filtered = df[mask_vaclep & mask_illit]

    df_aggregated = df_filtered.groupby(['NAMELSAD', 'LANCOUNT']).agg(
        VACLEP=('VACLEP', 'sum'),
        ILLIT=('ILLIT', 'sum'),
        total_population_vacit=('total_population_vacit', 'first')
    ).reset_index()

    sensitivity = max_contribution
    if mechanism == "Gaussian":
        sensitivity = np.sqrt(sensitivity)

    df_aggregated[['DP_VACLEP', 'DP_VACLIT', 'DP_ILLIT']] = df_aggregated.apply(
        lambda row: apply_dp_noise_vector(row, epsilon, sensitivity, delta, mechanism), axis=1, result_type="expand"
    )

    df_aggregated['dp_lep_percentage'] = df_aggregated['DP_VACLEP'] / df_aggregated['total_population_vacit']
    df_aggregated['dp_illiteracy_rate'] = df_aggregated['DP_ILLIT'] / df_aggregated['DP_VACLEP']

    df_aggregated['dp_coverage'] = (
        ((df_aggregated['dp_lep_percentage'] > percentage_threshold) |
         (df_aggregated['DP_VACLEP'] > absolute_threshold)) &
        (df_aggregated['dp_illiteracy_rate'] > illiteracy_threshold)
    )

    calculated_dp_covered = df_aggregated[df_aggregated['dp_coverage']]
    return df_aggregated, calculated_dp_covered

# Visualization tab in Streamlit
df, df_counties_no_dp = load_data()

df['LANCOUNT'] = df['LANCOUNT'].apply(convert_to_list)
df = df.explode("LANCOUNT").reset_index(drop=True)

# Use vacit_map loaded from JSON for mapping VACIT values onto df and df_counties_no_dp
df['total_population_vacit'] = df['NAMELSAD'].map(vacit_map)
df_counties_no_dp['total_population_vacit'] = df_counties_no_dp['NAMELSAD'].map(vacit_map)
df_counties_no_dp['lep_percentage'] = df_counties_no_dp['VACLEP'] / df_counties_no_dp['total_population_vacit']
df_counties_no_dp['illiteracy_rate'] = df_counties_no_dp['ILLIT'] / df_counties_no_dp['VACLEP']

df_counties_no_dp['coverage'] = (
    ((df_counties_no_dp['lep_percentage'] > percentage_threshold) |
     (df_counties_no_dp['VACLEP'] > absolute_threshold)) &
    (df_counties_no_dp['illiteracy_rate'] > illiteracy_threshold)
)

# Load Alaska GeoJSON file locally
geojson_path = './alaska-with-county-boundaries_1082.geojson'
gdf_counties = gpd.read_file(geojson_path)

# Visualization and User Inputs
# with st.tabs(["Visualization"])[0]:
    
st.header("Voting Rights Act Coverage under DP")
st.write("""
This visualization tool allows you to explore how differential privacy affects the determination of language assistance requirements under the Voting Rights Act. Adjust the privacy settings using the controls on the left to see how changes in epsilon, mechanism, and language limits impact the coverage of counties. The map will dynamically update to show counties that might be incorrectly excluded or included based on your settings. Use this to understand the trade-offs between privacy and accuracy in data protection scenarios.
""")
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    epsilon = st.number_input("Epsilon", min_value=0.0, value=.10, step=0.01, help="Epsilon controls the privacy budget; lower values mean higher privacy but less accuracy.")
    mechanism = st.selectbox("Mechanism", ["Laplace", "Gaussian"], help="Choose the noise mechanism for differential privacy.")
    max_languages = st.slider("Max number of languages spoken", min_value=1, max_value=72, value=72, help="Limits the number of languages considered, potentially reducing noise but also information.")
    delta = 1e-5 if mechanism == "Gaussian" else None

# Merge data as before
df_dp_alaska, calculated_dp_covered = calculate_dp_covered(df, mechanism, epsilon, max_languages, delta)
df1_selected = df_counties_no_dp[['NAMELSAD', 'LANCOUNT', 'total_population_vacit', 'VACLEP', 'ILLIT', 'lep_percentage', 'illiteracy_rate', 'coverage']]
df2_selected = df_dp_alaska[['NAMELSAD', 'LANCOUNT', 'DP_VACLEP', 'DP_VACLIT', 'DP_ILLIT', 'dp_lep_percentage', 'dp_illiteracy_rate', 'dp_coverage']]

merged_df = pd.merge(df1_selected, df2_selected, on=['LANCOUNT', 'NAMELSAD'], how='inner')

# Calculate dropped and spurious counts and counties
drop_count = merged_df[(merged_df['coverage'] == True) & (merged_df['dp_coverage'] == False)].shape[0]
spurious_count = merged_df[(merged_df['coverage'] == False) & (merged_df['dp_coverage'] == True)].shape[0]

# Calculate relative errors for VACLEP and ILLIT
merged_df['relative_error_vaclep'] = (merged_df['VACLEP'] - merged_df['DP_VACLEP']) / merged_df['VACLEP']
merged_df['relative_error_illit'] = (merged_df['ILLIT'] - merged_df['DP_ILLIT']) / merged_df['ILLIT']

# Replace infinities and NaNs in the relative errors (e.g., if VACLEP or ILLIT is zero)
merged_df['relative_error_vaclep'] = merged_df['relative_error_vaclep'].replace([np.inf, -np.inf], np.nan).fillna(0)
merged_df['relative_error_illit'] = merged_df['relative_error_illit'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Calculate average relative error per county
county_relative_errors = merged_df.groupby('NAMELSAD').agg(
    avg_relative_error_vaclep=('relative_error_vaclep', 'mean'),
    avg_relative_error_illit=('relative_error_illit', 'mean')
).reset_index()

# Add relative errors to the JSON results
results = {
    "drop_count": drop_count,
    "drop_counties": merged_df[(merged_df['coverage'] == True) & (merged_df['dp_coverage'] == False)]['NAMELSAD'].unique().tolist(),
    "spurious_count": spurious_count,
    "spurious_counties": merged_df[(merged_df['coverage'] == False) & (merged_df['dp_coverage'] == True)]['NAMELSAD'].unique().tolist(),
    "relative_error_vaclep": {
        row['NAMELSAD']: round(abs(row['avg_relative_error_vaclep']) * 100, 2) for _, row in county_relative_errors.iterrows()
    },
    "relative_error_illit": {
        row['NAMELSAD']: round(abs(row['avg_relative_error_illit']) * 100, 2) for _, row in county_relative_errors.iterrows()
    },
    "average_relative_error_vaclep": round(abs(county_relative_errors['avg_relative_error_vaclep'].mean()) * 100, 2),
    "average_relative_error_illit": round(abs(county_relative_errors['avg_relative_error_illit'].mean()) * 100, 2)
}

with col2:
    # Add radio buttons at the top for analysis selection
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Drop and Spurious Rate", "Relative Error Non English Speaking", "Relative Error Illiterate"],
        help="Choose what to visualize: counties that lost or gained coverage due to DP or the relative errors."
    )
    geojson = gdf_counties.__geo_interface__

    blues_cmap = cm.get_cmap('Blues')
    norm = Normalize(vmin=0, vmax=1)  # Normalize to 0-1 (0% to 100%)

    # Exploratory Tasks Section
    if analysis_type == "Drop and Spurious Rate":
        standardized_drop_counties = [county_mapping.get(name, name) for name in results["drop_counties"]]
        standardized_spurious_counties = [county_mapping.get(name, name) for name in results["spurious_counties"]]

        # Assign category to each county
        def assign_category(name):
            if name in standardized_drop_counties:
                return 'Dropped'
            elif name in standardized_spurious_counties:
                return 'Spurious'
            else:
                return 'Correctly Classified'

        gdf_counties['category'] = gdf_counties['name'].apply(assign_category)

        # Define color mapping for categories
        color_discrete_map = {
            'Dropped': 'blue',
            'Spurious': 'red',
            'Correctly Classified': 'lightgray'
        }

        # Create Plotly Choropleth Mapbox
        fig = px.choropleth_mapbox(
            gdf_counties,
            geojson=gdf_counties.__geo_interface__,
            locations='name',  # Match GeoDataFrame column with GeoJSON properties
            featureidkey="properties.name",  # GeoJSON property for county names
            color='category',  # Use categorical column for coloring
            color_discrete_map=color_discrete_map,  # Apply the custom categorical colors
            mapbox_style="carto-positron",
            center={"lat": 64.2008, "lon": -160.4937},
            zoom=2,  # Zoom out slightly
            title="Alaska Counties with Drop and Spurious Analysis"
        )

        # Update layout for margins
        fig.update_layout(
            legend_title_text="Category",  # Legend title
            margin={"r": 0, "t": 50, "l": 0, "b": 0}  # Remove extra margins
        )

    elif analysis_type == "Relative Error Non English Speaking":
        relative_error_vaclep = {
            county_mapping.get(name, name): error for name, error in results["relative_error_vaclep"].items()
        }

        # Assign relative error to gdf_counties
        gdf_counties['relative_error'] = gdf_counties['name'].map(
            lambda name: relative_error_vaclep.get(name, 0)  # Default to 0 if county not found
        )

        # Create Plotly Choropleth Mapbox
        fig = px.choropleth_mapbox(
            gdf_counties,
            geojson=geojson,
            locations='name',  # Column in gdf_counties matching the GeoJSON's features
            featureidkey="properties.name",  # Match GeoJSON feature with gdf_counties name
            color='relative_error',  # The column to be visualized
            color_continuous_scale="Blues",
            range_color=(0, 100),  # Range for relative error (as percentage)
            mapbox_style="carto-positron",
            center={"lat": 64.2008, "lon": -160.4937},
            zoom=2,  # Zoom out slightly
            title="Relative Error Non English Speaking (Max 100%)"
        )

        fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})  # Remove extra margins

    elif analysis_type == "Relative Error Illiterate":
        relative_error_illit = {
            county_mapping.get(name, name): error for name, error in results["relative_error_illit"].items()
        }

        # Assign relative error to gdf_counties
        gdf_counties['relative_error'] = gdf_counties['name'].map(
            lambda name: relative_error_illit.get(name, 0)  # Default to 0 if county not found
        )

        # Create Plotly Choropleth Mapbox
        fig = px.choropleth_mapbox(
            gdf_counties,
            geojson=geojson,
            locations='name',
            featureidkey="properties.name",
            color='relative_error',
            color_continuous_scale="Blues",
            range_color=(0, 100),
            mapbox_style="carto-positron",
            center={"lat": 64.2008, "lon": -160.4937},
            zoom=2,  # Zoom out slightly
            title="Relative Error Illiterate (Max 100%)"
        )

        fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})

    # Display the plot in Streamlit
    st.plotly_chart(fig)

with col3:
    unique_drop_count = len(results['drop_counties'])
    unique_spurious_count = len(results['spurious_counties'])
    st.subheader("Accuracy Metrics")
    error_metrics = {
        "Metric": ["Drop Rate", "Spurious Rate", "Avg. Error Non English Speaking", "Avg. Error Illiterate"],
        "Value": [unique_drop_count, unique_spurious_count, f"{results['average_relative_error_vaclep']}%", f"{results['average_relative_error_illit']}%"]
    }
    
    # Create DataFrame and reset the index
    df = pd.DataFrame(error_metrics)
    df = df.set_index("Metric")  # Set "Metric" column as the index
    
    # Display the table without an index column
    st.table(df)

st.header("Learning Tasks")# Task 1 - Naive Sensitivity

exp_grid = grid(2, [0.5, 0.5])  # 2 Rows, cells split 50%-50%
with exp_grid.expander("**Task: Naive Sensitivity Analysis**"):
    st.markdown('''
    **Task:**  
    Find the minimum epsilon needed to have no drop or spurious rates when the maximum number of languages spoken is 72.

    **What to look for:**  
    - Is the minimum epsilon generally acceptable?

    **Takeaways:**  
    72 is the theoretical maximum number of languages one could speak. This analysis naively protects extreme outliers but isn't realistic. Setting parameters without background knowledge results in a very high epsilon to maintain utility.
    ''')

# Task 2 - Minimum Epsilon
with exp_grid.expander("**Task: Finding Minimum Epsilon**"):
    st.markdown('''
    **Task:**  
    Determine the lowest epsilon where there are consistently no dropped or spurious counties. Add or subtract 0.01 from epsilon to check consistency.

    **What to look for:**  
    - How do different mechanisms affect accuracy?
    - How does changing the sensitivity (number of languages) impact accuracy?

    **Takeaways:**  
    This demonstrates what parameter tuning might look like in practice. Typically, tuning is done on public data before applying to private data. Notice how all parameters interact to change accuracy.
    ''')

# Task 3 - Bias/Variance Tradeoff
with exp_grid.expander("**Task: Exploring Bias/Variance Tradeoff**"):
    st.markdown('''
    **Task:**  
    Set epsilon to 1000 (virtually no privacy), clip the maximum languages spoken to 1. Gradually increase the number of languages and observe accuracy changes.

    **What to look for:**  
    - Does clipping introduce error?
    - What does the bias look like?
    - What might be an appropriate clipping bound for language count?

    **Takeaways:**  
    Clipping by setting bounds introduces bias, potentially altering the data's signal significantly. Capturing the full signal with high sensitivity (like 72 languages) can also lead to signal loss. DP implementers must balance this bias-variance tradeoff.
    ''')

# Task 4 - Mechanisms Comparison
with exp_grid.expander("**Task: Comparing Mechanisms**"):
    st.markdown('''
    **Task:**  
    Change the maximum number of languages to 5, 30, and 50. Switch mechanisms between Laplace and Gaussian to see which results in lower relative errors.

    **What to look for:**  
    - Which mechanism performs better at lower sensitivities?
    - Which mechanism performs better at higher sensitivities?

    **Takeaway:**  
    The Gaussian mechanism generally outperforms Laplace at higher sensitivities since L2 sensitivity is the square root of L1 sensitivity. At higher sensitivities, Gaussian might be preferable despite Laplace's narrower noise distribution.
    ''')
