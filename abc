import streamlit as st
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from streamlit_extras.metric_cards import style_metric_cards
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode
from st_aggrid import GridOptionsBuilder, AgGrid, DataReturnMode, GridUpdateMode
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode,ColumnsAutoSizeMode, JsCode
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder, JsCode


#import SessionState

# Setting page as wide
st.set_page_config(layout="wide")

# Heading for the app
st.markdown("<h1 style='text-align: center;'>Category Roles Toolkit</h1>", unsafe_allow_html=True)
#st.markdown('---')
#st.markdown("<br>", unsafe_allow_html=True) 

# Define your sidebar content

st.sidebar.image("logo4.png", use_column_width=True)
st.sidebar.markdown("*Powered by RACE 📊*")

st.sidebar.markdown("Category Roles Toolkit to optimize inventory management and strengthening brand positioning!")

#Adding Country Filter

st.sidebar.text("")
st.sidebar.markdown("Select the Options Below:")


# Create a dictionary to map the country codes to country names
country_mapping = {
    '🇦🇪 UAE': 'UAE',
    '🇪🇬 EGYPT': 'Egypt',
    '🇸🇦 KSA': 'KSA',
    '🇶🇦 Qatar': 'Qatar',
    '🇬🇪 Georgia': 'Georgia'
}

# Get the selected country from the sidebar
selected_country = st.sidebar.selectbox(
    'COUNTRY',
    ('🇦🇪 UAE', '🇪🇬 EGYPT')) # replace with your countries




if country_mapping[selected_country] == 'UAE':
    
    df = pd.read_csv('data_category.csv')
    df = df[df['Country']=='UAE']

    #st.markdown('🇦🇪')
    st.markdown("<h1 style='text-align: left;'>🇦🇪 UAE</h1>", unsafe_allow_html=True)
    #st.markdown("<h1 style='text-align: left;'>🇯🇴 JORDAN</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    
    
else: 
    
    df = pd.read_csv('data_category.csv')
    df = df[df['Country']=='UAE']
    st.markdown("<h1 style='text-align: left;'>🇪🇬 EGYPT</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    

# Filter the DataFrame based on the selected country
#df = df[df['COUNTRY'] == country_mapping[selected_country]]

data_sub_uae = pd.read_csv('uae_subs.csv')
data_sub_egp = pd.read_csv('eg_subs.csv')    
    
#Importing dataframe



def filter_data(df):
    col1, col2, col3, col4 = st.columns(4)
    filters = {}

    with col1:
        filter_1 = st.selectbox('STORE TYPE', options=np.insert(df['STORE TYPE'].unique(), 0, 'Supermarket'), index=0)
    if filter_1 != 'Supermarket':
        df = df[df['STORE TYPE'] == filter_1]
        filters['STORE TYPE'] = filter_1

        
        
    if not df.empty:
        with col2:
            filter_2 = st.selectbox('SECTION NAME', options=np.insert(df['SECTION'].unique(), 0, 'All'))
        if filter_2 != 'All':
            df = df[df['SECTION'] == filter_2]
            filters['SECTION'] = filter_2

    if not df.empty:
        with col3:
            filter_3 = st.selectbox('FAMILY NAME', options=np.insert(df['FAMILY NAME'].unique(), 0, 'All'))
        if filter_3 != 'All':
            df = df[df['FAMILY NAME'] == filter_3]
            filters['FAMILY NAME'] = filter_3

    if not df.empty:
        with col4:
            filter_4 = st.selectbox('CATEGORY ROLE', options=np.insert(df['CATEGORY ROLE - BASED ON DATA'].unique(), 0, 'All'))
        if filter_4 != 'All':
            df = df[df['CATEGORY ROLE - BASED ON DATA'] == filter_4]
            filters['CATEGORY ROLE - BASED ON DATA'] = filter_4

    

    return df, filters


def uae_compute_kpis(grouped_df):
    # Calculate the KPIs
    grouped_df['Keep sku percent'] = ((grouped_df['Keep sku count'] / (grouped_df['Keep sku count'] + grouped_df['Delist sku count']))*100).round(2)
    grouped_df['Keep Sales percent'] = ((grouped_df['Keep sales'] / (grouped_df['Keep sales'] + grouped_df['Delist sales']))*100).round(2)

    grouped_df['Delist sku percent'] = ((grouped_df['Delist sku count'] / (grouped_df['Keep sku count'] + grouped_df['Delist sku count']))*100).round(2)
    grouped_df['Delist Sales percent'] = ((grouped_df['Delist sales'] / (grouped_df['Keep sales'] + grouped_df['Delist sales']))*100).round(2)

    # Replace NaN values in specific columns
    grouped_df[['Keep sku percent', 'Keep Sales percent', 'Delist sku percent', 'Delist Sales percent']] = grouped_df[['Keep sku percent', 'Keep Sales percent', 'Delist sku percent', 'Delist Sales percent']].fillna(0)
    
    return grouped_df

# Define a function to compute the KPIs
def eg_compute_kpis(grouped_df):
    # Calculate the KPIs
    grouped_df['Delist (% ITEMS)'] = ((grouped_df['Delist SKU count'] / grouped_df['Total items'])*100).round(2)
    grouped_df['Review service or delist (% ITEMS)'] = ((grouped_df['Review service level or delist SKU count'] / grouped_df['Total items'])*100).round(2)
    grouped_df['Delist + Review service level (% ITEMS)'] = grouped_df['Delist (% ITEMS)'] + grouped_df['Review service or delist (% ITEMS)']

    grouped_df['Delist (% Sales)'] = ((grouped_df['Delist Sales'] / grouped_df['Total Sales'])*100).round(2)
    grouped_df['Review service or delist (% Sales)'] = ((grouped_df['Review service level or Delist Sales'] / grouped_df['Total Sales'])*100).round(2)
    grouped_df['Delist + Review service level (% Sales)'] = grouped_df['Delist (% Sales)'] + grouped_df['Review service or delist (% Sales)']
    
    return grouped_df

df_filtered, filters = filter_data(df)


#st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True) 
    
# Font size, and other settings

fontsize = 22
valign = "left"
lnk = '<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.12.1/css/all.css" crossorigin="anonymous">'


# # Initialize your metrics
# metrics = [
#     ("Category Analyzed", df_filtered['FAMILY'].nunique()),
#     ("Sub-Families Analyzed", len(df_filtered['SUBFAMILY'].unique())),
#     ("Brands Analyzed", len(df_filtered['BRAND'].unique())),
#     ("SKUs Analyzed", f"{df_filtered.shape[0]:,}"),
#     ("SKUs Delisted", len(df_filtered[df_filtered['BUSINESS RECOMMENDATION']=='Delist'])),
#     ("%SKUs Delisted", f"{len(df_filtered[df_filtered['BUSINESS RECOMMENDATION']=='Delist']) / len(df_filtered) * 100:.2f}%"),
#     ("%Sales Delisted SKUs", f"{(((pd.to_numeric(df_filtered[df_filtered['BUSINESS RECOMMENDATION']=='Delist']['SALES'], errors='coerce')).sum())-((pd.to_numeric(df_filtered[df_filtered['BUSINESS RECOMMENDATION']=='Delist']['TRANSFERENCE'], errors='coerce')).sum()))/ (pd.to_numeric(df_filtered['SALES'], errors='coerce')).sum() * 100:.2f}%"),
# ]
# Remove commas from the 'TOTAL SALES (LOCAL CUR)' column and convert to numeric
#df_filtered['TOTAL SALES (LOCAL CUR)'] = df_filtered['TOTAL SALES (LOCAL CUR)'].str.replace(',', '')

# Now, you can safely convert to numeric
df_filtered['TOTAL SALES (LOCAL CUR)'] = pd.to_numeric(df_filtered['TOTAL SALES (LOCAL CUR)'])
# Format the total_sales with commas as thousand separators
total_sales = df_filtered['TOTAL SALES (LOCAL CUR)'].sum().astype(int)
formatted_total_sales = "{:,}".format(total_sales)
# Initialize your metrics
metrics = [
    ("Category Analyzed", df_filtered['FAMILY NAME'].nunique()),
    ("Total Sales",formatted_total_sales) ,
    # ("Brands Analyzed", "{:,}".format(len(df_filtered['FAMILY NAME'].unique()))),
    # ("SKUs Analyzed", f"{df_filtered.shape[0]:,}"),
    # ("SKUs Delisted", "{:,}".format(len(df_filtered[df_filtered['BUSINESS RECOMMENDATION']=='Delist']))),
    # ("%SKUs Delisted", f"{len(df_filtered[df_filtered['BUSINESS RECOMMENDATION']=='Delist']) / len(df_filtered) * 100:.2f}%"),
    # ("%Sales Delisted SKUs", f"{(((pd.to_numeric(df_filtered[df_filtered['BUSINESS RECOMMENDATION']=='Delist']['SALES'], errors='coerce')).sum())-((pd.to_numeric(df_filtered[df_filtered['BUSINESS RECOMMENDATION']=='Delist']['TRANSFERENCE'], errors='coerce')).sum()))/ (pd.to_numeric(df_filtered['SALES'], errors='coerce')).sum() * 100:.2f}%"),

]


# Icon classes from Font Awesome
icons = [
    "fas fa-tags",
    "fas fa-layer-group",
    "fas fa-briefcase",
    "fas fa-barcode",
    "fas fa-thumbs-down",
    "fas fa-chart-bar",
    "fas fa-chart-pie",
]

# Color tuples
color_boxes = [(204, 239, 255)] * 7


wch_colour_font = (0, 0, 0)  # black font color

# Create 7 columns for the 7 metrics
cols = st.columns(2)

for i in range(2):
    sline, metric = metrics[i]
    htmlstr = f"""<p style='background-color: rgb({color_boxes[i][0]}, 
                                              {color_boxes[i][1]}, 
                                              {color_boxes[i][2]}, 0.75); 
                        color: rgb({wch_colour_font[0]}, 
                                   {wch_colour_font[1]}, 
                                   {wch_colour_font[2]}, 0.75); 
                        font-size: {fontsize}px; 
                        border-radius: 7px; 
                        padding-left: 12px; 
                        padding-top: 18px; 
                        padding-bottom: 18px; 
                        line-height:25px;'>
                        <i class='{icons[i]} fa-xs'></i> {metric}
                        </style><BR><span style='font-size: 14px; 
                        margin-top: 0;'>{sline}</style></span></p>"""

    cols[i].markdown(lnk + htmlstr, unsafe_allow_html=True)



st.markdown("---") 


##################################### working code #############################
#df_filtered['SELECT REASON']= df_filtered['SELECT REASON'].astype(str)
#df_filtered['SELECT REASON'].replace(np.nan, '-')
    
    

# Initialize GridOptionsBuilder
gb = GridOptionsBuilder.from_dataframe(df_filtered)

# Configure the column
dropdown_options = ('Delist', 'Keep', 'No Recommendation')
dropdown_options_2 = ('DESTINATION (WIN)', 'SERVICE (CONTROL)', 'TRAFFIC (DEFEND)','IMPULSE (BASKET BUILDER)')

# Define the formatter function in JavaScript. This will be run in the browser.
js_format_code = """
function(params) {
    var sales = params.value;
    return sales.toLocaleString('en-US', {maximumFractionDigits:1});
}
"""
cellsytle_jscode11 = JsCode("""
function(params) {
    if (params.value == 'DESTINATION (WIN)') {
        return {
            'color': 'black',
            'backgroundColor': '#FFCCCC'  // light red
        }
    } else if (params.value == 'SERVICE (CONTROL)') {
        return {
            'color': 'black',
            'backgroundColor': '#C8E6C9'  // light green
        }
    } else if (params.value == 'TRAFFIC (DEFEND)') {
        return {
            'color': 'black',
            'backgroundColor': '#F5F5F5'  // light gray
        }
    } else {
        return {
            'color': 'black',
            'backgroundColor': '#B0E0E6'
        }
    }
};
""")

cellsytle_jscode12 = JsCode("""
function(params) {
    if (params.value == 'DESTINATION (WIN)') {
        return {
            'color': 'black',
            'backgroundColor': '#FFC1C1'  // light red
        }
    } else if (params.value == 'SERVICE (CONTROL)') {
        return {
            'color': 'black',
            'backgroundColor': '#C8E6C9'  // light green
        }
    } else if (params.value == 'TRAFFIC (DEFEND)') {
        return {
            'color': 'black',
            'backgroundColor': '#F5F5F5'  // light gray
        }
    } else if (params.value == 'IMPULSE (BASKET BUILDER)') {
        return {
            'color': 'black',
            'backgroundColor': '#ADD8E6'  // light blue
        }
    } else if (params.value == 'Back Margin') {
        return {
            'color': 'black',
            'backgroundColor': '#B0E0E6'  // Powder blue
        }
    } else if (params.value == 'Future Trends') {
        return {
            'color': 'black',
            'backgroundColor': '#87CEFA'  // Light sky blue
        }
    } else if (params.value == 'Strategic Initiative') {
        return {
            'color': 'black',
            'backgroundColor': '#B0C4DE'  // Light steel blue
        }
    } else if (params.value == 'Others') {
        return {
            'color': 'black',
            'backgroundColor': '#E0FFFF'  // Light cyan
        }
    } else {
        return {
            'color': 'black',
            'backgroundColor': 'white'
        }
    }
};
""")

# Define the JavaScript format code for percentage formatting
js_format_code_per = """
    function(params) {
        var percentage = params.value * 100;
        return percentage.toLocaleString('en-US', {maximumFractionDigits: 1}) + '%';
    }
    """
    
    # Define the JavaScript sort comparator for percentage sorting
js_sort_comparator_per = """
    function(valueA, valueB, nodeA, nodeB, isInverted) {
        var numberA = parseFloat(valueA);
        var numberB = parseFloat(valueB);
        if (isNaN(numberA) || isNaN(numberB)) {
            return 0;
        }
        return (numberA - numberB) * (isInverted ? -1 : 1);
    }
    """
    
per_columns =["GROWTH RATE","PROFITABILITY","PENETRATION RATE","LOYALTY PEN SALES","FRONT MARGIN", "BACK MARGIN"]
    # Configure PERCENTAGE column

for column in per_columns:
            gb.configure_column(
                column,
                header_name=column,
                editable=False,
                type=["numericColumn", "numberColumnFilter"],
                valueFormatter=JsCode(js_format_code_per),
                sortComparator=JsCode(js_sort_comparator_per),
                precision=2,
            )


    # Configure columns to be pinned to the left by default
columns_to_pin = ["FAMILY NAME"]
for column in columns_to_pin:
        gb.configure_column(
            column,
            header_name=column,
            pinned="left",  # Pin to the left by default
        )

    # Configure column to be configured with comma seprated
columns_to_configure_comma_seprated = [
        "TOTAL SALES (LOCAL CUR)", "ASSOBASKET SPEND (LOCAL CUR)" ]
    

for column in columns_to_configure_comma_seprated:
        gb.configure_column(
            column,
            header_name=column,
            editable=False,
            type=["numericColumn", "numberColumnFilter"],
            valueFormatter=JsCode(js_format_code),
            precision=0
        )
#     # Configure the column where you want to allow editing
# gb.configure_column("dropdown_options_2", editable=True,cellEditor='agSelectCellEditor',cellStyle=cellsytle_jscode12, cellEditorParams={'values': dropdown_options_2})
    
    # gb.configure_column("BUSINESS RECOMMENDATION", editable=True, cellEditor='agSelectCellEditor', cellStyle=cellsytle_jscode11, cellEditorParams={'values': dropdown_options})
    

#gb.configure_grid_options(domLayout='normal')
#gb.configure_grid_options(**other_options)
gb.configure_default_column(min_column_width=200, sorteable=True)


# js = JsCode("""
# function(e) {
#     let api = e.api;
#     let rowIndex = e.rowIndex;
#     let col = e.column.colId;

#     let rowNode = api.getDisplayedRowAtIndex(rowIndex);
#     api.flashCells({
#       rowNodes: [rowNode],
#       columns: [col],
#       flashDelay: 10000000000
#     });

# };
# """)

#gb.configure_grid_options(onCellValueChanged=js) 


gridOptions = gb.build()

# Create a Streamlit app
#st.markdown("### SKUs Level Recommendation")
#st.markdown("---")
#st.markdown('## Delisting Summary')
#st.markdown('##### Here is summary data for delsting, you can provide your recommedation below :')
#st.tooltip("Hover over me for more information", "Click on checkbox to generate insights and double click to refresh data")
#st.radio(options=list, help="Click on checkbox to generate insights and double click to refresh data"

#help_input="This is the line1\n\nThis is the line 2\n\nThis is the line 3"
#st.text_area("Test",help=help_input)
#st.markdown("*Click on checkbox to generate insights and double click to refresh data*")
# return_mode = st.sidebar.selectbox("Return Mode", list(DataReturnMode.__members__), index=1)
# return_mode_value = DataReturnMode.__members__[return_mode]

# update_mode = st.sidebar.selectbox("Update Mode", list(GridUpdateMode.__members__), index=len(GridUpdateMode.__members__)-1)
# update_mode_value = GridUpdateMode.__members__[update_mode]


st.markdown("### Category Roles Summary Table")
st.markdown("---")
    
response = AgGrid(
        df_filtered,
        gridOptions=gridOptions,
        height=800,
        columns_auto_size_mode=2,
        width='100%',
        data_return_mode= DataReturnMode.FILTERED_AND_SORTED, # # GridUpdateMode.SELECTION_CHANGED or GridUpdateMode.VALUE_CHANGED or 
        update_mode=GridUpdateMode.MANUAL,#GridUpdateMode.MODEL_CHANGED
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=True,
        reload_data=False,
        theme= 'balham',#'material',#'balham',#'alpine',
        enable_enterprise_modules=False
    )

# #response['data'] = df_filtered
# if response['data'] is not None:  # Check if any data is returned
#     df_filtered = response['data']  # Overwrite your data with the updated data
#     st.dataframe(df_filtered)
# st.markdown("")
# Create eight columns
# Create eight columns

#######################################working code###############################################
# col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

# # Initialize df_filtered as an empty DataFrame
# #df_filtered = pd.DataFrame()

# # Add a button to the seventh column
# if col7.button('Submit Changes',key='DownloadButton11'):
#     df_filtered = response['data']
#     st.sidebar.markdown("")
#     st.sidebar.markdown("")
    
#     st.sidebar.warning("Changes are saved ✅")
#     # Setting a common index (if possible)
#     # Setting "ITEM NAME" as the index in both dataframes
#     df1 = df_filtered.set_index('ITEM NAME')
#     df2 = df.set_index('ITEM NAME')
    
#     # align the dataframes
#     df1, df2 = df1.align(df2)

#     # find rows where 'Business recommendation' is different
#     different_values_df = df1[df1['BUSINESS RECOMMENDATION'] != df2['BUSINESS RECOMMENDATION']]
    
#     st.dataframe(different_values_df)

#     # Convert dataframe to csv 
#     #csv = df_filtered.to_csv(index=False)

# # Create download button in the eighth column

# col8.download_button(
#     label="Download data",
#     data=df_filtered.to_csv(index=False),
#     file_name="my_data.csv",
#     mime="text/csv",
# )

################################################################################################
# import time
# st.markdown("")
# st.markdown("""
#     <style>
#     .stButton>button {
#         background-color: white;
#         color: black;
#     }
#     .stButton>button:hover {
#         background-color: #B22222;
#         color: white;
#     }
#     </style>
#     """, unsafe_allow_html=True)



# col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
# col6.button('Updated Changes',key='DownloadButton11')


# # # Add a button to the seventh column
# # if col6.button('Show Updated SKUs',key='DownloadButton11'):
# #     st.sidebar.markdown("")
# #     st.sidebar.markdown("")
    
# #     # Initialize the progress bar
# #     my_bar = st.sidebar.progress(0)
# #     progress_text = "Operation in progress. Please wait."
# #     st.sidebar.markdown(progress_text)

# #     df_filtered = response['data']
    
# #     # Setting a common index (if possible)
# #     # Setting "ITEM NAME" as the index in both dataframes
# #     df1 = df_filtered.set_index('ITEM NAME')
# #     df2 = df.set_index('ITEM NAME')
    
# #     # align the dataframes
# #     df1, df2 = df1.align(df2)

# #     # find rows where 'Business recommendation' is different
# #     different_values_df = df1[df1['BUSINESS RECOMMENDATION'] != df2['BUSINESS RECOMMENDATION']]
# #     st.markdown("")
# #     st.markdown("### List of Updated SKUs")
# #     # Create an expander named 'Product Substitute'
# #     updated_skus = st.expander('List of Updated SKUs')
# #     with updated_skus:
        
# #         st.dataframe(different_values_df)
    
    

# #     # Increase the progress bar to 50%
# #     my_bar.progress(50)

# #     # Increase the progress bar to 100%
# #     my_bar.progress(100)
# #     st.sidebar.markdown("Operation complete!")


    
# # Create download button in the eighth column
# col7.download_button(
#     label="Download dataset",
#     data=df_filtered.to_csv(index=False),
#     file_name="my_data.csv",
#     mime="text/csv",
# )

# # if st.button('Submit Changes'):
# #     df_filtered = response['data']
# #     #df = response['data']
# #     df = response['data']
# #     st.write(df_filtered)
# #st.write(response)
    
# # #Show selected rows
# if response['selected_rows']:

#     #st.subheader('Selected rows:')
#     #st.write(response['selected_rows'])
#     st.sidebar.markdown("<br>", unsafe_allow_html=True) 
#     st.sidebar.markdown("🅾️ Status of Summary Generation: ")
    

#     # Create a DataFrame from the selected row data
#     selected_row_df = pd.DataFrame(response['selected_rows'], index=[0])
#     #st.dataframe(response['data'])

#     # Filter by brand and calculate the desired values for Delisting
#     filtered_delisting_df1 = response['data'][response['data']['BRAND'] == selected_row_df['BRAND'][0]]

#     filtered_delisting_df2 = filtered_delisting_df1[filtered_delisting_df1['BUSINESS RECOMMENDATION']== 'Delist']

#     delisting_metrics = {
#         "Brand Name": selected_row_df['BRAND'][0],
#         "%SKU": round((len(filtered_delisting_df2) / len(filtered_delisting_df1) * 100),2),
#         "Absolute Count SKU": len(filtered_delisting_df2),
#         "Sales": (pd.to_numeric(filtered_delisting_df2['SALES'], errors='coerce')).sum(),
#         "Sales %": round(((pd.to_numeric(filtered_delisting_df2['SALES'], errors='coerce')).sum() / (pd.to_numeric(filtered_delisting_df1['SALES'], errors='coerce')).sum() * 100),2)
#     }
#     delisting_df = pd.DataFrame([delisting_metrics])


#     # Filter by brand and calculate the desired values for Keep
#     filtered_keep_df3 = filtered_delisting_df1[filtered_delisting_df1['BUSINESS RECOMMENDATION'] == 'Keep']
#     keep_metrics = {
#         "Brand Name": selected_row_df['BRAND'][0],
#         "%SKU": round((len(filtered_keep_df3) / len(filtered_delisting_df1) * 100),2),
#         "Absolute Count SKU": len(filtered_keep_df3),
#         "Sales": (pd.to_numeric(filtered_keep_df3['SALES'], errors='coerce')).sum(),
#         "Sales %": round(((pd.to_numeric(filtered_keep_df3['SALES'], errors='coerce')).sum() / (pd.to_numeric(filtered_delisting_df1['SALES'], errors='coerce')).sum() * 100),2)
#     }
#     keep_df = pd.DataFrame([keep_metrics])



#     st.markdown('---')
#     st.markdown('### Brand Level Summary')
# # # Create an expander named 'Brand & Supplier Summary'
#     my_expander = st.expander('Brand Level Summary')

#     with my_expander:
#         col1, col2 = st.columns(2)
#         # Display 'Brand Level Summary' using AgGrid
#         with col1:
#             st.markdown("#### Brand Delist Level")
#             #AgGrid(delisting_df,theme='alpine',height=160)
#             delisting_df.columns = delisting_df.columns.str.upper()
#             st.dataframe(delisting_df)

#         with col2:
#             st.markdown("#### Brand Keep Level")
#             #AgGrid(keep_df,theme='alpine',height=160)
#             keep_df.columns = keep_df.columns.str.upper()
#             st.dataframe(keep_df)
            
#     st.sidebar.info('Brand Summary Generated', icon="🕤")
            
#      ########################################## Supplier Level ##############################       
#     # Filter by brand and calculate the desired values for Delisting
#     filtered_delisting_df1_s = response['data'][response['data']['SUPPLIER'] == selected_row_df['SUPPLIER'][0]]

#     filtered_delisting_df2_s = filtered_delisting_df1_s[filtered_delisting_df1_s['BUSINESS RECOMMENDATION']== 'Delist']

#     delisting_metrics_s = {
#         "Supplier Name": selected_row_df['SUPPLIER'][0],
#         "%SKU": round((len(filtered_delisting_df2_s) / len(filtered_delisting_df1_s) * 100),2),
#         "Absolute Count SKU": len(filtered_delisting_df2_s),
#         "Sales": (pd.to_numeric(filtered_delisting_df2_s['SALES'], errors='coerce')).sum(),
#         "Sales %": round(((pd.to_numeric(filtered_delisting_df2_s['SALES'], errors='coerce')).sum() / (pd.to_numeric(filtered_delisting_df1_s['SALES'], errors='coerce')).sum() * 100),2)
#     }
#     delisting_df_s = pd.DataFrame([delisting_metrics_s])


#     # Filter by brand and calculate the desired values for Keep
#     filtered_keep_df3_s = filtered_delisting_df1_s[filtered_delisting_df1_s['BUSINESS RECOMMENDATION'] == 'Keep']
#     keep_metrics_s = {
#         "Supplier Name": selected_row_df['SUPPLIER'][0],
#         "%SKU": round((len(filtered_keep_df3_s) / len(filtered_delisting_df1_s) * 100),2),
#         "Absolute Count SKU": len(filtered_keep_df3_s),
#         "Sales": (pd.to_numeric(filtered_keep_df3_s['SALES'], errors='coerce')).sum(),
#         "Sales %": round(((pd.to_numeric(filtered_keep_df3_s['SALES'], errors='coerce')).sum() / (pd.to_numeric(filtered_delisting_df1_s['SALES'], errors='coerce')).sum() * 100),2)
#     }
#     keep_df_s = pd.DataFrame([keep_metrics_s])



#     st.markdown('---')
#     st.markdown('### Supplier Level Summary')
# # # Create an expander named 'Brand & Supplier Summary'
#     my_expander = st.expander('Supplier Level Summary')

#     with my_expander:
#         col1, col2 = st.columns(2)
#         # Display 'Brand Level Summary' using AgGrid
#         with col1:
#             st.markdown("#### Supplier Delist Level")
#             #AgGrid(delisting_df,theme='alpine',height=160)
#             delisting_df_s.columns = delisting_df_s.columns.str.upper()
#             st.dataframe(delisting_df_s)

#         with col2:
#             st.markdown("#### Supplier Keep Level")
#             #AgGrid(keep_df,theme='alpine',height=160)
#             keep_df_s.columns = keep_df_s.columns.str.upper()
#             st.dataframe(keep_df_s)
            
#     st.sidebar.warning('Supplier Summary Generated', icon="🕥")
            
   
#     if selected_row_df['COUNTRY'][0] == 'UAE':
        
#         data_sub = data_sub_uae 
        
#     else: 
        
#         data_sub = data_sub_egp
        
#     st.markdown('---')
#     st.markdown('### Product Substitue')   
    
#     data_sub = data_sub[data_sub['ITEM NAME']== selected_row_df['ITEM NAME'][0]]
#     data_sub = data_sub.head(1)
    
#     # Create an expander named 'Product Substitute'
#     product_expander = st.expander('Product Substitute')
#     with product_expander:
#     # Display 'Product Substitute' using AgGrid
#     #AgGrid(product_df, theme='alpine',height=160)
                        
#         st.dataframe(data_sub)   
        
#     st.sidebar.success('Product Substitue Generated', icon="🕦")
    
    
# else:
    
    
    
    
    
#     if country_mapping[selected_country] == 'UAE':
#         df_uae1= response['data']
#         df_uae1['DELISTING LEVEL'] = df_uae1['DELISTING LEVEL'].replace('Current', 'Conservative')
        
#         # DataFrame1
#         df1 = df_uae1.groupby(['BRAND', 'BUSINESS RECOMMENDATION']).agg(
#             sku_count=('ITEM NAME', 'count'),
#             sales_total=('SALES', 'sum')
#         ).reset_index()
        
#         # DataFrame2
#         df2 = df_uae1.groupby(['SUPPLIER', 'BUSINESS RECOMMENDATION']).agg(
#             sku_count=('ITEM NAME', 'count'),
#             sales_total=('SALES', 'sum')
#         ).reset_index()
        
        
#         # Calculate percentages
#         for dataframe in [df1, df2]:
#             total_skus = dataframe.groupby(dataframe.columns[0]).sku_count.transform('sum')
#             total_sales = dataframe.groupby(dataframe.columns[0]).sales_total.transform('sum')
    
#             dataframe['sku_percent'] = ((dataframe['sku_count'] / total_skus) * 100).round(2)
#             dataframe['sales_percent'] = ((dataframe['sales_total'] / total_sales) * 100).round(2)

#         # Pivot dataframes to get desired structure
#         df1 = df1.pivot(index='BRAND', columns='BUSINESS RECOMMENDATION').reset_index()
#         df2 = df2.pivot(index='SUPPLIER', columns='BUSINESS RECOMMENDATION').reset_index()

#         # Flatten MultiIndex columns
#         df1.columns = [f'{i}_{j}'.upper() for i, j in df1.columns]
#         df2.columns = [f'{i}_{j}'.upper() for i, j in df2.columns]

#         # Rename columns
#         rename_dict = {
#             '_DELIST': 'DELIST_SKU_COUNT',
#             '_KEEP': 'KEEP_SKU_COUNT',
#             'SKU_COUNT_DELIST': 'DELIST SKU COUNT',
#             'SKU_COUNT_KEEP': 'KEEP SKU COUNT',
#             'SALES_TOTAL_DELIST': 'DELIST_SALES',
#             'SALES_TOTAL_KEEP': 'KEEP_SALES',
#             'SALES_PERCENT_DELIST': 'DELIST_SALES_PERCENT',
#             'SALES_PERCENT_KEEP': 'KEEP_SALES_PERCENT'
#         }

#         delisting_df = df1.rename(columns=rename_dict)
#         delisting_df_s = df2.rename(columns=rename_dict)
        
        
#         # Fill NaN with 0
#         delisting_df = delisting_df.fillna(0)
#         delisting_df_s = delisting_df_s.fillna(0)

#         # Rename columns
#         delisting_df = delisting_df.rename(columns={'BRAND_': 'BRAND'})
#         delisting_df_s = delisting_df_s.rename(columns={'SUPPLIER_': 'SUPPLIER'})

        
        
        

        
# #         df_uae = pd.read_csv('brand_supplier_base_file.csv')
# #         df_uae['DELISTING LEVEL'] = df_uae['DELISTING LEVEL'].replace('Current', 'Conservative')
        
        
# #         for col, value in filters.items():  # apply the same filters to df2
# #             df_uae = df_uae[df_uae[col] == value]
            
# #         #st.dataframe(df_uae)
            
            
            
# #         grouped_df_b = df_uae.groupby('BRAND').agg({
# #                 'Delist sku count': 'sum',
# #                 'Delist sales': 'sum',
# #                 'Keep sku count': 'sum',
# #                 'Keep sales': 'sum'
# #             }).reset_index()
            
# #         delisting_df = uae_compute_kpis(grouped_df_b)
# #         delisting_df =delisting_df[['BRAND','Delist sku percent','Delist sku count','Delist Sales percent','Delist sales','Keep sku percent','Keep sku count','Keep Sales percent', 'Keep sales']]
        
        
            
# #         grouped_df_s = df_uae.groupby('SUPPLIER').agg({
# #                 'Delist sku count': 'sum',
# #                 'Delist sales': 'sum',
# #                 'Keep sku count': 'sum',
# #                 'Keep sales': 'sum'
# #             }).reset_index()
            
#         delisting_df_s = uae_compute_kpis(grouped_df_s)
#         delisting_df_s =delisting_df_s[['SUPPLIER','Delist sku percent','Delist sku count','Delist Sales percent','Delist sales','Keep sku percent','Keep sku count','Keep Sales percent', 'Keep sales']]
        
        
        
        
#     else: 
#         df_eg = pd.read_csv('eg_supp_band.csv')
#         for col, value in filters.items():  # apply the same filters to df2
#             df_eg = df_eg[df_eg[col] == value]
            
#         # Group by supplier
#         supplier_grouped_df = df_eg.groupby('SUPPLIER').agg({
#             'Total items': 'sum',
#             'Total Sales': 'sum',
#             'Delist SKU count': 'sum',
#             'Delist Sales': 'sum',
#             'Review service level or delist SKU count': 'sum',
#             'Review service level or Delist Sales': 'sum',
#         }).reset_index()
        
#         # Compute KPIs for supplier
#         delisting_df_s = eg_compute_kpis(supplier_grouped_df)
        
#         # Group by brand
#         brand_grouped_df = df_eg.groupby('BRAND').agg({
#             'Total items': 'sum',
#             'Total Sales': 'sum',
#             'Delist SKU count': 'sum',
#             'Delist Sales': 'sum',
#             'Review service level or delist SKU count': 'sum',
#             'Review service level or Delist Sales': 'sum',
#         }).reset_index()

#         # Compute KPIs for brand
#         delisting_df = eg_compute_kpis(brand_grouped_df)
        
        
#     #keep_df = pd.read_excel('KEEP_SUMMARY_UAE.xlsx')
#     st.markdown('---')
#     st.markdown('### Brand Level Summary')
#     # # Create an expander named 'Brand & Supplier Summary'
#     my_expander = st.expander('Brand Level Summary')
#     with my_expander:
#         delisting_df.columns = delisting_df.columns.str.upper()
#         st.dataframe(delisting_df)
        
# #         col1, col2, col3,col4, col5, col6, col7, col88 = st.columns(8)

# #         # Leave the first column empty

# #         # Add a button to the second column
# #         col88.button('Download',key='DownloadButton2')
        
        
        
#     st.markdown('---')
#     st.markdown('### Supplier Level Summary')
#     # # Create an expander named 'Brand & Supplier Summary'
#     my_expander = st.expander('Supplier Level Summary')
#     with my_expander:
#         delisting_df_s.columns = delisting_df_s.columns.str.upper()
#         st.dataframe(delisting_df_s)
#         # Create three columns
# #         col1, col2, col3,col4, col5, col6, col7, col888 = st.columns(8)

# #         # Leave the first column empty

# #         # Add a button to the second column
# #         col888.button('Download',key='DownloadButton3')
        
        
       
#     st.markdown('---')
#     st.markdown('### SKU Substitues/Alternatives')   
        
        
#     if country_mapping[selected_country] == 'UAE':
        
#         data_sub = data_sub_uae
#         data_sub['DELISTING LEVEL'] = data_sub['DELISTING LEVEL'].replace('Current', 'Conservative')
        
#     else: 
        
#         data_sub = data_sub_egp
        
        
#     for col, value in filters.items():  # apply the same filters to df2
#             data_sub = data_sub[data_sub[col] == value]
            
#     data_sub = data_sub.iloc[:, :-8]

            

    
#     # Create an expander named 'Product Substitute'
#     product_expander = st.expander('Product Substitute')
#     with product_expander:
#     # Display 'Product Substitute' using AgGrid
#     #AgGrid(product_df, theme='alpine',height=160)
                        
#         st.dataframe(data_sub)  
        
# #         col1, col2, col3,col4, col5, col6, col7, col88 = st.columns(8)

# #         # Leave the first column empty

# #         # Add a button to the second column
# #         col88.button('Download',key='DownloadButton4')
# #         col1, col2, col3,col4, col5, col6, col7, col8 = st.columns(8)

# #         # Leave the first column empty

# #         # Add a button to the second column
# #         col8.button('Download')
        
# st.markdown("")      
# st.markdown("")  
# st.markdown("")  
# # st.button("Update Master File")
# # st.markdown("""
# #     <style>
# #     .stButton>button {
# #         background-color: #4CAF50;
# #         color: white;
# #     }
# #     </style>
# #     """, unsafe_allow_html=True)
