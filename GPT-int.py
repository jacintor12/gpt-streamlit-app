import json
import pandas as pd
import streamlit as st
import smartsheet
import docx
import PyPDF2
from openai import OpenAI
import re
from bs4 import BeautifulSoup
import markdown
import plotly.express as px

# -------------------------------
# App & Clients Setup (single place)
# -------------------------------
st.set_page_config(page_title="Smartsheet GPT Analyst", layout="wide")
st.title("Smartsheet GPT Analyst")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SMARTSHEET_TOKEN = st.secrets["SMARTSHEET_ACCESS_TOKEN"]

client = OpenAI(api_key=OPENAI_API_KEY)
ss = smartsheet.Smartsheet(SMARTSHEET_TOKEN)
ss.errors_as_exceptions(True)
ss._session.headers["Smartsheet-Change-Agent"] = "psi-gpt-analytics/1.0"

# -------------------------------
# Helpers
# -------------------------------
def smartsheet_to_df(sheet_id: int, page_size: int = 5000) -> pd.DataFrame:
    """Fetch a Smartsheet and return a lightly-typed DataFrame."""
    sheet = ss.Sheets.get_sheet(sheet_id, page_size=page_size, include="columns")
    colmap = {c.id: c.title for c in sheet.columns}
    rows = []
    for r in sheet.rows:
        row = {
            (colmap.get(c.column_id, str(c.column_id))): (
                c.display_value if c.display_value is not None else c.value
            )
            for c in r.cells
        }
        row["_rowId"] = r.id
        rows.append(row)
    df = pd.DataFrame(rows)
    # light typing
    for c in df.columns:
        if c.startswith("_"):
            continue
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
        try:
            df[c] = pd.to_datetime(df[c], format="%m/%d/%y")
        except Exception:
            pass
    return df

# -------------------------------
# AI Graph Generation
# -------------------------------
def generate_ai_graph_code(df: pd.DataFrame, user_prompt: str) -> str:
    """
    Generate Plotly code based on user prompt and data schema using GPT-4
    
    Args:
        df: DataFrame to analyze
        user_prompt: User's natural language request for the graph
    
    Returns:
        Generated Python/Plotly code as string
    """
    if df is None or df.empty:
        return "# Error: No data available"
    
    # Get data schema and sample
    schema = []
    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "unique_values": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(3).tolist()
        }
        schema.append(col_info)
    
    # Get first few rows as sample
    sample_data = df.head(5).to_dict(orient="records")
    
    # Create the system prompt for GPT
    system_prompt = r"""You are an expert Python developer specializing in Plotly Express visualizations.
    Generate Python code that creates interactive charts using plotly.express (imported as px).
    
    REQUIREMENTS:
    1. Use ONLY the provided column names from the schema - no other columns exist
    2. Handle data type conversions safely (check if conversion is needed before applying)
    3. The DataFrame is available as 'df'
    4. Import plotly.express as px is already done
    5. Always assign the final chart to a variable named 'fig'
    6. Never use fig.show() or return statements
    7. Use appropriate chart types for the data
    8. Add proper titles and labels
    
    SAFE DATA CONVERSION PATTERN:
    ```python
    # Safe percentage conversion with error handling
    def safe_convert_percentage(series):
        try:
            # Handle empty/null values
            series = series.fillna('0%')
            # Convert percentage strings to float
            return series.str.rstrip('%').astype('float') / 100.0
        except:
            # If conversion fails, try to extract numbers only
            import re
            return series.str.extract(r'(\d+(?:\.\d+)?)').astype('float') / 100.0
    
    # Check if column contains percentage strings before converting
    if df['column_name'].dtype == 'object' and df['column_name'].str.contains('%', na=False).any():
        df['column_name'] = safe_convert_percentage(df['column_name'])
    
    # For numeric columns that might be strings
    if df['column_name'].dtype == 'object':
        df['column_name'] = pd.to_numeric(df['column_name'], errors='coerce')
    ```
    
    EXAMPLE CODE STRUCTURE:
    ```python
    import plotly.express as px
    
    # Safe data preparation with type checking
    # ... data preprocessing ...
    
    fig = px.bar(  # or other chart type
        df,
        x="actual_column_name",  # Must be from schema
        y="actual_column_name",  # Must be from schema
        color="actual_column_name"  # Must be from schema or omit
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'total descending'},
        showlegend=True
    )
    
    # The fig variable will be automatically displayed in Streamlit
    ```

    CRITICAL: Always end with the 'fig' variable assigned, never use fig.show() or return statements.
    Only return the Python code, no explanations."""
    
    # Create the user message with data context
    user_message = f"""
    User Request: {user_prompt}

    AVAILABLE DATA COLUMNS (ONLY USE THESE):
    {[col['name'] for col in schema]}

    DETAILED DATA SCHEMA:
    {schema}

    SAMPLE DATA (first 5 rows):
    {sample_data}

    IMPORTANT: You must ONLY use column names from the available columns list above. 
    If the user asks for a column that doesn't exist, use the most appropriate available column instead.

    Generate appropriate Plotly Express code for this visualization request using ONLY the available columns.
    """
    
    try:
        # Call GPT-4 to generate the code
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        
        generated_code = response.choices[0].message.content
        
        # Clean up the code (remove markdown code blocks if present)
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0]
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0]
        
        return generated_code.strip()
        
    except Exception as e:
        return f"# Error generating code: {str(e)}"

def execute_ai_graph_code(code: str, df: pd.DataFrame):
    """
    Safely execute the generated Plotly code and return the figure
    
    Args:
        code: Generated Python code
        df: DataFrame to use in execution
    
    Returns:
        Plotly figure object or None if error
    """
    try:
        # Clean the code first
        cleaned_code = code.strip()
        
        # Remove any markdown code blocks
        if "```python" in cleaned_code:
            cleaned_code = cleaned_code.split("```python")[1].split("```")[0]
        elif "```" in cleaned_code:
            cleaned_code = cleaned_code.split("```")[1].split("```")[0]
        
        cleaned_code = cleaned_code.strip()
        
        # Create execution environment with a copy of the DataFrame to avoid modification issues
        df_copy = df.copy()
        exec_globals = {
            '__builtins__': __builtins__,
            'px': px,
            'pd': pd,
            'df': df_copy,
            'fig': None
        }
        
        # Execute the code
        exec(cleaned_code, exec_globals)
        
        # Get the figure
        fig = exec_globals.get('fig', None)
        return fig
        
    except Exception as e:
        st.error(f"Error executing generated code: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# -------------------------------
# CSV Export Utility
# -------------------------------
def safe_download_button(df: pd.DataFrame, filename: str):
    if df is not None and not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {filename}",
            data=csv,
            file_name=filename,
            mime='text/csv'
        )

# -------------------------------
# Sidebar Configuration (single source of truth)
# -------------------------------
st.sidebar.header("Configuration")
config_file = "config.json"
default_config = {
    "sheet_id": "",
    "sheet_url": "",
    "custom_prompt": (
        "You are a data analyst. Be concise, bullet key points, include simple "
        "calculations, flag data quality issues, and suggest next actions. Avoid jargon."
    ),
}

try:
    with open(config_file, "r") as f:
        config = json.load(f)
except Exception:
    config = default_config.copy()

sheet_id = st.sidebar.text_input("Smartsheet Sheet ID", value=config.get("sheet_id", ""))
custom_prompt = st.sidebar.text_area(
    "Custom Prompt",
    value=config.get("custom_prompt", default_config["custom_prompt"]),
    height=100,
)

if st.sidebar.button("Save Configuration"):
    cfg = {"sheet_id": sheet_id, "custom_prompt": custom_prompt}
    with open(config_file, "w") as f:
        json.dump(cfg, f)
    st.sidebar.success("Configuration saved!")

# -------------------------------
# Smartsheet Fetch & Preview
# -------------------------------
st.header("Smartsheet Sheet Data")
if "sheet_data" not in st.session_state:
    st.session_state["sheet_data"] = None

if st.button("Fetch Smartsheet Sheet"):
    try:
        if sheet_id:
            st.session_state["sheet_data"] = smartsheet_to_df(int(sheet_id))
            st.success("Smartsheet data loaded!")
            st.write("Preview of Smartsheet data:")
            st.dataframe(st.session_state["sheet_data"].head(20))
        else:
            st.warning("Please enter a valid Smartsheet Sheet ID in the configuration tab.")
    except Exception as e:
        import traceback
        st.error(f"Error fetching Smartsheet data: {e}\n{traceback.format_exc()}")

# -------------------------------
# Question Input
# -------------------------------
st.write("Welcome! Your app is running.")
question = st.text_area("Ask a question about the uploaded data or text:", height=100)

# Submit button directly below chatbox
submit_clicked = st.button("Submit")

# -------------------------------
# Single File Uploader (CSV/Excel/PDF/Word)
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload one or more CSV, Excel, PDF, or Word files to analyze",
    type=["csv", "xlsx", "xls", "pdf", "docx"],
    accept_multiple_files=True,
    key="main_file_uploader",
)

dataframes = []  # list[(filename, df)]
documents = []   # list[(filename, text)]

if uploaded_files:
    for uploaded_file in uploaded_files:
        filetype = uploaded_file.name.split(".")[-1].lower()
        try:
            if filetype == "csv":
                df = pd.read_csv(uploaded_file)
                dataframes.append((uploaded_file.name, df))
                st.success(f"CSV file '{uploaded_file.name}' loaded successfully!")
                st.write(f"Preview of '{uploaded_file.name}':")
                st.dataframe(df.head(20))

            elif filetype in ["xlsx", "xls"]:
                df = pd.read_excel(uploaded_file)
                dataframes.append((uploaded_file.name, df))
                st.success(f"Excel file '{uploaded_file.name}' loaded successfully!")
                st.write(f"Preview of '{uploaded_file.name}':")
                st.dataframe(df.head(20))

            elif filetype == "pdf":
                reader = PyPDF2.PdfReader(uploaded_file)
                text_content = "\n".join(page.extract_text() or "" for page in reader.pages)
                documents.append((uploaded_file.name, text_content))
                st.success(f"PDF file '{uploaded_file.name}' loaded successfully!")
                if st.button(f"Show preview of '{uploaded_file.name}'"):
                    st.text(text_content[:2000])

            elif filetype == "docx":
                doc = docx.Document(uploaded_file)
                text_content = "\n".join(para.text for para in doc.paragraphs)
                documents.append((uploaded_file.name, text_content))
                st.success(f"Word file '{uploaded_file.name}' loaded successfully!")
                if st.button(f"Show preview of '{uploaded_file.name}'"):
                    st.text(text_content[:2000])

        except Exception as e:
            st.error(f"Error loading file '{uploaded_file.name}': {e}")

# -------------------------------
# Submit -> Compose prompt & call OpenAI once
# -------------------------------
if submit_clicked:
    if question:
        with st.spinner("Getting GPT response..."):
            try:
                user_parts = []

                # Smartsheet data
                if st.session_state["sheet_data"] is not None:
                    sheet_data = st.session_state["sheet_data"]
                    schema = [
                        {"name": c, "dtype": str(sheet_data[c].dtype), "unique": int(sheet_data[c].nunique())}
                        for c in sheet_data.columns
                    ]
                    preview = sheet_data.head(30).to_dict(orient="records")
                    user_parts.append(
                        f"Smartsheet Data\nSchema: {schema}\nPreview (first 30 rows): {preview}"
                    )

                # Uploaded tabular files
                for fname, df in dataframes:
                    schema = [
                        {"name": c, "dtype": str(df[c].dtype), "unique": int(df[c].nunique())}
                        for c in df.columns
                    ]
                    preview = df.head(30).to_dict(orient="records")
                    user_parts.append(
                        f"File: {fname}\nSchema: {schema}\nPreview (first 30 rows): {preview}"
                    )

                # Uploaded documents
                for fname, text_content in documents:
                    user_parts.append(f"File: {fname}\nDocument text: {text_content[:8000]}")

                user_msg = (f"Question: {question}\n" + "\n\n".join(user_parts)) if user_parts else f"Question: {question}"
                system_msg = custom_prompt

                response = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                )
                answer = response.choices[0].message.content
                st.success("GPT Response:")
                st.write(answer)

                # Try to extract markdown table first
                table_match = re.search(r'(\|.+\|\n)+', answer)
                gpt_df = None
                if table_match:
                    table_text = table_match.group(0)
                    lines = [line for line in table_text.splitlines() if '|' in line]
                    if len(lines) >= 2:
                        columns = [col.strip() for col in lines[0].split('|')[1:-1]]
                        data_rows = [
                            [cell.strip() for cell in row.split('|')[1:-1]]
                            for row in lines[2:] if len(row.split('|')) == len(columns)+2
                        ]
                        gpt_df = pd.DataFrame(data_rows, columns=columns)
                
                # If markdown table not found, try HTML table
                if gpt_df is None:
                    html = markdown.markdown(answer)
                    soup = BeautifulSoup(html, 'html.parser')
                    table = soup.find("table")
                    if table:
                        rows = table.find_all("tr")
                        headers = [th.get_text(strip=True) for th in rows[0].find_all("th")]
                        data = [[td.get_text(strip=True) for td in row.find_all("td")] for row in rows[1:]]
                        gpt_df = pd.DataFrame(data, columns=headers)
                
                # Display and download if table found
                if gpt_df is not None and not gpt_df.empty:
                    st.dataframe(gpt_df)
                    safe_download_button(gpt_df, "gpt_table.csv")
                else:
                    st.info("No valid table found in GPT response.")

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question before submitting.")

# -------------------------------
# Tabs Section
# -------------------------------
tabs = st.tabs(["General Analysis", "Sheets Viewer", "Graphs"])

with tabs[0]:
    st.markdown(
        "[Open General Analysis](https://chatgpt.com/g/g-68be080896048191a93e8384b6a52f4b-monitoring-n-evaluation?model=gpt-4o)",
        unsafe_allow_html=True
    )

with tabs[1]:
    st.header("Smartsheet Sheet Viewer")

    # Load workspace_id from config.json if present and not in session_state
    if "workspace_id" not in st.session_state:
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            st.session_state["workspace_id"] = config.get("workspace_id", "")
        except Exception:
            st.session_state["workspace_id"] = ""

    workspace_id_input = st.text_input(
        "Enter Smartsheet Workspace ID",
        value=st.session_state["workspace_id"],
        key="workspace_id_input"
    )
    if st.button("Save Workspace ID", key="save_workspace_id"):
        st.session_state["workspace_id"] = workspace_id_input
        # Save to config.json
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
        except Exception:
            config = {}
        config["workspace_id"] = workspace_id_input
        with open(config_file, "w") as f:
            json.dump(config, f)

    # Auto-fetch sheets when workspace_id is set
    sheets_list = []
    if "workspace_id" in st.session_state and st.session_state["workspace_id"]:
        try:
            workspace = ss.Workspaces.get_workspace(st.session_state["workspace_id"], include="sheets")
            sheets_list = [(sheet.name, sheet.id) for sheet in workspace.sheets]
        except Exception as e:
            st.error(f"Error fetching sheets for workspace: {e}")

    if sheets_list:
        sheet_options = {name: sid for name, sid in sheets_list}
        selected_sheet_name = st.selectbox("Select a sheet to view", list(sheet_options.keys()), key="sheet_select")
        selected_sheet_id = sheet_options[selected_sheet_name]

        if st.button("Load Sheet Data", key="load_sheet_data"):
            try:
                sheet_df = smartsheet_to_df(int(selected_sheet_id))
                st.session_state["selected_sheet_df"] = sheet_df
                st.success(f"Fetched {len(sheet_df)} rows from sheet '{selected_sheet_name}' (ID: {selected_sheet_id})")
            except Exception as e:
                st.error(f"Error fetching sheet: {e}")

        # Show dataframe if already loaded
        if "selected_sheet_df" in st.session_state:
            st.dataframe(st.session_state["selected_sheet_df"])
            if st.button("Save for Analysis", key="save_for_analysis"):
                st.session_state["sheet_data"] = st.session_state["selected_sheet_df"]
                st.success("Sheet data saved for analysis!")

with tabs[2]:
    st.header("Interactive Graphs Generator")

    # Data source selection
    data_source = st.radio("Select data source:", ["Smartsheet Sheet", "Upload Spreadsheet"], key="graph_data_source")

    df = None
    # Option 1: Use fetched Smartsheet sheet data
    if data_source == "Smartsheet Sheet":
        if "selected_sheet_df" in st.session_state:
            df = st.session_state["selected_sheet_df"]
            st.success("Using data from selected Smartsheet sheet.")
        else:
            st.info("No Smartsheet sheet data loaded. Please load a sheet in the Sheets Viewer tab.")

    # Option 2: Upload spreadsheet
    if data_source == "Upload Spreadsheet":
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], key="graph_file_uploader")
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success(f"Loaded {len(df)} rows from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    # If we have a dataframe, show graph options
    if df is not None:
        st.dataframe(df)
        
        # Graph generation method selection
        st.subheader("Choose Graph Generation Method")
        generation_method = st.radio(
            "How would you like to create your graph?",
            ["AI-Powered Generation", "Manual Configuration"],
            key="generation_method"
        )
        
        if generation_method == "AI-Powered Generation":
            # AI-powered graph generation section
            st.markdown("### AI Graph Generation")
            st.markdown("Describe what kind of visualization you want in natural language:")
            
            # Example prompts
            with st.expander("📝 Example Prompts"):
                st.markdown("""
                **Examples of what you can ask:**
                - "Create a bar chart showing the planned versus actual performance indicators by quarter with different colors for each quarter"
                - "Generate an interactive line chart showing trends over time"
                - "Make a pie chart displaying the top five grantees based on performance indicators"
                - "Show a grouped bar chart comparing planned vs actual values by grantee"
                - "Create a scatter plot showing the relationship between planned versus actual objectives met by quarter"
                - "Generate a time series chart showing quarterly performance"
                """)
            
            # User prompt input
            user_prompt = st.text_area(
                "Describe your visualization:",
                placeholder="e.g., Create an interactive bar chart showing call duration by department with different colors for each call type",
                height=100,
                key="ai_graph_prompt"
            )
            
            if st.button("🚀 Generate AI Graph", key="generate_ai_graph"):
                if user_prompt.strip():
                    with st.spinner("🤖 AI is generating your graph code..."):
                        generated_code = generate_ai_graph_code(df, user_prompt)
                        
                        if "Error" not in generated_code:
                            st.success("✅ Code generated successfully!")
                            
                            # Show the generated code
                            with st.expander("📋 View Generated Code"):
                                st.code(generated_code, language="python")
                            
                            # Execute the code and show the graph
                            with st.spinner("📊 Creating your visualization..."):
                                fig = execute_ai_graph_code(generated_code, df)
                                
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.success("🎉 Graph generated successfully!")
                                else:
                                    st.error("❌ Failed to execute the generated code.")
                        else:
                            st.error(f"❌ {generated_code}")
                else:
                    st.warning("⚠️ Please enter a description for your visualization.")
        
        else:
            # Manual configuration section - simplified for basic charts
            st.markdown("### Manual Chart Configuration")
            st.info("💡 For advanced visualizations, try the AI-Powered Generation option above!")
            
            # Basic chart configuration
            chart_type = st.selectbox("Select chart type", ["Bar", "Pie", "Line"], key="graph_chart_type")
            columns = df.columns.tolist()
            
            # Add a simple test button
            if st.button("🧪 Test Simple Chart", key="test_simple_chart"):
                try:
                    # Create a simple bar chart with the first two suitable columns
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if len(numeric_cols) >= 1:
                        # Try to find a good categorical column for x-axis
                        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                        
                        if categorical_cols and numeric_cols:
                            x_col = categorical_cols[0]  # First categorical column
                            y_col = numeric_cols[0]      # First numeric column
                            
                            st.write(f"**Testing with:** X={x_col}, Y={y_col}")
                            
                            # Create simple bar chart
                            fig = px.bar(df.head(10), x=x_col, y=y_col, 
                                       title=f"Test Chart: {y_col} by {x_col}")
                            st.plotly_chart(fig, use_container_width=True)
                            st.success("✅ Simple test chart created successfully!")
                        else:
                            st.warning("No suitable columns found for chart")
                    else:
                        st.warning("No numeric columns found")
                except Exception as e:
                    st.error(f"Test chart error: {str(e)}")
            
            # Add a percentage data test button
            if st.button("📊 Test Percentage Chart", key="test_percentage_chart"):
                try:
                    # Create a chart specifically for the percentage data
                    df_test = df.head(10).copy()  # Work with first 10 rows
                    
                    # Convert percentage columns safely
                    percentage_cols = ['Planned Indicator', 'Q1 Actual', 'Q2 Actual', 'Q3 Actual', 'Q4 Actual']
                    
                    # Check which columns exist
                    available_cols = [col for col in percentage_cols if col in df_test.columns]
                    st.write(f"**Available percentage columns:** {available_cols}")
                    
                    if available_cols:
                        # Convert percentage strings to numbers
                        for col in available_cols:
                            try:
                                # Handle NaN and empty values
                                df_test[col] = df_test[col].fillna('0%')
                                # Remove % and convert to float
                                df_test[col] = df_test[col].str.rstrip('%').astype('float') / 100.0
                                st.write(f"**Converted {col}:** {df_test[col].tolist()}")
                            except Exception as e:
                                st.error(f"Error converting {col}: {e}")
                        
                        # Create a simple bar chart
                        if 'Performance Indicator' in df_test.columns and 'Planned Indicator' in available_cols:
                            fig = px.bar(df_test, 
                                       x='Performance Indicator', 
                                       y='Planned Indicator',
                                       title='Test: Planned Indicators by Performance Indicator')
                            st.plotly_chart(fig, use_container_width=True)
                            st.success("✅ Percentage test chart created successfully!")
                        else:
                            st.warning("Required columns not found for percentage chart")
                    else:
                        st.warning("No percentage columns found")
                        
                except Exception as e:
                    st.error(f"Percentage chart error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # Regular chart configuration
            
            if chart_type == "Pie":
                values_col = st.selectbox("Values column", columns, key="pie_values_col")
                names_col = st.selectbox("Names column", columns, key="pie_names_col")
                
                fig = px.pie(df, values=values_col, names=names_col, 
                           title=f"Pie Chart: {values_col} by {names_col}")
                
            else:  # Bar or Line charts
                x_col = st.selectbox("X-axis column", columns, key="graph_x_col")
                y_col = st.selectbox("Y-axis column", columns, key="graph_y_col")
                
                if chart_type == "Bar":
                    fig = px.bar(df, x=x_col, y=y_col, 
                               title=f"Bar Chart: {y_col} vs {x_col}")
                else:  # Line chart
                    fig = px.line(df, x=x_col, y=y_col, 
                                title=f"Line Chart: {y_col} vs {x_col}")
            
            # Display the chart
            if 'fig' in locals():
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select columns to generate a chart.")

# -------------------------------
# End of file
# -------------------------------

