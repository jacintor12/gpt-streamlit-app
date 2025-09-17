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

st.set_page_config(page_title="Smartsheet GPT Analyst", layout="wide")
st.title("Smartsheet GPT Analyst")

# Inject custom CSS for larger font size for non-title text
st.markdown(
    """
    <style>
    .stTextInput label, .stTextArea label, .stMarkdown, .stDataFrameContainer, .stButton button, .stFileUploader label {
        font-size: 14pt !important;
    }
    .stTextInput input, .stTextArea textarea, .stFileUploader input {
        font-size: 14pt !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
    system_prompt = """You are an expert data visualization developer specializing in Plotly Express for Streamlit applications. 
    Generate Python code using plotly.express to create interactive visualizations based on user requests.

    IMPORTANT REQUIREMENTS:
    1. Always import plotly.express as px at the top
    2. The DataFrame is already available as 'df'
    3. Generate complete, executable code that works in Streamlit
    4. Use appropriate chart types based on data and request
    5. Include proper titles, labels, and hover data
    6. Add layout customizations when beneficial
    7. ALWAYS assign the figure to variable 'fig' (do NOT use fig.show())
    8. Handle missing data appropriately
    9. Use appropriate color schemes and styling
    10. For grouped/comparative charts, use appropriate grouping techniques
    11. Ensure all column names referenced actually exist in the data

    EXAMPLE OUTPUT FORMAT:
    ```python
    import plotly.express as px

    # Your descriptive comment here
    fig = px.bar(
        df,
        x="column_name",
        y="value_column",
        title="Descriptive Title",
        labels={"x": "X Label", "y": "Y Label"},
        color="category_column"
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

    Data Schema:
    {schema}

    Sample Data (first 5 rows):
    {sample_data}

    Generate appropriate Plotly Express code for this visualization request.
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
        # Create a safe execution environment
        exec_globals = {
            'px': px,
            'pd': pd,
            'df': df,
            'fig': None
        }
        
        # Remove fig.show() calls and replace with storing the figure
        modified_code = code.replace('fig.show()', '# fig stored for return')
        
        # Execute the code
        exec(modified_code, exec_globals)
        
        # Return the figure if it was created
        fig = exec_globals.get('fig', None)
        return fig
        
    except Exception as e:
        st.error(f"Error executing generated code: {str(e)}")
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
# Sidebar Configuration
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
custom_prompt = st.sidebar.text_area("Custom Prompt", value=config.get("custom_prompt", default_config["custom_prompt"]), height=100)

if st.sidebar.button("Save Configuration"):
    cfg = {"sheet_id": sheet_id, "custom_prompt": custom_prompt}
    with open(config_file, "w") as f:
        json.dump(cfg, f)
    st.sidebar.success("Configuration saved!")

# -------------------------------
# Smartsheet Fetch
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
            st.warning("Please enter a valid Smartsheet Sheet ID.")
    except Exception as e:
        st.error(f"Error fetching Smartsheet data: {e}")

# -------------------------------
# File Uploads
# -------------------------------
st.write("Upload CSV, Excel, PDF, or Word files to analyze:")
uploaded_files = st.file_uploader("Upload files", type=["csv", "xlsx", "xls", "pdf", "docx"], accept_multiple_files=True)

dataframes = []
documents = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        filetype = uploaded_file.name.split(".")[-1].lower()
        try:
            if filetype == "csv":
                df = pd.read_csv(uploaded_file)
                dataframes.append((uploaded_file.name, df))
                st.success(f"{uploaded_file.name} loaded.")
                st.dataframe(df.head(20))

            elif filetype in ["xlsx", "xls"]:
                df = pd.read_excel(uploaded_file)
                dataframes.append((uploaded_file.name, df))
                st.success(f"{uploaded_file.name} loaded.")
                st.dataframe(df.head(20))

            elif filetype == "pdf":
                reader = PyPDF2.PdfReader(uploaded_file)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                documents.append((uploaded_file.name, text))
                st.success(f"{uploaded_file.name} loaded.")
                if st.button(f"Show {uploaded_file.name}"):
                    st.text(text[:2000])

            elif filetype == "docx":
                doc = docx.Document(uploaded_file)
                text = "\n".join(p.text for p in doc.paragraphs)
                documents.append((uploaded_file.name, text))
                st.success(f"{uploaded_file.name} loaded.")
                if st.button(f"Show {uploaded_file.name}"):
                    st.text(text[:2000])

        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")

# -------------------------------
# Ask a Question to GPT
# -------------------------------
question = st.text_area("Ask a question about the data:")
submit_clicked = st.button("Submit")

if submit_clicked and question:
    with st.spinner("Getting GPT response..."):
        try:
            user_parts = []

            if st.session_state["sheet_data"] is not None:
                sheet_data = st.session_state["sheet_data"]
                schema = [{"name": c, "dtype": str(sheet_data[c].dtype), "unique": int(sheet_data[c].nunique())} for c in sheet_data.columns]
                preview = sheet_data.head(30).to_dict(orient="records")
                user_parts.append(f"Smartsheet Data\nSchema: {schema}\nPreview: {preview}")

            for fname, df in dataframes:
                schema = [{"name": c, "dtype": str(df[c].dtype), "unique": int(df[c].nunique())} for c in df.columns]
                preview = df.head(30).to_dict(orient="records")
                user_parts.append(f"File: {fname}\nSchema: {schema}\nPreview: {preview}")

            for fname, text in documents:
                user_parts.append(f"File: {fname}\nDocument text: {text[:8000]}")

            user_msg = f"Question: {question}\n\n" + "\n\n".join(user_parts)
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
elif submit_clicked:
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
            workspace = ss.Workspaces.get_workspace(st.session_state["workspace_id"])
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
            ["🤖 AI-Powered Generation", "🔧 Manual Configuration"],
            key="generation_method"
        )
        
        if generation_method == "🤖 AI-Powered Generation":
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
