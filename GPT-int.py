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

# -------------------------------
# App & Clients Setup
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

            # Try to extract markdown table
            # Try to extract markdown table first
            import re
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
                import markdown
                from bs4 import BeautifulSoup
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

