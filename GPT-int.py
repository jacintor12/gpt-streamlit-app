import json
import pandas as pd
import streamlit as st
import smartsheet
import docx
import PyPDF2
from openai import OpenAI

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

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question before submitting.")

# -------------------------------
# End of file
# -------------------------------

