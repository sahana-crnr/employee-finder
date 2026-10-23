import os
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from src.agent.employee_agent import create_employee_agent, run_agent_stream
from src.config import get_llm
from src.tools.search_tools import classify_seniority, parse_company_and_domain


st.set_page_config(
    page_title="Employee Finder",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --navy-950: #0b1f3a;
            --navy-900: #102a4c;
            --navy-800: #173b66;
            --navy-700: #244e7a;
            --navy-600: #35638f;
            --navy-500: #587ba1;
            --navy-400: #8ba2bf;
            --navy-300: #b7c7dc;
            --navy-200: #d7e1ef;
            --navy-100: #e8eef6;
            --navy-50: #f3f6fb;
            --page: var(--navy-50);
            --ink: var(--navy-950);
            --muted: var(--navy-600);
            --lift: 7px 7px 16px #d5deea, -7px -7px 16px #fbfcfe;
            --lift-soft: 4px 4px 10px #d9e2ed, -4px -4px 10px #fbfcfe;
        }
        html, body, [data-testid="stAppViewContainer"] { color: var(--ink); }
        .stApp { background: var(--page); color: var(--ink); }
        .stApp a { color: var(--navy-700); }
        .stApp a:hover { color: var(--navy-950); }
        .stApp input { accent-color: var(--navy-700); }
        [data-testid="stHeader"] { background: rgba(243, 246, 251, .9); }
        .block-container { max-width: 1180px; padding-top: 2.5rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: var(--ink); letter-spacing: -.025em; }
        .page-intro { margin: 0 0 1.8rem; }
        .page-intro h1 { margin-bottom: .45rem; }
        .page-intro p { color: var(--muted); font-size: 1.02rem; max-width: 720px; line-height: 1.6; }

        /* Soft, low-contrast clay relief for the main search surface. */
        div[data-testid="stForm"] {
            background: linear-gradient(145deg, #f8fafd, var(--navy-100));
            border: 1px solid #fbfcfe;
            border-radius: 18px;
            padding: 1.35rem 1.5rem;
            box-shadow: var(--lift);
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stPasswordInput"] input,
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
        div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
            background: var(--navy-50);
            border: 1px solid var(--navy-200);
            border-radius: 10px;
            box-shadow: inset 2px 2px 5px #dce5ef, inset -2px -2px 5px #fbfcfe;
        }
        div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
            color: var(--navy-900);
            background: var(--navy-200);
        }
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stPasswordInput"] input:focus {
            border-color: var(--navy-500);
            box-shadow: 0 0 0 3px rgba(53, 99, 143, .16), inset 1px 1px 3px #dce5ef;
        }
        div.stButton > button,
        div[data-testid="stFormSubmitButton"] button,
        div.stDownloadButton > button {
            color: var(--navy-900);
            border: 1px solid var(--navy-200);
            background: linear-gradient(145deg, var(--navy-100), var(--navy-200));
            border-radius: 10px;
            font-weight: 600;
            min-height: 2.7rem;
            box-shadow: var(--lift-soft);
            transition: transform .16s ease, box-shadow .16s ease, background .16s ease;
        }
        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover,
        div.stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 5px 5px 12px #d1dce9, -5px -5px 12px #fbfcfe;
        }
        div.stButton > button:active,
        div[data-testid="stFormSubmitButton"] button:active,
        div.stDownloadButton > button:active {
            transform: translateY(0);
            box-shadow: inset 2px 2px 5px rgba(11, 31, 58, .18);
        }
        div.stButton > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] button[kind="primary"] {
            color: #f8fafd;
            background: linear-gradient(145deg, var(--navy-700), var(--navy-900));
            border-color: var(--navy-800);
            box-shadow: 4px 4px 10px #cbd7e6, -3px -3px 9px #fbfcfe;
        }
        div.stButton > button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
            background: linear-gradient(145deg, var(--navy-600), var(--navy-800));
        }
        div[data-testid="stFormSubmitButton"] button:focus,
        div[data-testid="stFormSubmitButton"] button:focus-visible,
        div.stButton > button:focus-visible,
        div.stDownloadButton > button:focus-visible {
            outline: 3px solid rgba(53, 99, 143, .28) !important;
            outline-offset: 2px;
            border-color: var(--navy-600) !important;
            box-shadow: 0 0 0 2px var(--navy-100), var(--lift-soft) !important;
        }

        div[data-testid="stSlider"] [role="slider"] {
            background-color: var(--navy-700) !important;
            border-color: var(--navy-700) !important;
        }
        div[data-testid="stSlider"] [data-testid="stThumbValue"] {
            color: var(--navy-800) !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
            background-color: var(--navy-300);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(160deg, #f8fafd, var(--navy-100));
            border-right: 1px solid var(--navy-200);
            box-shadow: 4px 0 14px rgba(16, 42, 76, .06);
        }
        section[data-testid="stSidebar"] h2 { font-size: 1.1rem; }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--navy-200);
            border-radius: 12px;
            box-shadow: var(--lift-soft);
            overflow: hidden;
        }
        [data-testid="stAlert"],
        [data-testid="stAlert"] > div,
        [data-testid="stAlert"] [role="alert"] {
            color: var(--navy-900);
            background: var(--navy-100) !important;
            border: 1px solid var(--navy-200);
            border-radius: 12px;
            box-shadow: var(--lift-soft);
        }
        [data-testid="stAlert"] * { color: var(--navy-800) !important; }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { transition-duration: .01ms !important; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.subheader("Search settings")
    provider = st.selectbox(
        "LLM provider",
        ["Ollama (Local)", "Groq (Cloud)", "OpenAI", "Google Gemini"],
    )
    provider_map = {
        "Ollama (Local)": "ollama",
        "Groq (Cloud)": "groq",
        "OpenAI": "openai",
        "Google Gemini": "gemini",
    }
    provider_key = provider_map[provider]
    model_options = {
        "ollama": ["llama3.2", "phi3", "qwen2.5", "mistral", "gemma2"],
        "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "openai": ["gpt-4o-mini", "gpt-4o", "o3-mini"],
        "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
    }
    model = st.selectbox("Model", model_options[provider_key])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    api_key = None
    if provider_key != "ollama":
        env_var = {
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GOOGLE_API_KEY",
        }[provider_key]
        env_val = os.getenv(env_var, "")
        if not env_val and hasattr(st, "secrets") and env_var in st.secrets:
            env_val = str(st.secrets[env_var])
        api_key = st.text_input(
            "API key",
            type="password",
            value=env_val,
            placeholder=f"Enter {env_var}",
        )
        if not api_key:
            st.caption(f"{env_var} is required to use this provider.")


st.markdown(
    """
    <div class="page-intro">
      <h1>Employee Finder</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("search_form"):
    company_input = st.text_input(
        "Company name or website",
        placeholder="e.g. Company or company.com",
        help="Enter a company name or its website URL.",
    )
    search_clicked = st.form_submit_button(
        "Search employees", type="primary", use_container_width=True
    )


if search_clicked:
    # Discard results from previous app versions or earlier searches immediately.
    st.session_state.pop("last_result", None)
    st.session_state.pop("result_version", None)
    if not company_input.strip():
        st.warning("Enter a company name or website to start a search.")
    else:
        company_name, company_domain = parse_company_and_domain(company_input)
        try:
            llm = get_llm(
                provider=provider_key,
                model_name=model,
                temperature=temperature,
                api_key=api_key,
            )
            agent = create_employee_agent(llm_instance=llm)
        except Exception as err:
            st.error(f"Could not initialize {provider}: {err}")
            st.stop()

        search_title = f"Searching {company_name}"
        if company_domain:
            search_title += f" ({company_domain})"
        st.subheader(search_title)
        progress = st.empty()
        progress.info("Searching company and professional directory sources…")

        result = None
        try:
            for event in run_agent_stream(
                company_or_url=company_input,
                role_preference=None,
                agent_instance=agent,
            ):
                if event["type"] == "action":
                    if event["tool"] == "verify_company_google":
                        progress.info("Verifying the company through Google…")
                    elif event["tool"] == "verify_company_instagram":
                        progress.info("Checking Instagram for an official company profile…")
                    else:
                        progress.info("Company verification complete. Searching LinkedIn for current leaders…")
                elif event["type"] == "result":
                    result = event.get("data", {})
                    progress.success("Search complete.")
        except Exception as err:
            progress.error(f"Search failed: {err}")
            st.stop()

        st.session_state["last_result"] = result
        st.session_state["company_searched"] = company_name
        st.session_state["result_version"] = 7


if st.session_state.get("result_version") == 7 and st.session_state.get("last_result"):
    result_data = st.session_state["last_result"]
    employees = result_data.get("employees", [])
    company_name = st.session_state.get("company_searched", "Company")
    st.divider()
    st.subheader(f"Results for {company_name}")

    if not result_data.get("company_verified"):
        st.warning("Could not confirm this exact company through Google or an Instagram company profile, so no member search was run. Check the company spelling.")
        if result_data.get("raw_output"):
            st.caption(result_data["raw_output"])
    elif not employees:
        st.info("The company was verified. LinkedIn search did not return indexed profiles with enough evidence of a current senior role at this company; this does not mean the company has no members.")
        if result_data.get("company_page_url"):
            source_label = result_data.get("company_verification_source", "company verification source")
            st.markdown(f"[Open the {source_label} company result]({result_data['company_page_url']})")
        raw_output = result_data.get("raw_output")
        if raw_output:
            with st.expander("Search notes"):
                st.code(raw_output)
    else:
        records = []
        for employee in employees:
            name = (employee.get("name") or "").strip()
            role = (employee.get("role") or employee.get("job_title") or "").strip()
            seniority = employee.get("seniority") or classify_seniority(role)
            profile_url = (employee.get("source_url") or "").strip()
            source = employee.get("source_domain", "")
            if not source and profile_url:
                source = urlparse(profile_url).netloc.removeprefix("www.")
            records.append(
                {
                    "Name": name,
                    "Role": role,
                    "Seniority": seniority,
                    "Employment": " · ".join(
                        value for value in (
                            employee.get("current_employment_status", "Presently working"),
                            employee.get("employment_timeline", "Present"),
                        ) if value
                    ),
                    "Source": source or "Web",
                    "Current employment evidence": employee.get("source_snippet", ""),
                    "Profile": profile_url,
                }
            )

        df = pd.DataFrame(records)
        seniority_options = sorted(df["Seniority"].dropna().unique())
        selected_seniority = st.multiselect(
            "Filter by seniority", seniority_options, default=seniority_options
        )
        filtered_df = df[df["Seniority"].isin(selected_seniority)]
        st.dataframe(
            filtered_df,
            column_config={
                "Name": st.column_config.TextColumn("Name", width="medium"),
                "Role": st.column_config.TextColumn("Role", width="large"),
                "Seniority": st.column_config.TextColumn("Seniority", width="medium"),
                "Employment": st.column_config.TextColumn("Current employment / timeline", width="medium"),
                "Source": st.column_config.TextColumn("Source", width="small"),
                "Current employment evidence": st.column_config.TextColumn("Current employment evidence", width="large"),
                "Profile": st.column_config.LinkColumn("Profile link", width="medium"),
            },
            hide_index=True,
            use_container_width=True,
        )

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        json_data = filtered_df.to_json(orient="records", indent=2).encode("utf-8")
        csv_col, json_col = st.columns(2)
        with csv_col:
            st.download_button(
                "Download CSV",
                csv_data,
                file_name=f"{company_name.lower()}_employees.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with json_col:
            st.download_button(
                "Download JSON",
                json_data,
                file_name=f"{company_name.lower()}_employees.json",
                mime="application/json",
                use_container_width=True,
            )
