import os
import json
import pandas as pd
import streamlit as st
from urllib.parse import urlparse

from src.config import get_llm
from src.agent.employee_agent import create_employee_agent, run_agent_stream
from src.tools.search_tools import parse_company_and_domain, classify_seniority

# -------------------------------------------------------------
# 1. Page Configuration & Claymorphic CSS Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="Executive & Employee Finder | AI Agent",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Claymorphic CSS Theme
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #f2f5f9;
        color: #2d3748;
    }

    /* Claymorphic Card */
    .clay-card {
        background: #f2f5f9;
        border-radius: 24px;
        box-shadow: 10px 10px 22px #d5dce7, -10px -10px 22px #ffffff;
        padding: 24px 28px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.6);
    }

    /* Claymorphic Hero Banner */
    .clay-hero {
        background: linear-gradient(135deg, #f5f8fc 0%, #edf2f7 100%);
        border-radius: 28px;
        box-shadow: 12px 12px 28px #d1d9e6, -12px -12px 28px #ffffff;
        padding: 32px 36px;
        margin-bottom: 28px;
        border: 1px solid rgba(255, 255, 255, 0.8);
    }

    /* Metric Badges */
    .clay-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 14px;
        font-weight: 600;
        font-size: 0.82rem;
        box-shadow: 3px 3px 7px #d5dce7, -3px -3px 7px #ffffff;
        margin-right: 6px;
    }
    .badge-csuite { background-color: #e0e7ff; color: #4338ca; }
    .badge-founder { background-color: #d1fae5; color: #065f46; }
    .badge-vp { background-color: #e0f2fe; color: #0369a1; }
    .badge-lead { background-color: #fef3c7; color: #92400e; }

    /* Metric Stat Card */
    .stat-card {
        background: #f2f5f9;
        border-radius: 20px;
        box-shadow: 8px 8px 18px #d5dce7, -8px -8px 18px #ffffff;
        padding: 18px 22px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.7);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #4f46e5;
        margin-bottom: 4px;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Live Agent Step Pill */
    .agent-step {
        background: #f8fafc;
        border-radius: 14px;
        padding: 10px 16px;
        margin: 6px 0;
        font-size: 0.88rem;
        box-shadow: inset 2px 2px 5px #d5dce7, inset -2px -2px 5px #ffffff;
        border-left: 4px solid #6366f1;
        font-family: monospace;
    }

    /* Primary Clay Button Override */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: #ffffff;
        font-weight: 700;
        border-radius: 18px;
        padding: 14px 28px;
        border: none;
        box-shadow: 6px 6px 14px rgba(79, 70, 229, 0.35), inset 2px 2px 4px rgba(255, 255, 255, 0.3), inset -2px -2px 4px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
        width: 100%;
        font-size: 1.05rem;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        box-shadow: 4px 4px 10px rgba(79, 70, 229, 0.45);
        transform: translateY(-1px);
    }
    div.stButton > button:first-child:active {
        box-shadow: inset 4px 4px 8px rgba(0, 0, 0, 0.3);
        transform: translateY(1px);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ebf0f7;
        border-right: 1px solid rgba(255, 255, 255, 0.7);
    }
    section[data-testid="stSidebar"] .clay-card {
        box-shadow: 6px 6px 14px #d1d9e6, -6px -6px 14px #ffffff;
    }

    /* Table Container */
    .stDataFrame {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 8px 8px 20px #d5dce7, -8px -8px 20px #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# 2. Sidebar: Model Switching within LangChain Framework
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 12px 0 20px 0;">
        <h2 style="font-weight: 800; color: #312e81; margin-bottom: 2px;">⚡ Model Engine</h2>
        <p style="font-size: 0.85rem; color: #64748b; margin: 0;">LangChain Model Orchestrator</p>
    </div>
    """, unsafe_allow_html=True)

    provider = st.selectbox(
        "LLM Provider",
        options=["Ollama (Local)", "Groq (Cloud)", "OpenAI", "Google Gemini"],
        index=0,
        help="Select which AI framework provider to use with LangChain.",
    )

    # Provider mapping
    provider_map = {
        "Ollama (Local)": "ollama",
        "Groq (Cloud)": "groq",
        "OpenAI": "openai",
        "Google Gemini": "gemini",
    }
    selected_provider_key = provider_map[provider]

    # Model options per provider
    model_options = {
        "ollama": ["llama3.2", "phi3", "qwen2.5", "mistral", "gemma2"],
        "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "openai": ["gpt-4o-mini", "gpt-4o", "o3-mini"],
        "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
    }

    selected_model = st.selectbox(
        "Active Model",
        options=model_options[selected_provider_key],
        index=0,
        help="Model used by the LangChain Agent for reasoning and extraction.",
    )

    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0.0 provides the most grounded and fact-based profile extractions.",
    )

    api_key_input = None
    if selected_provider_key != "ollama":
        env_var_name = {
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GOOGLE_API_KEY",
        }.get(selected_provider_key)
        
        default_val = os.getenv(env_var_name, "")
        api_key_input = st.text_input(
            f"{provider} API Key",
            type="password",
            value=default_val,
            placeholder=f"Enter {env_var_name}...",
        )
        if not api_key_input:
            st.caption(f"⚠️ {env_var_name} required to invoke {provider}.")

    st.markdown("---")
    st.markdown(f"""
    <div style="background: #ffffff; border-radius: 16px; padding: 14px; box-shadow: 4px 4px 10px #d5dce7; font-size: 0.82rem;">
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="height: 10px; width: 10px; background-color: #10b981; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>
            <strong style="color: #1e293b;">Engine Ready</strong>
        </div>
        <div style="color: #64748b;">Provider: <b>{selected_provider_key.upper()}</b></div>
        <div style="color: #64748b;">Model: <b>{selected_model}</b></div>
        <div style="color: #64748b;">Temperature: <b>{temperature}</b></div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# 3. Hero Header
# -------------------------------------------------------------
st.markdown("""
<div class="clay-hero">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;">
        <div>
            <span class="clay-badge badge-csuite" style="margin-bottom: 10px;">🤖 Autonomous LangChain AI Agent</span>
            <h1 style="font-weight: 800; color: #1e1b4b; font-size: 2.2rem; margin: 6px 0;">Executive & Employee Finder</h1>
            <p style="color: #475569; font-size: 1.02rem; margin: 0; max-width: 650px;">
                Discover CEOs, CTOs, Founders, Directors, and Managers across company websites, Crunchbase, The Org, and LinkedIn in real time.
            </p>
        </div>
        <div style="text-align: right;">
            <span class="clay-badge badge-founder">✨ Claymorphism UI</span>
            <span class="clay-badge badge-vp">🌐 Multi-Source SerpApi</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# 4. Search Input Section
# -------------------------------------------------------------
with st.container():
    st.markdown('<div class="clay-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2.5, 1.5])
    with col1:
        company_input = st.text_input(
            "Company Name or Website URL",
            placeholder="e.g. twelve.co or http://www.twelve.co or Stripe or Anthropic",
            help="Enter a company domain URL or company name to start searching.",
        )
    with col2:
        role_filter_input = st.text_input(
            "Target Department / Role Focus (Optional)",
            placeholder="e.g. Engineering, Founders, Product, C-Suite",
            help="Optionally focus the agent on specific leadership roles or leave empty for all executive roles.",
        )

    search_clicked = st.button("🔍 Discover Leadership Team", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# -------------------------------------------------------------
# 5. Agent Execution & Live Streaming
# -------------------------------------------------------------
if search_clicked:
    if not company_input.strip():
        st.warning("⚠️ Please enter a company name or website URL to begin.")
    else:
        parsed_name, parsed_domain = parse_company_and_domain(company_input)
        
        # Initialize dynamic LLM for LangChain
        try:
            custom_llm = get_llm(
                provider=selected_provider_key,
                model_name=selected_model,
                temperature=temperature,
                api_key=api_key_input,
            )
            agent_instance = create_employee_agent(llm_instance=custom_llm)
        except Exception as err:
            st.error(f"❌ Failed to initialize {provider} model: {err}")
            st.stop()

        # UI Progress elements
        st.markdown('<div class="clay-card">', unsafe_allow_html=True)
        st.markdown(f"### 🚀 Investigating Leadership at **{parsed_name}**" + (f" (`{parsed_domain}`)" if parsed_domain else ""))
        
        progress_placeholder = st.empty()
        log_expander = st.expander("👁️ View Agent Live Thinking & Tool Calls", expanded=True)
        step_logs = []
        
        with progress_placeholder:
            st.info("Agent starting: Formulating search queries across company website, directories, and LinkedIn...")

        # Run stream
        final_result_data = None
        try:
            for event in run_agent_stream(
                company_or_url=company_input,
                role_preference=role_filter_input if role_filter_input.strip() else None,
                agent_instance=agent_instance,
            ):
                if event["type"] == "action":
                    tool_name = event["tool"]
                    tool_args = event.get("args", {})
                    log_text = f"⚙️ Calling Tool: {tool_name} with {tool_args}"
                    step_logs.append(log_text)
                    with log_expander:
                        st.markdown(f"<div class='agent-step'>[Agent Tool Call] <b>{tool_name}</b><br><span style='color:#64748b;'>Args: {tool_args}</span></div>", unsafe_allow_html=True)
                    progress_placeholder.info(f"🔎 Agent searching via **{tool_name}**...")

                elif event["type"] == "observation":
                    tool_name = event["tool"]
                    log_text = f"✅ Received observation from: {tool_name}"
                    step_logs.append(log_text)
                    with log_expander:
                        st.markdown(f"<div class='agent-step' style='border-left-color: #10b981;'>[Observation Received] Search results aggregated for {tool_name}</div>", unsafe_allow_html=True)

                elif event["type"] == "result":
                    final_result_data = event.get("data", {})
                    progress_placeholder.success("🎉 Leadership extraction and verification complete!")

        except Exception as ex:
            progress_placeholder.error(f"❌ An error occurred during agent execution: {ex}")
            st.stop()

        st.markdown('</div>', unsafe_allow_html=True)

        # Store in session state for caching/viewing
        st.session_state["last_result"] = final_result_data
        st.session_state["company_searched"] = parsed_name


# -------------------------------------------------------------
# 6. Results Presentation in Table Form
# -------------------------------------------------------------
if "last_result" in st.session_state and st.session_state["last_result"]:
    result_data = st.session_state["last_result"]
    employees = result_data.get("employees", [])
    company_name = st.session_state.get("company_searched", "Company")

    st.markdown('<div class="clay-card">', unsafe_allow_html=True)
    st.markdown(f"## 📋 Verified Leadership Profiles for **{company_name}**")
    
    if not employees:
        st.warning("No public leadership profiles were conclusively retrieved for this company. Try a different company name or broader search.")
    else:
        # Normalize and prepare DataFrame
        records = []
        for emp in employees:
            name = emp.get("name", "").strip()
            role = emp.get("role") or emp.get("job_title", "").strip()
            seniority = emp.get("seniority") or classify_seniority(role)
            source_url = emp.get("source_url", "").strip()
            
            # Domain parsing
            domain_label = emp.get("source_domain", "")
            if not domain_label and source_url:
                parsed_u = urlparse(source_url)
                domain_label = parsed_u.netloc.replace("www.", "")

            records.append({
                "Name": name,
                "Role / Designation": role,
                "Seniority Level": seniority,
                "Platform / Source": domain_label.capitalize() if domain_label else "Web",
                "Profile URL": source_url,
            })

        df = pd.DataFrame(records)

        # Stat cards row
        c_suite_count = sum(1 for r in records if r["Seniority Level"] == "C-Suite")
        founders_count = sum(1 for r in records if "Founder" in r["Seniority Level"])
        vps_count = sum(1 for r in records if "VP" in r["Seniority Level"] or "Director" in r["Seniority Level"])
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{len(records)}</div><div class="stat-label">Total Leaders Found</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="stat-card"><div class="stat-number" style="color: #4338ca;">{c_suite_count}</div><div class="stat-label">C-Suite Executives</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="stat-card"><div class="stat-number" style="color: #065f46;">{founders_count}</div><div class="stat-label">Founders & Presidents</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="stat-card"><div class="stat-number" style="color: #0369a1;">{vps_count}</div><div class="stat-label">VPs & Directors</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Seniority Filter Multi-Select
        seniority_options = sorted(list(set(df["Seniority Level"].dropna().unique())))
        selected_seniority = st.multiselect(
            "Filter by Seniority Level:",
            options=seniority_options,
            default=seniority_options,
        )

        filtered_df = df[df["Seniority Level"].isin(selected_seniority)]

        # Display Table with Clickable Link Column Config
        st.dataframe(
            filtered_df,
            column_config={
                "Name": st.column_config.TextColumn("Executive Name", width="medium"),
                "Role / Designation": st.column_config.TextColumn("Designation", width="large"),
                "Seniority Level": st.column_config.TextColumn("Seniority Tier", width="medium"),
                "Platform / Source": st.column_config.TextColumn("Source Domain", width="small"),
                "Profile URL": st.column_config.LinkColumn("Verified Profile Link", width="medium"),
            },
            hide_index=True,
            use_container_width=True,
        )

        # Export Buttons
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        json_data = json.dumps(records, indent=2).encode('utf-8')
        
        btn_col1, btn_col2, _ = st.columns([1, 1, 2])
        with btn_col1:
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"{company_name.lower()}_leadership.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with btn_col2:
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"{company_name.lower()}_leadership.json",
                mime="application/json",
                use_container_width=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)
