import streamlit as st
from pathlib import Path
import base64

def inject_brand_css(dark: bool = True):
    """Inject custom CSS for ALPHA brand styling"""
    css = """
    <style>
    /* Global Dark Theme */
    .stApp {
        background: #0B0B0B;
        color: #F5F5F5;
    }
    
    /* ALPHA Card Component */
    .alpha-card {
        background: #111111;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 2px 16px rgba(0, 0, 0, 0.25);
        margin: 12px 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .alpha-card:hover {
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        transition: all 0.3s ease;
    }
    
    /* Buttons */
    .stButton > button {
        background: #FFFFFF;
        color: #0B0B0B;
        border-radius: 999px;
        padding: 8px 24px;
        font-weight: 700;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #E5E5E5;
        box-shadow: 0 4px 12px rgba(255, 255, 255, 0.2);
    }
    
    /* Metric Badge */
    .metric-badge {
        background: #FFFFFF;
        color: #0B0B0B;
        border-radius: 999px;
        padding: 6px 14px;
        font-weight: 700;
        display: inline-block;
        margin: 4px;
    }
    
    .metric-badge.success {
        background: #10B981;
        color: white;
    }
    
    .metric-badge.warning {
        background: #F59E0B;
        color: white;
    }
    
    .metric-badge.danger {
        background: #EF4444;
        color: white;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: #FFFFFF;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 700;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: #111111;
        border-radius: 8px;
    }
    
    /* Dataframe */
    .dataframe {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0B0B0B;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def _read_logo_base64() -> str | None:
    # .../Streamlit/assets/logo.png
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "logo.png"
    if not logo_path.exists():
        return None
    return base64.b64encode(logo_path.read_bytes()).decode("ascii")

def brand_header(title: str = "ALPHA COMPANY • Customer Intelligence Suite"):
    b64 = _read_logo_base64()
    logo_html = (
        f'<img src="data:image/png;base64,{b64}" alt="logo" '
        'style="height:72px;margin-right:16px;border-radius:10px;" />'
        if b64 else ""
    )

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;margin:10px 0 24px 0;">
            {logo_html}
            <div style="display:flex;flex-direction:column">
                <h1 style="margin:0;padding:0;font-size:2.2rem;line-height:1.1;">{title}</h1>
                <div style="opacity:.8;font-weight:600;letter-spacing:.02em">ALPHA COMPANY • Customer Intelligence Suite</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )