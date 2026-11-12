import json
import os
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient

ST_CONFIG_FILE = "style_anchor.json"

DEFAULT_ANCHOR = {
    "core_aesthetic": "Cinematic AMD Radeon tech style, razor-sharp edge details, multi-layer microchip geometry, futuristic processors and circuit boards, sine waves, matte black and brushed titanium substrates, UHD, volumetric lighting.",
    "color_palette": "Crimson Red (#C50000), Royal Gold (#D4AF37), Deep Carbon (#111111)",
    "negative_constraints": "No cartoonish illustration, no flat lighting, no pastel colors, no chaotic or cluttered backgrounds."
}

def load_anchor():
    if os.path.exists(ST_CONFIG_FILE):
        with open(ST_CONFIG_FILE, "r") as f:
            return json.load(f)
    else:
        with open(ST_CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_ANCHOR, f, indent=4)
        return DEFAULT_ANCHOR

anchor_data = load_anchor()

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

@st.cache_resource
def get_hf_client():
    token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN"))
    return InferenceClient(api_key=token)

client = get_hf_client()

st.set_page_config(page_title="CIAT25 Visual Consistency Engine (API)", layout="wide")

st.title("⚡ CIAT25.1: Automated Visual Engine")
st.markdown("*Gate 0 Perimeter Ingress & Free-Tier Serverless Execution*")

# Sidebar: Manage Style Anchor & Dynamic Thresholds
with st.sidebar:
    st.header("1. Style Anchor Vault")
    new_aesthetic = st.text_area("Core Aesthetic", value=anchor_data["core_aesthetic"], height=100)
    new_palette = st.text_input("Color Palette", value=anchor_data["color_palette"])
    new_negative = st.text_area("Negative Constraints", value=anchor_data["negative_constraints"], height=80)
    
    st.header("2. Gate Calibration")
    similarity_threshold = st.slider("Drift Tolerance Threshold", min_value=0.1, max_value=0.9, value=0.4, step=0.05)
    
    if st.button("Save Style Anchor"):
        updated_anchor = {
            "core_aesthetic": new_aesthetic,
            "color_palette": new_palette,
            "negative_constraints": new_negative
        }
        with open(ST_CONFIG_FILE, "w") as f:
            json.dump(updated_anchor, f, indent=4)
        st.success("Style Anchor updated successfully!")
        anchor_data = updated_anchor

# Main Panel: Prompt Ingress & Automated API Generation
st.header("3. Structural Prompt Realignment Gate")
raw_prompt = st.text_input("Enter your prompt or descriptive scene:", placeholder="e.g., an entire landscape of folded blue-white ice...")

if st.button("Realign, Synthesize & Generate via API"):
    if not raw_prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Evaluating semantic drift and weaving prose payload..."):
            anchor_embedding = model.encode(anchor_data["core_aesthetic"], convert_to_tensor=True)
            prompt_embedding = model.encode(raw_prompt, convert_to_tensor=True)
            
            similarity_score = util.cos_sim(prompt_embedding, anchor_embedding).item()
            
            # PROSE WEAVING: Integrates anchor parameters directly into the narrative syntax
            clean_subject = raw_prompt.strip().rstrip('.')
            fused_prompt_string = (
                f"{clean_subject}, built with {anchor_data['core_aesthetic']}, "
                f"graded in {anchor_data['color_palette']}, "
                f"high-end production render, volumetric lighting, 8k resolution."
            )
            
            # Machine Audit Payload
            structured_payload = {
                "ingress_metadata": {
                    "gate": "Gate-0-Perimeter",
                    "similarity_score": round(similarity_score, 4),
                    "threshold_applied": similarity_threshold,
                    "status": "APPROVED" if similarity_score >= similarity_threshold else "RE-ANCHORED"
                },
                "fused_prompt": fused_prompt_string,
                "negative_constraints": anchor_data['negative_constraints']
            }
            
            # Display Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Semantic Similarity Score", value=f"{similarity_score:.2f}")
            with col2:
                status_color = "🟢" if similarity_score >= similarity_threshold else "🟠"
                st.markdown(f"{status_color} **Status: {structured_payload['ingress_metadata']['status']}**")
            
            st.markdown("---")
            st.subheader("🚀 Cloud API Generation Output")
            
            try:
                # Using a model endpoint that supports free serverless inference on Hugging Face
                image = client.text_to_image(
                    prompt=fused_prompt_string,
                    negative_prompt=anchor_data['negative_constraints'],
                    model="stabilityai/stable-diffusion-2-1"
                )
                st.image(image, caption="Generated via Free-Tier Serverless API", use_column_width=True)
            except Exception as e:
                st.error(f"Inference Engine Error: {str(e)}")
            
            st.subheader("📋 Machine Audit JSON Payload")
            st.code(json.dumps(structured_payload, indent=4), language="json")
