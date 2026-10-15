import importlib

try:
    st = importlib.import_module("streamlit")
except ModuleNotFoundError as exc:
    if exc.name == "streamlit":
        raise SystemExit("Streamlit is required to run this app. Install it with: pip install streamlit") from exc
    raise
try:
    torch = importlib.import_module("torch")
    transforms = importlib.import_module("torchvision.transforms")
    nn = importlib.import_module("torch.nn")
    timm = importlib.import_module("timm")
except ModuleNotFoundError as exc:
    if exc.name in {"torch", "torchvision", "timm"}:
        raise SystemExit(
            f"{exc.name} is required to run this app. Install the dependencies with: "
            "pip install torch torchvision timm"
        ) from exc
    raise
try:
    Image = importlib.import_module("PIL.Image")
except ModuleNotFoundError as exc:
    if exc.name == "PIL":
        raise SystemExit("Pillow is required to run this app. Install it with: pip install pillow") from exc
    raise

# ==========================================
# 1. LOAD MODEL ARCHITECTURE & WEIGHTS
# ==========================================
class JaundiceClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b0'):
        super(JaundiceClassifier, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out.squeeze(1)

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JaundiceClassifier(model_name='efficientnet_b0')
    try:
        model.load_state_dict(torch.load('jaundice_efficientnet_b0.pth', map_location=device))
    except FileNotFoundError:
        st.error("Model weights file 'jaundice_efficientnet_b0.pth' not found. Please run train_jaundice.py first.")
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# ==========================================
# 2. IMAGE PREPROCESSING
# ==========================================
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ==========================================
# 3. STREAMLIT UI DESIGN
# ==========================================
st.set_page_config(page_title="AI Jaundice Screening Tool", page_icon="👁️", layout="centered")

st.title("👁️ AI-Powered Jaundice Screening Tool")
st.markdown("""
This application serves as an **early-awareness preliminary screener**. It evaluates eye images (focusing on the sclera) to check for potential signs of yellow discoloration associated with jaundice.
> **Disclaimer:** This tool is not a replacement for professional medical diagnosis or laboratory blood tests.
""")

st.sidebar.header("Input Method")
input_option = st.sidebar.radio("Choose how to provide an eye image:", ("Upload Image File", "Take Photo via Camera"))

image = None

if input_option == "Upload Image File":
    uploaded_file = st.file_uploader("Upload a clear, well-lit image of the eye...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
elif input_option == "Take Photo via Camera":
    camera_image = st.camera_input("Position your eye clearly in front of the camera and snap a photo")
    if camera_image is not None:
        image = Image.open(camera_image).convert('RGB')

# ==========================================
# 4. INFERENCE & RESULTS
# ==========================================
if image is not None:
    st.image(image, caption="Selected Eye Image", use_container_width=True)
    
    if st.button("Run Jaundice Screening", type="primary"):
        with st.spinner("Analyzing image features..."):
            input_tensor = preprocess_image(image).to(device)
            
            with torch.no_grad():
                output_logit = model(input_tensor)
                probability = torch.sigmoid(output_logit).item()
            
            st.divider()
            st.subheader("Screening Results")
            
            col1, col2 = st.columns(2)
            col1.metric("Calculated Risk Score", f"{probability:.2f}")
            
            if probability > 0.6:
                col2.error("Status: Possible Signs Detected")
                st.warning("""
                **Recommendation:** Possible signs of yellow discoloration have been detected. 
                Please schedule a consultation with a healthcare professional immediately for clinical evaluation and serum bilirubin laboratory testing.
                """)
            elif probability >= 0.3:
                col2.warning("Status: Borderline / Indeterminate")
                st.info("""
                **Recommendation:** Results are inconclusive, potentially due to lighting conditions or image angle. 
                Try retaking the photo in bright, natural daylight focusing tightly on the white of the eye.
                """)
            else:
                col2.success("Status: No Apparent Signs")
                st.success("""
                **Recommendation:** No apparent signs of jaundice detected based on this image. 
                Continue routine health monitoring. If symptoms persist or you feel unwell, consult a doctor.
                """)