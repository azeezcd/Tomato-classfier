import os
import sys
from predict import get_prediction

# ============================================================
# LOCAL AGRONOMY KNOWLEDGE BASE
# ============================================================
DISEASE_KNOWLEDGE_BASE = {
    "Tomato Bacterial Spot": {
        "summary": "Bacterial Spot is caused by Xanthomonas species, producing small, water-soaked dark lesions on leaves and fruit, leading to heavy leaf drop in warm, wet conditions.",
        "treatment": [
            "Prune infected lower leaves immediately using sanitized shears.",
            "Spray with an organic copper-based bactericide every 7 to 10 days.",
            "Water plants at the base (drip irrigation) to prevent leaf splash."
        ],
        "prevention": [
            "Rotate crops with non-solanaceous plants (e.g., corn, beans) every 2 years.",
            "Use certified disease-free seeds and spray neem oil preventatively."
        ]
    },
    "Tomato Leaf Mold": {
        "summary": "Leaf Mold is a fungal disease caused by Passalora fulva, thriving in high humidity with pale yellow leaf spots and olive-green velvety undersides.",
        "treatment": [
            "Prune inner canopy leaves to maximize air circulation through the plant.",
            "Apply a bio-fungicide containing Bacillus subtilis.",
            "Reduce humidity around plants below 85%."
        ],
        "prevention": [
            "Use drip irrigation so foliage stays completely dry overnight.",
            "Plant mold-resistant tomato cultivars."
        ]
    },
    "Tomato Early Blight": {
        "summary": "Early Blight is caused by Alternaria solani, characterized by dark brown spots with distinct concentric 'target' rings on older foliage.",
        "treatment": [
            "Remove and destroy severely spotted lower leaves.",
            "Apply copper or sulfur-based fungicides at the first sign of leaf spots.",
            "Mulch heavily around the plant base to stop soil spores from splashing upward."
        ],
        "prevention": [
            "Space plants at least 24 inches apart to ensure rapid leaf drying.",
            "Practice a strict 3-year crop rotation schedule."
        ]
    },
    "Tomato Late Blight": {
        "summary": "Late Blight is a destructive water mold (Phytophthora infestans) that causes large, water-soaked dark lesions on leaves and stems during cool, damp weather.",
        "treatment": [
            "Immediately cut off and destroy infected stems (do NOT compost).",
            "Apply bio-fungicides or copper sprays on uninfected surrounding stems.",
            "Isolate affected plants to prevent airborne spore dissemination."
        ],
        "prevention": [
            "Avoid overhead watering and ensure optimal canopy airflow.",
            "Plant certified late-blight-resistant varieties."
        ]
    },
    "Tomato Septoria Leaf Spot": {
        "summary": "Septoria Leaf Spot is a fungal issue causing tiny circular spots with dark margins and gray centers, moving from lower leaves upward.",
        "treatment": [
            "Strip off infected lower leaves immediately.",
            "Apply an organic copper fungicide weekly during wet weather.",
            "Apply fresh mulch to cap soil spores."
        ],
        "prevention": [
            "Clean and sanitize all garden stakes and cages before planting.",
            "Avoid working in the garden while foliage is wet."
        ]
    },
    "Tomato Target Spot": {
        "summary": "Target Spot is caused by Corynespora cassiicola, forming dark brown leaf spots with light centers that enlarge into target-like lesions.",
        "treatment": [
            "Prune infected foliage to open up air flow.",
            "Apply broad-spectrum bio-fungicide or chlorothalonil/copper sprays.",
            "Ensure proper soil drainage."
        ],
        "prevention": [
            "Destroy all crop residues immediately after harvest.",
            "Maintain balanced soil nutrition to avoid plant stress."
        ]
    },
    "Tomato Spider Mites Two-Spotted Spider Mite": {
        "summary": "Two-Spotted Spider Mites are tiny pests that suck plant sap, causing yellow stippling on leaves and delicate silk webbing underneath.",
        "treatment": [
            "Spray leaves thoroughly with insecticidal soap or neem oil.",
            "Blast the undersides of leaves with a firm stream of water to dislodge mites.",
            "Introduce beneficial predatory mites (Phytoseiulus persimilis)."
        ],
        "prevention": [
            "Keep soil consistently moist to reduce dry, dusty conditions favored by mites.",
            "Avoid excessive nitrogen fertilizers which boost mite reproduction."
        ]
    },
    "Tomato Yellow Leaf Curl Virus": {
        "summary": "Tomato Yellow Leaf Curl Virus (TYLCV) is transmitted by whiteflies, causing leaves to curl upward, turn yellow at margins, and stunt growth.",
        "treatment": [
            "Remove and discard infected plants immediately to protect healthy crops.",
            "Control whitefly populations using yellow sticky traps and insecticidal soap.",
            "Cover young plants with fine mesh row covers."
        ],
        "prevention": [
            "Plant virus-resistant tomato hybrids (e.g., 'Tycoon' or 'Skyway').",
            "Keep garden beds completely free of weed hosts."
        ]
    },
    "Tomato Healthy": {
        "summary": "Your tomato plant foliage shows no visible signs of fungal, bacterial, or viral disease.",
        "treatment": [
            "No active disease treatment is required at this time.",
            "Continue standard watering and organic fertilization schedules.",
            "Inspect foliage weekly for early warning signs."
        ],
        "prevention": [
            "Maintain soil mulch to regulate moisture and soil temperature.",
            "Ensure proper spacing for sunny, well-ventilated growth."
        ]
    }
}


def generate_local_ai_report(disease_name: str, confidence: float) -> str:
    info = DISEASE_KNOWLEDGE_BASE.get(disease_name)

    if not info:
        for k in DISEASE_KNOWLEDGE_BASE:
            if k.lower() in disease_name.lower():
                info = DISEASE_KNOWLEDGE_BASE[k]
                break

    report = []

    if info:
        report.append(f"### What is {disease_name}?")
        report.append(f"{info['summary']}\n")

        report.append("### Immediate Action Steps:")
        for step in info['treatment']:
            report.append(f"• {step}")

        report.append("\n### Long-Term Crop Prevention:")
        for tip in info['prevention']:
            report.append(f"• {tip}")
    else:
        report.append("### Immediate Actions:")
        report.append("• Prune heavily spotted foliage to reduce spore spread.")
        report.append("• Apply an organic copper-based multi-purpose agricultural spray.")
        report.append("• Consult your local agricultural extension office for localized guidance.")

    return "\n".join(report)


# ============================================================
# EXECUTION PIPELINE
# ============================================================
image_path = "test/septoria.JPG"

if not os.path.exists(image_path):
    print(f"Error: Target image '{image_path}' not found.")
    sys.exit(1)

# Step 1: Run local vision model prediction
result = get_prediction(image_path)

# Step 2: Extract ONLY the top prediction
top = result[0]
raw_class = top["class"]
confidence = top["confidence"]
clean_name = raw_class.replace("___", " ").replace("_", " ").strip()

# Step 3: Print single result & report
print("=" * 60)
print("🤖 AI AGRONOMIST DIAGNOSIS & ACTION PLAN")
print("=" * 60)
print(f"Detected Condition : {clean_name}")
print(f"Match Confidence   : {confidence:.2f}%\n")

report = generate_local_ai_report(clean_name, confidence)
print(report)