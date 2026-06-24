from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io

print("Loading CLIP model...")

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

object_tags = [
    # Nature & Environment
    "tree", "forest", "mountain", "ocean", "river", "beach", "sky", "cloud", "sun", "moon",
    "star", "rain", "snow", "landscape", "grass", "flower", "animal", "bird", "dog", "cat",
    "fish", "insect",

    # People & Anatomy
    "person", "man", "woman", "child", "face", "eye", "hair", "hand", "crowd", "portrait",
    "smile", "clothing", "shoes",

    # Urban & Architecture
    "city", "building", "house", "street", "road", "bridge", "car", "train", "airplane", "boat",
    "bicycle", "architecture", "interior", "exterior", "window", "door", "furniture",

    # Objects & Technology
    "phone", "computer", "camera", "book", "pen", "cup", "bottle", "food", "drink", "plate",
    "table", "chair", "bed", "clock", "watch", "glasses",

    # Abstract, Concepts & Photographic Attributes
    "blur", "sharp", "dark", "bright", "colorful", "monochrome", "vintage", "modern", "abstract", "pattern",
    "texture", "shadow", "light", "reflection", "silhouette", "macro", "close-up", "wide-angle",

    # Activities & Events
    "sports", "running", "jumping", "swimming", "eating", "working", "playing", "dancing", "concert", "party",
    "wedding", "travel", "art", "music"
]

vibe_tags = [
    # Internet Aesthetics & Subcultures
    "cyberpunk", "vaporwave", "cottagecore", "dark academia", "steampunk", "minimalist",
    "maximalist", "retro", "vintage", "futuristic", "y2k", "grunge", "gothic", "boho",
    "industrial", "midcentury", "art deco", "synthwave", "lo-fi", "wabi-sabi",

    # Lighting & Atmospheric Qualities
    "golden hour", "blue hour", "neon", "cinematic", "ethereal", "moody", "washed out",
    "high contrast", "low key", "high key", "pastel", "muted", "vibrant", "desaturated",
    "gloomy", "radiant", "hazy", "crisp", "shadow play", "lens flare",

    # Emotional Tone & Mood
    "melancholic", "nostalgic", "euphoric", "serene", "chaotic", "eerie", "romantic",
    "ominous", "uplifting", "cozy", "lonely", "intimate", "whimsical", "somber",
    "playful", "tense", "peaceful", "mysterious", "dramatic", "joyful",

    # Artistic & Compositional Execution
    "surreal", "dreamlike", "gritty", "polished", "raw", "documentary", "posed",
    "abstract", "geometric", "organic", "soft focus", "hyperrealistic", "painterly",
    "glitch", "psychedelic", "avant-garde", "kitsch", "balanced", "evocative", "conceptual",

    # Energy, Setting & Thematic Undertones
    "dynamic", "static", "energetic", "tranquil", "rebellious", "traditional", "modern",
    "rustic", "urban decay", "lush", "barren", "claustrophobic", "expansive", "liminal",
    "domestic", "wild", "sterile", "opulent", "rugged", "delicate"
]


def get_image_tags(image_bytes, is_vibe_tags=True):
    tags = vibe_tags if is_vibe_tags else object_tags
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(text=tags, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).detach().numpy()[0]

    results = [{"tag": label, "probability": float(prob)} for label, prob in zip(tags, probs)]
    results = sorted(results, key=lambda x: x["probability"], reverse=True)
    return results[:5]