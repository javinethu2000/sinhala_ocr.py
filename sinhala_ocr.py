import os
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict
import tensorflow as tf
from keras import layers, models
from keras.optimizers import SGD
from keras.utils import to_categorical
from flask import Flask, request, jsonify, send_file, render_template
import io
import base64
from werkzeug.utils import secure_filename


# CONFIG

IMG_SIZE = 28   # same as paper (28x28)
CHANNELS = 1
MODEL_NAME = "sinhala_cnn.h5"
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp"}
# Comprehensive Sinhala character mapping using English labels to avoid Unicode path issues on Windows
# Each label maps to its corresponding Sinhala Unicode character for training and display
label_to_unicode = {
    # Independent vowels
    "a": "අ",      # a
    "aa": "ආ",     # aa
    "ae": "ඇ",     # ae
    "aae": "ඈ",    # aae
    "i": "ඉ",      # i
    "ii": "ඊ",     # ii
    "u": "උ",      # u
    "uu": "ඌ",     # uu
    "ri": "ඍ",     # ri
    "e": "එ",      # e
    "ee": "ඒ",     # ee
    "ai": "ඓ",     # ai
    "o": "ඔ",      # o
    "oo": "ඕ",     # oo
    "au": "ඖ",     # au
    
    # Consonants with vowel combinations (consonant + vowel diacritics)
    # ක-series
    "ka": "ක", "kaa": "කා", "ki": "කි", "kii": "කී", "ku": "කු", "kuu": "කූ", 
    "kae": "කැ", "ke": "කේ", "kai": "කෛ", "ko": "කො", "koo": "කෝ", "kau": "කෞ",
    
    # ඛ-series  
    "kha": "ඛ", "khaa": "ඛා", "khi": "ඛි", "khii": "ඛී", "khu": "ඛු", "khuu": "ඛූ",
    "khae": "ඛැ", "khe": "ඛේ", "khai": "ඛෛ", "kho": "කො", "khoo": "කෝ", "khau": "ඛෞ",
    
    # ග-series
    "ga": "ග", "gaa": "ගා", "gi": "ගි", "gii": "ගී", "gu": "ගු", "guu": "ගූ",
    "gae": "ගැ", "ge": "ගේ", "gai": "ගෛ", "go": "ගො", "goo": "ගෝ", "gau": "ගෞ",
    
    # ඝ-series
    "gha": "ඝ", "ghaa": "ඝා", "ghi": "ඝි", "ghii": "ඝී", "ghu": "ඝු", "ghuu": "ඝූ",
    "ghae": "ඝැ", "ghe": "ඝේ", "ghai": "ඝෛ", "gho": "ඝො", "ghoo": "ඝෝ", "ghau": "ඝෞ",
    
    # ඞ-series
    "nga": "ඞ", "ngaa": "ඞා", "ngi": "ඞි", "ngii": "ඞී", "ngu": "ඞු", "nguu": "ඞූ",
    "ngae": "ඞැ", "nge": "ඞේ", "ngai": "ඞෛ", "ngo": "ඞො", "ngoo": "ඞෝ", "ngau": "ඞෞ",
    
    # ච-series
    "ca": "ච", "caa": "චා", "ci": "චි", "cii": "චී", "cu": "චු", "cuu": "චූ",
    "cae": "චැ", "ce": "චේ", "cai": "චෛ", "co": "චො", "coo": "චෝ", "cau": "චෞ",
    
    # ඡ-series
    "cha": "ඡ", "chaa": "ඡා", "chi": "ඡි", "chii": "ඡී", "chu": "ඡු", "chuu": "ඡූ",
    "chae": "ඡැ", "che": "ඡේ", "chai": "ඡෛ", "cho": "ඡො", "choo": "ඡෝ", "chau": "ඡෞ",
    
    # ජ-series
    "ja": "ජ", "jaa": "ජා", "ji": "ජි", "jii": "ජී", "ju": "ජු", "juu": "ජූ",
    "jae": "ජැ", "je": "ජේ", "jai": "ජෛ", "jo": "ජො", "joo": "ජෝ", "jau": "ජෞ",
    
    # ඣ-series
    "jha": "ඣ", "jhaa": "ඣා", "jhi": "ඣි", "jhii": "ඣී", "jhu": "ඣු", "jhuu": "ඣූ",
    "jhae": "ඣැ", "jhe": "ඣේ", "jhai": "ඣෛ", "jho": "ඣො", "jhoo": "ඣෝ", "jhau": "ඣෞ",
    
    # ඤ-series
    "nya": "ඤ", "nyaa": "ඤා", "nyi": "ඤි", "nyii": "ඤී", "nyu": "ඤු", "nyuu": "ඤූ",
    "nyae": "ඤැ", "nye": "ඤේ", "nyai": "ඤෛ", "nyo": "ඤො", "nyoo": "ඤෝ", "nyau": "ඤෞ",
    
    # ට-series
    "tta": "ට", "ttaa": "ටා", "tti": "ටි", "ttii": "ටී", "ttu": "ටු", "ttuu": "ටූ",
    "ttae": "ටැ", "tte": "ටේ", "ttai": "ටෛ", "tto": "ටො", "ttoo": "ටෝ", "ttau": "ටෞ",
    
    # ඨ-series
    "ttha": "ඨ", "tthaa": "ඨා", "tthi": "ඨි", "tthii": "ඨී", "tthu": "ඨු", "tthuu": "ඨූ",
    "tthae": "ඨැ", "tthe": "ඨේ", "tthai": "ඨෛ", "ttho": "ඨො", "tthoo": "ඨෝ", "tthau": "ඨෞ",
    
    # ඩ-series
    "dda": "ඩ", "ddaa": "ඩා", "ddi": "ඩි", "ddii": "ඩී", "ddu": "ඩු", "dduu": "ඩූ",
    "ddae": "ඩැ", "dde": "ඩේ", "ddai": "ඩෛ", "ddo": "ඩො", "ddoo": "ඩෝ", "ddau": "ඩෞ",
    
    # ඪ-series
    "ddha": "ඪ", "ddhaa": "ඪා", "ddhi": "ඪි", "ddhii": "ඪී", "ddhu": "ඪු", "ddhuu": "ඪූ",
    "ddhae": "ඪැ", "ddhe": "ඪේ", "ddhai": "ඪෛ", "ddho": "ඪො", "ddhoo": "ඪෝ", "ddhau": "ඪෞ",
    
    # ණ-series
    "nna": "ණ", "nnaa": "ණා", "nni": "ණි", "nnii": "ණී", "nnu": "ණු", "nnuu": "ණූ",
    "nnae": "ණැ", "nne": "ණේ", "nnai": "ණෛ", "nno": "ණො", "nnoo": "ණෝ", "nnau": "ණෞ",
    
    # ත-series
    "ta": "ත", "taa": "තා", "ti": "ති", "tii": "තී", "tu": "තු", "tuu": "තූ",
    "tae": "තැ", "te": "තේ", "tai": "තෛ", "to": "තො", "too": "තෝ", "tau": "තෞ",
    
    # ථ-series
    "tha": "ථ", "thaa": "థా", "thi": "థි", "thii": "థී", "thu": "థు", "thuu": "థූ",
    "thae": "థැ", "the": "థේ", "thai": "థෛ", "tho": "థො", "thoo": "థෝ", "thau": "థෞ",
    
    # ද-series
    "da": "ද", "daa": "දා", "di": "දි", "dii": "දී", "du": "දු", "duu": "දූ",
    "dae": "දැ", "de": "දේ", "dai": "දෛ", "do": "දො", "doo": "දෝ", "dau": "දෞ",
    
    # ධ-series
    "dha": "ධ", "dhaa": "ධා", "dhi": "ධි", "dhii": "ධී", "dhu": "ධු", "dhuu": "ධූ",
    "dhae": "ධැ", "dhe": "ධේ", "dhai": "ධෛ", "dho": "ධො", "dhoo": "ධෝ", "dhau": "ධෞ",
    
    # න-series
    "na": "න", "naa": "නා", "ni": "නි", "nii": "නී", "nu": "නු", "nuu": "නූ",
    "nae": "නැ", "ne": "නේ", "nai": "නෛ", "no": "නො", "noo": "නෝ", "nau": "නෞ",
    
    # ප-series
    "pa": "ප", "paa": "පා", "pi": "පි", "pii": "පී", "pu": "පු", "puu": "පූ",
    "pae": "පැ", "pe": "පේ", "pai": "පෛ", "po": "පො", "poo": "පෝ", "pau": "පෞ",
    
    # ඵ-series
    "pha": "ඵ", "phaa": "ඵා", "phi": "ඵි", "phii": "ඵී", "phu": "ඵු", "phuu": "ඵූ",
    "phae": "ඵැ", "phe": "ඵේ", "phai": "ඵෛ", "pho": "ඵො", "phoo": "ඵෝ", "phau": "ඵෞ",
    
    # බ-series
    "ba": "බ", "baa": "බා", "bi": "බි", "bii": "බී", "bu": "බු", "buu": "බූ",
    "bae": "බැ", "be": "බේ", "bai": "බෛ", "bo": "බො", "boo": "බෝ", "bau": "බෞ",
    
    # භ-series
    "bha": "භ", "bhaa": "භා", "bhi": "භි", "bhii": "භී", "bhu": "භු", "bhuu": "භූ",
    "bhae": "භැ", "bhe": "භේ", "bhai": "භෛ", "bho": "භො", "bhoo": "භෝ", "bhau": "භෞ",
    
    # ම-series
    "ma": "ම", "maa": "මා", "mi": "මි", "mii": "මී", "mu": "මු", "muu": "මූ",
    "mae": "මැ", "me": "මේ", "mai": "මෛ", "mo": "මො", "moo": "මෝ", "mau": "මෞ",
    
    # ය-series
    "ya": "ය", "yaa": "යා", "yi": "යි", "yii": "යී", "yu": "යු", "yuu": "යූ",
    "yae": "යැ", "ye": "යේ", "yai": "යෛ", "yo": "යො", "yoo": "යෝ", "yau": "යෞ",
    
    # ර-series
    "ra": "ර", "raa": "රා", "ri": "රි", "rii": "රී", "ru": "රු", "ruu": "රූ",
    "rae": "රැ", "re": "රේ", "rai": "රෛ", "ro": "රො", "roo": "රෝ", "rau": "රෞ",
    
    # ල-series
    "la": "ල", "laa": "ලා", "li": "ලි", "lii": "ලී", "lu": "ලු", "luu": "ලූ",
    "lae": "ලැ", "le": "ලේ", "lai": "ලෛ", "lo": "ලො", "loo": "ලෝ", "lau": "ලෞ",
    
    # ව-series
    "va": "ව", "vaa": "වා", "vi": "වි", "vii": "වී", "vu": "වු", "vuu": "වූ",
    "vae": "වැ", "ve": "වේ", "vai": "වෛ", "vo": "වො", "voo": "වෝ", "vau": "වෞ",
    
    # ශ-series
    "sha": "ශ", "shaa": "ශා", "shi": "ශි", "shii": "ශී", "shu": "ශු", "shuu": "ශූ",
    "shae": "ශැ", "she": "ශේ", "shai": "ශෛ", "sho": "ශො", "shoo": "ශෝ", "shau": "ශෞ",
    
    # ෂ-series
    "ssa": "ෂ", "ssaa": "ෂා", "ssi": "ෂි", "ssii": "ෂී", "ssu": "ෂු", "ssuu": "ෂූ",
    "ssae": "ෂැ", "sse": "ෂේ", "ssai": "ෂෛ", "sso": "ෂො", "ssoo": "ෂෝ", "ssau": "ෂෞ",
    
    # ස-series
    "sa": "ස", "saa": "සා", "si": "සි", "sii": "සී", "su": "සු", "suu": "සූ",
    "sae": "සැ", "se": "සේ", "sai": "සෛ", "so": "සො", "soo": "සෝ", "sau": "සෞ",
    
    # හ-series
    "ha": "හ", "haa": "හා", "hi": "හි", "hii": "හී", "hu": "හු", "huu": "හූ",
    "hae": "හැ", "he": "හේ", "hai": "හෛ", "ho": "හො", "hoo": "හෝ", "hau": "හෞ",
    
    # ළ-series
    "lla": "ළ", "llaa": "ළා", "lli": "ළි", "llii": "ළී", "llu": "ළු", "lluu": "ළූ",
    "llae": "ළැ", "lle": "ළේ", "llai": "ළෛ", "llo": "ළො", "lloo": "ළෝ", "llau": "ළෞ",
    
    # ෆ-series
    "fa": "ෆ", "faa": "ෆා", "fi": "ෆි", "fii": "ෆී", "fu": "ෆු", "fuu": "ෆූ",
    "fae": "ෆැ", "fe": "ෆේ", "fai": "ෆෛ", "fo": "ෆො", "foo": "ෆෝ", "fau": "ෆෞ",
    
    # Additional characters to complete 55-character mapping
    "rii": "ඎ",    # rii
    "lu": "ඏ",     # lu
    "luu": "ඐ",    # luu
    "jnya": "ඥ",   # jnya
    "ndda": "ඬ",   # ndda
    "nda": "ඳ",    # nda
    "mba": "ඹ",    # mba
    "shha": "ෂ",   # shha (alternative)
    
    # Numbers
    "0": "0",
    "1": "1", 
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
}
unicode_to_label = {v: k for k, v in label_to_unicode.items()}


# ---------------------------
# Utility functions
# ---------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def is_allowed_filename(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXT


# ---------------------------
# Synthetic dataset generation (typed characters)
# ---------------------------
def generate_synthetic_dataset(out_dir: str,
                               labels: List[str],
                               font_path: str,
                               samples_per_label: int = 200,
                               image_size: int = 128):
    """
    Renders typed characters from a TTF font to create training images.
    out_dir/
        label1/
            img_0001.png
        label2/
            ...
    Args:
        out_dir: base output directory
        labels: list of label strings (label names; will be used as folder names)
        font_path: path to a Sinhala-capable TTF font
        samples_per_label: how many images per label
        image_size: render size (we will later resize to IMG_SIZE)
    """
    ensure_dir(out_dir)
    font = ImageFont.truetype(font_path, size=int(image_size * 0.6))
    for label in labels:
        lab_dir = os.path.join(out_dir, label)
        ensure_dir(lab_dir)
        # Get the Unicode character for this label
        unicode_char = label_to_unicode.get(label, label)
        for i in range(samples_per_label):
            img = Image.new("L", (image_size, image_size), color=255)  # white background
            draw = ImageDraw.Draw(img)
            # Slight random translations/scale for diversity
            dx = np.random.randint(-6, 6)
            dy = np.random.randint(-6, 6)
            # draw Unicode character
            try:
                draw.text((image_size // 4 + dx, image_size // 6 + dy), unicode_char, font=font, fill=0)
            except Exception as e:
                # if Unicode fails, draw label as placeholder
                draw.text((image_size // 4 + dx, image_size // 6 + dy), label, font=font, fill=0)
            # small rotation
            angle = np.random.uniform(-5, 5)
            img = img.rotate(angle, expand=False, fillcolor=255)
            # save
            fname = os.path.join(lab_dir, f"{label}_{i:04d}.png")
            img.save(fname)
    print(f"Synthetic dataset created at {out_dir}")


# ---------------------------
# Data loading + preprocessing
# ---------------------------
def load_image_paths_and_labels(dataset_dir: str) -> Tuple[List[str], List[str]]:
    """
    Expects dataset_dir/<label>/*.png
    Returns lists of image paths and corresponding labels (folder names)
    """
    img_paths = []
    labels = []
    for lab in sorted(os.listdir(dataset_dir)):
        labp = os.path.join(dataset_dir, lab)
        if not os.path.isdir(labp):
            continue
        for f in os.listdir(labp):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                img_paths.append(os.path.join(labp, f))
                labels.append(lab)
    return img_paths, labels


def preprocess_image_for_model(img: np.ndarray, img_size=IMG_SIZE) -> np.ndarray:
    """
    Input: grayscale (H,W) (uint8) or color (H,W,3)
    Output: normalized (img_size,img_size,1) float32
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    # resize while keeping aspect by padding
    target = img_size
    # threshold/normalize
    img = cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


def make_dataset_arrays(dataset_dir: str, img_size=IMG_SIZE):
    paths, labs = load_image_paths_and_labels(dataset_dir)
    if len(paths) == 0:
        raise RuntimeError("No images found in dataset_dir. Check folders.")
    unique_labels = sorted(list({l for l in labs}))
    label_to_index = {lab: i for i, lab in enumerate(unique_labels)}
    X = np.zeros((len(paths), img_size, img_size, 1), dtype=np.float32)
    y = np.zeros((len(paths),), dtype=np.int32)
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            X[i] = preprocess_image_for_model(img, img_size)
            y[i] = label_to_index[labs[i]]
    y_cat = to_categorical(y, num_classes=len(unique_labels))
    return X, y_cat, unique_labels


# ---------------------------
# Model (CNN)
# ---------------------------
def build_model(num_classes: int, input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    """
    A small CNN architecture inspired by simple LeNet-ish networks and the paper approach.
    Tunable hyperparameters: dropout, number of filters.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.4))
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
    return model

# ---------------------------
# Training helpers
# ---------------------------
def train_model(dataset_dir: str,
                model_save_path: str = MODEL_NAME,
                epochs: int = 40,
                batch_size: int = 64,
                use_augmentation: bool = False):
    """
    Trains the CNN on the dataset at dataset_dir. Saves model and label mapping.
    """
    X, y, labels = make_dataset_arrays(dataset_dir, IMG_SIZE)
    num_classes = len(labels)
    model = build_model(num_classes)
    print(model.summary())
    
    # Simple training without augmentation for now
    model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    
    model.save(model_save_path)
    # save label mapping
    labels_path = model_save_path + ".labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)
    print(f"Model saved to {model_save_path}; labels saved to {labels_path}")


def load_model_and_labels(model_path: str = MODEL_NAME):
    """Load model and labels, trying complete model first."""
    # Try complete model first, then essential, then fallback to original
    model_paths = ['sinhala_complete_cnn.h5', 'sinhala_essential_cnn.h5', 'sinhala_cnn.h5', model_path]
    
    for path in model_paths:
        try:
            if os.path.exists(path):
                print(f"Loading model: {path}")
                model = models.load_model(path)
                labels_path = path + ".labels.json"
                
                if os.path.exists(labels_path):
                    with open(labels_path, "r", encoding="utf-8") as f:
                        labels = json.load(f)
                    print(f"✅ Loaded {len(labels)} characters from {path}")
                    print(f"📝 Sample characters: {labels[:10]}")
                    return model, labels
                else:
                    print(f"⚠️ Labels file not found: {labels_path}")
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")
            continue
    
    # Fallback to original implementation if all fails
    raise FileNotFoundError("No valid model found")

# ---------------------------
# Segmentation (from a scanned/photographed line or block)
# ---------------------------
def segment_characters_from_image(image: np.ndarray,
                                  min_area=80,
                                  debug=False) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Simple segmentation pipeline that finds connected components / contours of characters.
    Returns list of tuples: (cropped_image_gray, bbox (x,y,w,h))
    For complicated images (touching glyphs), more advanced segmentation is required.
    """
    # convert to gray
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    # binarize (adaptive threshold is robust to illumination)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 10)
    # morphological operations to connect parts of same glyph
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    # find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < min_area or w < 5 or h < 5:
            continue
        boxes.append((x, y, w, h))
    # optionally sort left-to-right, top-to-bottom
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    results = []
    for (x, y, w, h) in boxes:
        pad = 4
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(gray.shape[1], x + w + pad)
        y1 = min(gray.shape[0], y + h + pad)
        crop = gray[y0:y1, x0:x1]
        results.append((crop, (x0, y0, x1 - x0, y1 - y0)))
    if debug:
        print(f"Found {len(results)} candidate char boxes")
    return results


# ---------------------------
# Inference
# ---------------------------
def predict_on_image(image: np.ndarray,
                     model,
                     label_list: List[str],
                     top_k=1) -> List[Dict]:
    """
    Given an image (could be whole line or block), segment characters and predict using model.
    Returns list: [{'char':unicode, 'prob':float, 'bbox':(x,y,w,h)}]
    """
    # Determine the expected input size from the model
    try:
        # Try different ways to get input shape
        if hasattr(model, 'layers') and len(model.layers) > 0:
            # For Sequential models, get from first layer
            first_layer = model.layers[0]
            if hasattr(first_layer, 'input_spec') and first_layer.input_spec:
                input_shape = first_layer.input_spec.shape
            elif hasattr(first_layer, 'input_shape'):
                input_shape = first_layer.input_shape
            else:
                # Fallback: try calling the model with a dummy input to build it
                dummy_input = np.zeros((1, 64, 64, 1), dtype=np.float32)
                try:
                    _ = model(dummy_input, training=False)
                    input_shape = model.input.shape
                except:
                    # Final fallback
                    input_shape = (None, 64, 64, 1)  # Assume 64x64 for comprehensive model
        else:
            input_shape = (None, 64, 64, 1)  # Default fallback
        
        expected_img_size = input_shape[1] if input_shape[1] is not None else 64
        print(f"Model expects input size: {expected_img_size}x{expected_img_size}")
    except Exception as e:
        print(f"Could not determine input shape, using default 64x64: {e}")
        expected_img_size = 64  # Default for comprehensive model
    
    segs = segment_characters_from_image(image)
    results = []
    for crop, bbox in segs:
        xcrop = preprocess_image_for_model(crop, img_size=expected_img_size)
        xcrop = np.expand_dims(xcrop, axis=0)  # batch dim
        preds = model.predict(xcrop)
        idx = np.argmax(preds[0])
        prob = float(preds[0][idx])
        label = label_list[idx]
        
        # Since the model was trained with Sinhala characters as labels,
        # the label itself is already the Unicode character
        # No need to map through label_to_unicode
        unicode_char = label
        
        results.append({"char": unicode_char, "prob": prob, "bbox": bbox, "label": label})
    # sort by bbox x coordinate
    results = sorted(results, key=lambda r: r["bbox"][0])
    return results

# ---------------------------
# Flask app for inference
# ---------------------------
def create_app(model_path: str = MODEL_NAME):
    app = Flask(__name__)
    model, labels = None, None

    @app.before_request
    def load_model_once():
        nonlocal model, labels
        if model is None:
            model_local, labels_local = load_model_and_labels(model_path)
            model, labels = model_local, labels_local
            print("Model & labels loaded for Flask app.")

    @app.route("/", methods=["GET"])
    def home():
        return render_template("index.html")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict_route():
        nonlocal model, labels
        if 'file' not in request.files:
            return jsonify({"error": "no file part"}), 400
        file = request.files['file']
        if file.filename == '' or file.filename is None:
            return jsonify({"error": "no selected file"}), 400
        filename = secure_filename(file.filename)
        if not is_allowed_filename(filename):
            return jsonify({"error": "file type not allowed"}), 400
        in_memory = file.read()
        nparr = np.frombuffer(in_memory, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "could not decode image"}), 400
        if model is None or labels is None:
            return jsonify({"error": "model not loaded"}), 500
        results = predict_on_image(img, model, labels)
        
        # Create annotated image with bounding boxes and predicted characters
        annotated = img.copy()
        for r in results:
            x, y, w, h = r["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, r["char"], (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Convert annotated image to base64 for web display
        _, buf = cv2.imencode('.png', annotated)
        ann_base64 = base64.b64encode(buf).decode('utf-8')
        
        # build response
        return jsonify({
            "predictions": results,
            "annotated_image": ann_base64
        })

    return app


# ---------------------------
# CLI-like helpers
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sinhala OCR training/inference helper")
    parser.add_argument("--generate", nargs=3, metavar=("OUTDIR", "FONT_PATH", "LABELS_CSV"),
                        help="generate synthetic dataset; LABELS_CSV is a newline-separated file of labels (Unicode characters)")
    parser.add_argument("--train", metavar="DATASET_DIR", help="train model on dataset dir")
    parser.add_argument("--serve", action="store_true", help="start Flask server (requires saved model)")
    parser.add_argument("--model", default=MODEL_NAME, help="model path (h5)")
    args = parser.parse_args()

    if args.generate:
        outdir, fontpath, labels_csv = args.generate
        with open(labels_csv, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        generate_synthetic_dataset(outdir, labels, fontpath, samples_per_label=300, image_size=128)

    elif args.train:
        train_model(args.train, model_save_path=args.model, epochs=40, batch_size=64)

    elif args.serve:
        app = create_app(args.model)
        print("Starting Flask app on http://0.0.0.0:5000")
        app.run(host="0.0.0.0", port=5000)
    else:
        parser.print_help()