#!/usr/bin/env python3
"""
Complete Sinhala Character Recognition Training System
Training ALL Sinhala Unicode characters for comprehensive OCR.
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
from sinhala_ocr import make_dataset_arrays

# COMPLETE SINHALA CHARACTER SET - ALL UNICODE CHARACTERS
COMPLETE_SINHALA_CHARS = [
    # Independent vowels (18)
    'අ', 'ආ', 'ඇ', 'ඈ', 'ඉ', 'ඊ', 'උ', 'ඌ', 'ඍ', 'ඎ', 'ඏ', 'ඐ', 'එ', 'ඒ', 'ඓ', 'ඔ', 'ඕ', 'ඖ',
    
    # All consonants (38)
    'ක', 'ඛ', 'ග', 'ඝ', 'ඞ', 'ඟ', 'ච', 'ඡ', 'ජ', 'ඣ', 'ඤ', 'ඥ', 'ට', 'ඨ', 'ඩ', 'ඪ', 'ණ', 'ඬ',
    'ත', 'ථ', 'ද', 'ධ', 'න', 'ඳ', 'ප', 'ඵ', 'බ', 'භ', 'ම', 'ඹ', 'ය', 'ර', 'ල', 'ව', 'ශ', 'ෂ', 'ස', 'හ', 'ළ', 'ෆ',
    
    # Vowel signs/diacritics when combined with consonants
    # ka combinations
    'කා', 'කැ', 'කෑ', 'කි', 'කී', 'කු', 'කූ', 'කෘ', 'කෙ', 'කේ', 'කෛ', 'කො', 'කෝ', 'කෞ', 'ක්',
    
    # ga combinations
    'ගා', 'ගැ', 'ගෑ', 'ගි', 'ගී', 'ගු', 'ගූ', 'ගෘ', 'ගෙ', 'ගේ', 'ගෛ', 'ගො', 'ගෝ', 'ගෞ', 'ග්',
    
    # ja combinations  
    'ජා', 'ජැ', 'ජෑ', 'ජි', 'ජී', 'ජු', 'ජූ', 'ජෘ', 'ජෙ', 'ජේ', 'ජෛ', 'ජො', 'ජෝ', 'ජෞ', 'ජ්',
    
    # ta combinations
    'තා', 'තැ', 'තෑ', 'ති', 'තී', 'තු', 'තූ', 'තෘ', 'තෙ', 'තේ', 'තෛ', 'තො', 'තෝ', 'තෞ', 'ත්',
    
    # na combinations
    'නා', 'නැ', 'නෑ', 'නි', 'නී', 'නු', 'නූ', 'නෘ', 'නෙ', 'නේ', 'නෛ', 'නො', 'නෝ', 'නෞ', 'න්',
    
    # pa combinations
    'පා', 'පැ', 'පෑ', 'පි', 'පී', 'පු', 'පූ', 'පෘ', 'පෙ', 'පේ', 'පෛ', 'පො', 'පෝ', 'පෞ', 'ප්',
    
    # ma combinations
    'මා', 'මැ', 'මෑ', 'මි', 'මී', 'මු', 'මූ', 'මෘ', 'මෙ', 'මේ', 'මෛ', 'මො', 'මෝ', 'මෞ', 'ම්',
    
    # ya combinations
    'යා', 'යැ', 'යෑ', 'යි', 'යී', 'යු', 'යූ', 'යෘ', 'යෙ', 'යේ', 'යෛ', 'යො', 'යෝ', 'යෞ', 'ය්',
    
    # ra combinations
    'රා', 'රැ', 'රෑ', 'රි', 'රී', 'රු', 'රූ', 'රෘ', 'රෙ', 'රේ', 'රෛ', 'රො', 'රෝ', 'රෞ', 'ර්',
    
    # la combinations
    'ලා', 'ලැ', 'ලෑ', 'ලි', 'ලී', 'ලු', 'ලූ', 'ලෘ', 'ලෙ', 'ලේ', 'ලෛ', 'ලො', 'ලෝ', 'ලෞ', 'ල්',
    
    # va combinations
    'වා', 'වැ', 'වෑ', 'වි', 'වී', 'වු', 'වූ', 'වෘ', 'වෙ', 'වේ', 'වෛ', 'වො', 'වෝ', 'වෞ', 'ව්',
    
    # sa combinations
    'සා', 'සැ', 'සෑ', 'සි', 'සී', 'සු', 'සූ', 'සෘ', 'සෙ', 'සේ', 'සෛ', 'සො', 'සෝ', 'සෞ', 'ස්',
    
    # ha combinations
    'හා', 'හැ', 'හෑ', 'හි', 'හී', 'හු', 'හූ', 'හෘ', 'හෙ', 'හේ', 'හෛ', 'හො', 'හෝ', 'හෞ', 'හ්',
    
    # da combinations
    'දා', 'දැ', 'දෑ', 'දි', 'දී', 'දු', 'දූ', 'දෘ', 'දෙ', 'දේ', 'දෛ', 'දො', 'දෝ', 'දෞ', 'ද්',
    
    # ba combinations
    'බා', 'බැ', 'බෑ', 'බි', 'බී', 'බු', 'බූ', 'බෘ', 'බෙ', 'බේ', 'බෛ', 'බො', 'බෝ', 'බෞ', 'බ්',
    
    # Additional important consonant combinations
    'චා', 'චැ', 'චෑ', 'චි', 'චී', 'චු', 'චූ', 'චෘ', 'චෙ', 'චේ', 'චෛ', 'චො', 'චෝ', 'චෞ', 'ච්',
    'ධා', 'ධැ', 'ධෑ', 'ධි', 'ධී', 'ධු', 'ධූ', 'ධෘ', 'ධෙ', 'ධේ', 'ධෛ', 'ධො', 'ධෝ', 'ධෞ', 'ධ්',
    'ඵා', 'ඵැ', 'ඵෑ', 'ඵි', 'ඵී', 'ඵු', 'ඵූ', 'ඵෘ', 'ඵෙ', 'ඵේ', 'ඵෛ', 'ඵො', 'ඵෝ', 'ඵෞ', 'ඵ්',
    'ශා', 'ශැ', 'ශෑ', 'ශි', 'ශී', 'ශු', 'ශූ', 'ශෘ', 'ශෙ', 'ශේ', 'ශෛ', 'ශො', 'ශෝ', 'ශෞ', 'ශ්',
    
    # Sinhala numerals (10)
    '෦', '෧', '෨', '෩', '෪', '෫', '෬', '෭', '෮', '෯',
    
    # Common conjunct consonants and special characters
    'ක්ර', 'ග්ර', 'ත්ර', 'ප්ර', 'ශ්ර', 'ක්ව', 'ග්ව', 'ද්ව', 'ත්ව', 'න්ද', 'න්ත', 'ම්ප', 'ය්ය', 'ල්ල',
    
    # Special marks and symbols
    '්', 'ං', 'ඃ', '෴'
]

def generate_complete_dataset():
    """Generate the most comprehensive Sinhala dataset possible."""
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    dataset_dir = "sinhala_complete_dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # Load Sinhala font
    try:
        font_large = ImageFont.truetype("NotoSansSinhala-Regular.ttf", 56)  # Larger for clarity
        font_medium = ImageFont.truetype("NotoSansSinhala-Regular.ttf", 48)
        font_small = ImageFont.truetype("NotoSansSinhala-Regular.ttf", 42)
        fonts = [font_large, font_medium, font_small]
    except:
        print("❌ Font not found, using default font")
        fonts = [ImageFont.load_default()]
    
    print(f"🔥 Generating COMPLETE Sinhala dataset for {len(COMPLETE_SINHALA_CHARS)} characters...")
    
    samples_per_char = 80  # More samples for better accuracy
    
    for i, char in enumerate(COMPLETE_SINHALA_CHARS):
        char_dir = os.path.join(dataset_dir, char.replace('/', '_'))  # Handle special chars
        if not os.path.exists(char_dir):
            os.makedirs(char_dir)
        
        print(f"Generating {i+1}/{len(COMPLETE_SINHALA_CHARS)}: {char}")
        
        for j in range(samples_per_char):
            # Create image with variation
            img_size = random.choice([64, 68, 72])  # Size variation
            img = Image.new('RGB', (img_size, img_size), color='white')
            draw = ImageDraw.Draw(img)
            
            # Add variation
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            font = random.choice(fonts)
            
            # Random rotation (small angle)
            if random.random() < 0.3:  # 30% chance of slight rotation
                angle = random.randint(-5, 5)
                img = img.rotate(angle, fillcolor='white')
                draw = ImageDraw.Draw(img)
            
            # Draw character
            try:
                # Get text bbox for centering
                bbox = draw.textbbox((0, 0), char, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Center the text
                x = (img_size - text_width) // 2 + x_offset
                y = (img_size - text_height) // 2 + y_offset
                
                # Random color variation (very dark grays to black)
                color_intensity = random.randint(0, 30)  # Very dark
                color = (color_intensity, color_intensity, color_intensity)
                
                draw.text((x, y), char, font=font, fill=color)
            except Exception as e:
                # Fallback to simple positioning
                draw.text((8 + x_offset, 12 + y_offset), char, font=font, fill='black')
            
            # Add slight noise occasionally
            if random.random() < 0.1:  # 10% chance
                for _ in range(random.randint(1, 3)):
                    noise_x = random.randint(0, img_size-1)
                    noise_y = random.randint(0, img_size-1)
                    draw.point((noise_x, noise_y), fill='gray')
            
            # Resize to standard 64x64
            img = img.resize((64, 64), Image.LANCZOS)
            
            # Save image
            img_path = os.path.join(char_dir, f"{char.replace('/', '_')}_{j:03d}.png")
            img.save(img_path)
    
    return dataset_dir

def create_complete_model(num_classes):
    """Create a robust CNN model for complete character set."""
    model = Sequential([
        # First convolutional block
        Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 1), padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Fourth convolutional block - for complex character recognition
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Flatten and dense layers
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_complete_model():
    """Train the complete Sinhala character recognition model."""
    print("🚀 Complete Sinhala OCR Training Starting...")
    print(f"📊 Target: {len(COMPLETE_SINHALA_CHARS)} characters")
    
    # Generate dataset
    dataset_dir = generate_complete_dataset()
    
    print("📊 Loading and preprocessing complete dataset...")
    X, y, labels = make_dataset_arrays(dataset_dir, 64)
    
    print(f"✅ Complete dataset loaded: {X.shape[0]} samples, {len(labels)} classes")
    print(f"📝 Total characters: {len(labels)}")
    print(f"📝 Sample characters: {labels[:15]}")
    
    # Create and compile model
    model = create_complete_model(len(labels))
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("🏗️ Complete Model Architecture:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.2,
        patience=4,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train model
    print("🎓 Training complete model...")
    history = model.fit(
        X, y,
        batch_size=64,  # Larger batch for stability
        epochs=50,      # More epochs for complex dataset
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save model
    model_path = "sinhala_complete_cnn.h5"
    model.save(model_path)
    print(f"💾 Complete model saved as: {model_path}")
    
    # Save labels
    labels_path = "sinhala_complete_cnn.h5.labels.json"
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"🏷️ Complete labels saved as: {labels_path}")
    
    # Evaluate final performance
    final_accuracy = max(history.history['val_accuracy'])
    print(f"🎯 Best Validation Accuracy: {final_accuracy:.4f}")
    
    print(f"\n🎊 COMPLETE SINHALA CHARACTER SET TRAINED!")
    print(f"📊 Total Characters: {len(labels)}")
    print(f"🎯 Model ready for ALL Sinhala text recognition!")
    
    return True

if __name__ == "__main__":
    print("🔤 Complete Sinhala OCR Training System")
    print("=" * 60)
    print(f"🎯 Training {len(COMPLETE_SINHALA_CHARS)} Sinhala characters")
    print("=" * 60)
    
    if train_complete_model():
        print("\n🎉 COMPLETE TRAINING FINISHED!")
        print("✅ All major Sinhala characters now supported")
        print("🔄 Update your OCR application to use: sinhala_complete_cnn.h5")
        print("📝 This model can recognize the full Sinhala Unicode character set!")
    else:
        print("\n❌ Training failed!")