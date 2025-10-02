# Complete mapping for 55 Sinhala characters as used in research papers
# This maps the paper's label naming convention to actual Unicode Sinhala characters

COMPLETE_SINHALA_MAPPING = {
    # Vowels
    "a": "අ",      # U+0D85
    "aa": "ආ",     # U+0D86  
    "ae": "ඇ",     # U+0D87
    "aae": "ඈ",    # U+0D88
    "i": "ඉ",      # U+0D89
    "ii": "ඊ",     # U+0D8A
    "u": "උ",      # U+0D8B
    "uu": "ඌ",     # U+0D8C
    "ri": "ඍ",     # U+0D8D
    "rii": "ඎ",    # U+0D8E
    "lu": "ඏ",     # U+0D8F
    "luu": "ඐ",    # U+0D90
    "e": "එ",      # U+0D91
    "ee": "ඒ",     # U+0D92
    "ai": "ඓ",     # U+0D93
    "o": "ඔ",      # U+0D94
    "oo": "ඕ",     # U+0D95
    "au": "ඖ",     # U+0D96
    
    # Consonants
    "ka": "ක",     # U+0D9A
    "kha": "ඛ",    # U+0D9B
    "ga": "ග",     # U+0D9C
    "gha": "ඝ",    # U+0D9D
    "nga": "ඞ",    # U+0D9E
    "ca": "ච",     # U+0D9F
    "cha": "ඡ",    # U+0DA0
    "ja": "ජ",     # U+0DA1
    "jha": "ඣ",    # U+0DA2
    "nya": "ඤ",    # U+0DA3
    "jnya": "ඥ",   # U+0DA4
    "tta": "ට",    # U+0DA5
    "ttha": "ඨ",   # U+0DA6
    "dda": "ඩ",    # U+0DA7
    "ddha": "ඪ",   # U+0DA8
    "nna": "ණ",    # U+0DA9
    "ndda": "ඬ",   # U+0DAA
    "ta": "ත",     # U+0DAB
    "tha": "ථ",    # U+0DAC
    "da": "ද",     # U+0DAD
    "dha": "ධ",    # U+0DAE
    "na": "න",     # U+0DAF
    "nda": "ඳ",    # U+0DB0
    "pa": "ප",     # U+0DB1
    "pha": "ඵ",    # U+0DB2
    "ba": "බ",     # U+0DB3
    "bha": "භ",    # U+0DB4
    "ma": "ම",     # U+0DB5
    "mba": "ඹ",    # U+0DB6
    "ya": "ය",     # U+0DB7
    "ra": "ර",     # U+0DB8
    "la": "ල",     # U+0DB9
    "wa": "ව",     # U+0DBA
    "sha": "ශ",    # U+0DB7
    "shha": "ෂ",   # U+0DB8
    "sa": "ස",     # U+0DB9
    "ha": "හ",     # U+0DBA
    "lla": "ළ",    # U+0DBB
    "fa": "ෆ",     # U+0DBC
}

# Alternative mapping with numeric suffixes as used in some papers
NUMERIC_SINHALA_MAPPING = {
    "a1": "අ", "a2": "ආ", "a3": "ඇ", "a4": "ඈ", "a5": "ඉ",
    "a6": "ඊ", "a7": "උ", "a8": "ඌ", "a9": "ඍ", "a10": "ඎ",
    "a11": "ඏ", "a12": "ඐ", "a13": "එ", "a14": "ඒ", "a15": "ඓ",
    "a16": "ඔ", "a17": "ඕ", "a18": "ඖ",
    
    "k1": "ක", "k2": "ඛ", "k3": "ග", "k4": "ඝ", "k5": "ඞ",
    "c1": "ච", "c2": "ඡ", "c3": "ජ", "c4": "ඣ", "c5": "ඤ", "c6": "ඥ",
    "t1": "ට", "t2": "ඨ", "t3": "ඩ", "t4": "ඪ", "t5": "ණ", "t6": "ඬ",
    "t7": "ත", "t8": "ථ", "t9": "ද", "t10": "ධ", "t11": "න", "t12": "ඳ",
    "p1": "ප", "p2": "ඵ", "p3": "බ", "p4": "භ", "p5": "ම", "p6": "ඹ",
    "y1": "ය", "r1": "ර", "l1": "ල", "w1": "ව",
    "s1": "ශ", "s2": "ෂ", "s3": "ස", "h1": "හ", "l2": "ළ", "f1": "ෆ"
}

def get_mapping_choice():
    """Let user choose which mapping to use"""
    print("Choose mapping style:")
    print("1. Descriptive (ka, ga, ma, etc.) - Recommended")
    print("2. Numeric (a1, k1, t1, etc.) - Research paper style")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        return NUMERIC_SINHALA_MAPPING
    else:
        return COMPLETE_SINHALA_MAPPING

if __name__ == "__main__":
    # Display the complete mapping
    print("=== COMPLETE SINHALA CHARACTER MAPPING ===\n")
    
    print("Descriptive mapping (55 characters):")
    for label, char in COMPLETE_SINHALA_MAPPING.items():
        print(f"  {label:8} -> {char}")
    
    print(f"\nTotal characters: {len(COMPLETE_SINHALA_MAPPING)}")
    print("This mapping covers the standard Sinhala alphabet used in OCR research.")