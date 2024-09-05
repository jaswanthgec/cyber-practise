import streamlit as st
from Crypto.Cipher import AES, DES, DES3, Blowfish, ChaCha20, CAST, ARC2
from Crypto.Util.Padding import pad, unpad
import base64

# Adding top buttons for documentation links
st.markdown("""
    <style>
    .top-buttons {
        display: flex;
        justify-content: space-between;
        margin-bottom: 30px;
    }
    .top-buttons a {
        background-color: #66d9ef;
        color: white;
        text-align: center;
        padding: 10px 20px;
        text-decoration: none;
        border-radius: 5px;
        font-size: 16px;
        transition: 0.3s;
    }
    .top-buttons a:hover {
        background-color: #ff6666;
    }
    </style>
    <div class="top-buttons">
        <a href="encryption-doc.html" target="_blank">Encryption Documentation</a>
        <a href="main-doc.html" target="_blank">Main Documentation</a>
        <a href="decryption-doc.html" target="_blank">Decryption Documentation</a>
    </div>
    """, unsafe_allow_html=True)

# Implementations of basic ciphers
def caesar_cipher(text, shift, mode='encrypt'):
    result = ""
    shift = shift % 26
    for i in text:
        if i.isalpha():
            base = ord('A') if i.isupper() else ord('a')
            shift_value = (ord(i) - base + shift) % 26 if mode == 'encrypt' else (ord(i) - base - shift) % 26
            result += chr(base + shift_value)
        else:
            result += i
    return result

def atbash_cipher(text):
    result = ""
    for char in text:
        if char.isupper():
            result += chr(ord('Z') - (ord(char) - ord('A')))
        elif char.islower():
            result += chr(ord('z') - (ord(char) - ord('a')))
        else:
            result += char
    return result

def xor_cipher(text, key):
    return ''.join(chr(ord(c) ^ ord(key)) for c, key in zip(text, key * (len(text) // len(key) + 1)))

# Symmetric encryption/decryption
def symmetric_encrypt(algorithm, key, plaintext):
    try:
        if algorithm == "AES":
            cipher = AES.new(key, AES.MODE_ECB)
            ciphertext = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
            return base64.b64encode(ciphertext).decode()

        elif algorithm == "DES":
            cipher = DES.new(key[:8], DES.MODE_ECB)
            ciphertext = cipher.encrypt(pad(plaintext.encode(), DES.block_size))
            return base64.b64encode(ciphertext).decode()

        elif algorithm == "3DES":
            cipher = DES3.new(key[:24], DES3.MODE_ECB)  # 3DES requires a 16 or 24-byte key
            ciphertext = cipher.encrypt(pad(plaintext.encode(), DES3.block_size))
            return base64.b64encode(ciphertext).decode()

        elif algorithm == "Blowfish":
            cipher = Blowfish.new(key[:16], Blowfish.MODE_ECB)
            ciphertext = cipher.encrypt(pad(plaintext.encode(), Blowfish.block_size))
            return base64.b64encode(ciphertext).decode()

        elif algorithm == "ChaCha20":
            cipher = ChaCha20.new(key=key[:32])
            ciphertext = cipher.encrypt(plaintext.encode())
            return base64.b64encode(ciphertext).decode()

        elif algorithm == "CAST5":
            cipher = CAST.new(key[:16], CAST.MODE_ECB)
            ciphertext = cipher.encrypt(pad(plaintext.encode(), CAST.block_size))
            return base64.b64encode(ciphertext).decode()

        elif algorithm == "RC2":
            cipher = ARC2.new(key[:16], ARC2.MODE_ECB)
            ciphertext = cipher.encrypt(pad(plaintext.encode(), ARC2.block_size))
            return base64.b64encode(ciphertext).decode()

    except Exception as e:
        st.error(f"Encryption failed: {str(e)}")

def symmetric_decrypt(algorithm, key, ciphertext):
    try:
        ciphertext = base64.b64decode(ciphertext)
        if algorithm == "AES":
            cipher = AES.new(key, AES.MODE_ECB)
            plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size).decode()
            return plaintext

        elif algorithm == "DES":
            cipher = DES.new(key[:8], DES.MODE_ECB)
            plaintext = unpad(cipher.decrypt(ciphertext), DES.block_size).decode()
            return plaintext

        elif algorithm == "3DES":
            cipher = DES3.new(key[:24], DES3.MODE_ECB)  # 3DES requires a 16 or 24-byte key
            plaintext = unpad(cipher.decrypt(ciphertext), DES3.block_size).decode()
            return plaintext

        elif algorithm == "Blowfish":
            cipher = Blowfish.new(key[:16], Blowfish.MODE_ECB)
            plaintext = unpad(cipher.decrypt(ciphertext), Blowfish.block_size).decode()
            return plaintext

        elif algorithm == "ChaCha20":
            cipher = ChaCha20.new(key=key[:32])
            plaintext = cipher.decrypt(ciphertext).decode()
            return plaintext

        elif algorithm == "CAST5":
            cipher = CAST.new(key[:16], CAST.MODE_ECB)
            plaintext = unpad(cipher.decrypt(ciphertext), CAST.block_size).decode()
            return plaintext

        elif algorithm == "RC2":
            cipher = ARC2.new(key[:16], ARC2.MODE_ECB)
            plaintext = unpad(cipher.decrypt(ciphertext), ARC2.block_size).decode()
            return plaintext

    except Exception as e:
        st.error(f"Decryption failed: {str(e)}")

# Cybersecurity-themed styling
st.markdown("""
    <style>
    body {
        background-color: #1d1f21;
        color: #f8f8f2;
    }
    .stButton>button {
        background-color: #f92672;
        color: white;
        border-radius: 10px;
        width: 150px;
        height: 40px;
    }
    h1, h2, h3, p {
        color: #66d9ef;
        font-family: 'Courier New', Courier, monospace;
    }
    input {
        background-color: #272822;
        color: #f8f8f2;
    }
    textarea {
        background-color: #272822;
        color: #f8f8f2;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
st.title("🔒 Advanced Encryption/Decryption Tool")


# Input text from user
user_input = st.text_area("Enter the text to encrypt or decrypt", "")

# Choose operation: Encryption or Decryption
operation = st.selectbox("Select Operation", ["Encryption", "Decryption"])

# Dropdown to select encryption algorithm
algorithm = st.selectbox(
    "Select Algorithm",
    ["AES", "DES", "3DES", "Blowfish", "ChaCha20", "CAST5", "RC2", "Caesar Cipher", "Atbash Cipher", "XOR Cipher"]
)

# Key Information for Symmetric Encryption
if algorithm == "AES":
    st.markdown("""
    **Key Information for AES:**
    - AES uses a key length of 16, 24, or 32 bytes.
    - AES is a highly secure algorithm, widely used in industries for data encryption.
    - Ensure the key length is correct or the encryption will fail.
    """)

elif algorithm == "DES":
    st.markdown("""
    **Key Information for DES:**
    - DES uses a key length of 8 bytes.
    - DES is a legacy algorithm, considered insecure today due to its short key length.
    """)

elif algorithm == "3DES":
    st.markdown("""
    **Key Information for 3DES:**
    - 3DES (Triple DES) uses a key length of 16 or 24 bytes.
    - Ensure the key does not have repeated blocks (e.g., '1234567812345678' is not valid).
    - It applies the DES algorithm three times to increase security.
    - Though more secure than DES, it is slower compared to AES.
    """)
    key = st.text_input(f"Enter the key for {algorithm} (16 or 24 bytes)", type="password")

    # Validate key length for 3DES
    if len(key) not in [16, 24]:
        st.error("❗ Key length for 3DES must be either 16 or 24 bytes.")
    elif key[:8] == key[8:16]:  # Avoid repeated blocks
        st.error("❗ 3DES key should not have repeated blocks like '1234567812345678'.")


elif algorithm == "Blowfish":
    st.markdown("""
    **Key Information for Blowfish:**
    - Blowfish uses a key length of up to 448 bits, but typically 16 bytes is common.
    - It is fast and free for use, making it popular in software encryption.
    """)

elif algorithm == "ChaCha20":
    st.markdown("""
    **Key Information for ChaCha20:**
    - ChaCha20 uses a 32-byte key.
    - It is designed to be faster than AES on devices without hardware acceleration.
    - It is a secure alternative to AES for high-performance use cases.
    """)

elif algorithm == "CAST5":
    st.markdown("""
    **Key Information for CAST5:**
    - CAST5 uses a key length of 5 to 16 bytes.
    - It is used in various encryption standards and protocols.
    """)

elif algorithm == "RC2":
    st.markdown("""
    **Key Information for RC2:**
    - RC2 uses a key length of 8 to 16 bytes.
    - It is a block cipher that was designed to be a replacement for DES.
    """)

# Key input
if algorithm in ["AES", "DES", "3DES", "Blowfish", "ChaCha20", "CAST5", "RC2"]:
    key = st.text_input(f"Enter the key for {algorithm} (length varies per algorithm)", type="password")

elif algorithm == "Caesar Cipher":
    shift = st.number_input("Enter the shift value for Caesar Cipher (integer)", min_value=1, max_value=25)

elif algorithm == "XOR Cipher":
    st.markdown("""
    **Key Information for XOR Cipher:**
    - XOR cipher works by applying the XOR operation between the plaintext and key.
    - The key should be at least as long as the plaintext for maximum security.
    """)
    key = st.text_input(f"Enter the key for XOR Cipher", type="password")

# Process based on operation and algorithm
if st.button("Run"):
    if operation == "Encryption":
        if algorithm in ["AES", "DES", "3DES", "Blowfish", "ChaCha20", "CAST5", "RC2"]:
            if len(key) > 0:
                encrypted_text = symmetric_encrypt(algorithm, key.encode(), user_input)
                st.success(f"🔐 Encrypted Text: {encrypted_text}")
            else:
                st.error("❗ Key is required for encryption.")
        elif algorithm == "Caesar Cipher":
            encrypted_text = caesar_cipher(user_input, int(shift), mode='encrypt')
            st.success(f"🔐 Encrypted Text: {encrypted_text}")
        elif algorithm == "Atbash Cipher":
            encrypted_text = atbash_cipher(user_input)
            st.success(f"🔐 Encrypted Text: {encrypted_text}")
        elif algorithm == "XOR Cipher":
            if len(key) > 0:
                encrypted_text = xor_cipher(user_input, key)
                st.success(f"🔐 Encrypted Text: {encrypted_text}")
            else:
                st.error("❗ Key is required for XOR encryption.")

    elif operation == "Decryption":
        if algorithm in ["AES", "DES", "3DES", "Blowfish", "ChaCha20", "CAST5", "RC2"]:
            if len(key) > 0:
                try:
                    decrypted_text = symmetric_decrypt(algorithm, key.encode(), user_input)
                    st.success(f"🔓 Decrypted Text: {decrypted_text}")
                except Exception as e:
                    st.error(f"❗ Decryption failed: {str(e)}")
            else:
                st.error("❗ Key is required for decryption.")
        elif algorithm == "Caesar Cipher":
            decrypted_text = caesar_cipher(user_input, int(shift), mode='decrypt')
            st.success(f"🔓 Decrypted Text: {decrypted_text}")
        elif algorithm == "Atbash Cipher":
            decrypted_text = atbash_cipher(user_input)
            st.success(f"🔓 Decrypted Text: {decrypted_text}")
        elif algorithm == "XOR Cipher":
            if len(key) > 0:
                decrypted_text = xor_cipher(user_input, key)
                st.success(f"🔓 Decrypted Text: {decrypted_text}")
            else:
                st.error("❗ Key is required for XOR decryption.")

st.markdown("""
    <style>
    .footer {
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #1d1f21;
        color: #f8f8f2;
        text-align: center;
        padding: 10px 0;
        font-family: 'Courier New', Courier, monospace;
        font-size: 14px;
    }
    .footer a {
        color: #66d9ef;
        text-decoration: none;
    }
    .footer a:hover {
        color: #ff6666;
    }
    </style>

    <div class="footer">
        <p>Developed by <strong>Jaswanth Kollipara</strong></p>
        <p>
            <a href="https://github.com/jaswanthgec" target="_blank">GitHub</a> |
            <a href="https://www.linkedin.com/in/jaswanthkollipara/" target="_blank">LinkedIn</a> |
            <a href="https://sites.google.com/view/jaswanth-kollipara" target="_blank">Portfolio</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
