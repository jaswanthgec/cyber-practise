import streamlit as st
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64 
import numpy as np

# --- Encryption Algorithm Implementations ---

def caesar_cipher(text, shift):
    result = ''
    for char in text:
        if char.isalpha():
            start = ord('a') if char.islower() else ord('A')
            shifted_char = chr((ord(char) - start + shift) % 26 + start)
        elif char.isdigit():
            shifted_char = str((int(char) + shift) % 10)
        else:
            shifted_char = char
        result += shifted_char
    return result

def monoalphabetic_cipher(text, key):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    mapping = str.maketrans(alphabet, key)
    return text.translate(mapping)

def playfair_cipher(text, key):
    # ... (Implementation of Playfair Cipher) ...
    text = text.upper().replace("J", "I")
    key = key.upper().replace("J", "I")
    matrix = generate_matrix(key)
    ciphertext = encrypt_playfair(text, matrix)
    return ciphertext

def hill_cipher(text, key_matrix):
    # ... (Implementation of Hill Cipher) ...
    text = text.upper().replace(" ", "")
    key_matrix = np.array(key_matrix)
    ciphertext = encrypt_hill(text, key_matrix)
    return ciphertext

def polyalphabetic_cipher(text, key):
    # ... (Implementation of Polyalphabetic Cipher - Vigenere Example) ...
    text = text.upper()
    key = key.upper()
    key_len = len(key)
    ciphertext = ''
    for i, char in enumerate(text):
        if char.isalpha():
            shift = ord(key[i % key_len]) - ord('A')
            shifted_char = chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
        else:
            shifted_char = char
        ciphertext += shifted_char
    return ciphertext

def one_time_pad(text, key):
    # ... (Implementation of One-Time Pad) ...
    if len(text) != len(key):
        return "Error: Key length must match plaintext length for One-Time Pad."
    text = text.upper()
    key = key.upper()
    ciphertext = ''.join(chr((ord(t) + ord(k) - 2 * ord('A')) % 26 + ord('A')) 
                         for t, k in zip(text, key))
    return ciphertext

def rail_fence(text, rails):
    # ... (Implementation of Rail Fence Cipher) ...
    fence = [['' for i in range(len(text))] for j in range(rails)]
    direction = 1
    row, col = 0, 0
    for char in text:
        fence[row][col] = char
        col += 1
        if row + direction == rails or row + direction < 0:
            direction *= -1
        row += direction
    ciphertext = ''.join(char for row in fence for char in row if char)
    return ciphertext

def row_column_transposition(text, key):
    # ... (Implementation of Row Column Transposition) ...
    order = sorted(range(len(key)), key=lambda k: key[k])
    ciphertext = ''
    for i in order:
        col = i
        while col < len(text):
            ciphertext += text[col]
            col += len(key)
    return ciphertext

def aes_encrypt(plaintext, key):
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    padded_plaintext = padding_plaintext(plaintext)
    ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
    return base64.b64encode(ciphertext).decode('utf-8')

def aes_decrypt(ciphertext, key):
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()
    decoded_ciphertext = base64.b64decode(ciphertext)
    plaintext_padded = decryptor.update(decoded_ciphertext) + decryptor.finalize()
    plaintext = unpadding_plaintext(plaintext_padded)
    return plaintext.decode('utf-8')

def des_encrypt(plaintext, key):
    # ... (Implementation of DES encryption) ...
    cipher = Cipher(algorithms.TripleDES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    padded_plaintext = padding_plaintext(plaintext)
    ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
    return base64.b64encode(ciphertext).decode('utf-8')

def twofish_encrypt(plaintext, key):
    # ... (Implementation of Twofish encryption) ...
    pass  # Replace with your Twofish encryption logic

def blowfish_encrypt(plaintext, key):
    # ... (Implementation of Blowfish encryption) ...
    pass  # Replace with your Blowfish encryption logic

def idea_encrypt(plaintext, key):
    # ... (Implementation of IDEA encryption) ...
    pass  # Replace with your IDEA encryption logic

def rsa_encrypt(plaintext, public_key):
    ciphertext = public_key.encrypt(
        plaintext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return base64.b64encode(ciphertext).decode('utf-8') 

def rsa_decrypt(ciphertext, private_key):
    plaintext = private_key.decrypt(
        base64.b64decode(ciphertext),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return plaintext.decode('utf-8')

def ecc_encrypt(plaintext, public_key):
    # ... (Implementation of ECC encryption) ...
    pass  # Replace with your ECC encryption logic

# -- Playfair Helper functions 
def generate_matrix(key):
    key = key.upper().replace("J", "I")  # Replace J with I in the key
    alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ" 
    matrix = []
    for char in key:
        if char not in matrix and char in alphabet: # Check if char is in alphabet
            matrix.append(char)
    for char in alphabet:
        if char not in matrix:
            matrix.append(char)
    return [matrix[i:i + 5] for i in range(0, 25, 5)]

def find_position(matrix, char):
    char = char.upper().replace("J", "I")  # Replace J with I in the character
    for row in range(5):
        for col in range(5):
            if matrix[row][col] == char:
                return row, col
    return None, None # Return None for both if character not found

def encrypt_playfair(text, matrix):
    text = text.upper().replace("J", "I")
    if len(text) % 2 != 0:
        text += 'X'
    ciphertext = ''
    for i in range(0, len(text), 2):
        a, b = text[i], text[i + 1]
        row_a, col_a = find_position(matrix, a)
        row_b, col_b = find_position(matrix, b)

        # Handle cases where a character is not found
        if row_a is None or row_b is None:
            st.error(f"Invalid plaintext character: '{a}' or '{b}' not in matrix.")
            return

        if row_a == row_b:
            ciphertext += matrix[row_a][(col_a + 1) % 5]
            ciphertext += matrix[row_b][(col_b + 1) % 5]
        elif col_a == col_b:
            ciphertext += matrix[(row_a + 1) % 5][col_a]
            ciphertext += matrix[(row_b + 1) % 5][col_b]
        else:
            ciphertext += matrix[row_a][col_b]
            ciphertext += matrix[row_b][col_a]
    return ciphertext

# --- Hill cipher Helper functions ---
def encrypt_hill(text, key):
    n = len(key)
    if len(text) % n != 0:
        text += 'X' * (n - len(text) % n) # Pad the text if necessary
    ciphertext = ''
    for i in range(0, len(text), n):
        block = text[i:i+n]
        block_vector = np.array([ord(char) - ord('A') for char in block])
        encrypted_vector = np.dot(key, block_vector) % 26
        ciphertext += ''.join([chr(val + ord('A')) for val in encrypted_vector])
    return ciphertext

# --- Helper Functions ---

def padding_plaintext(plaintext):
    block_size = algorithms.AES.block_size
    padding_length = block_size - (len(plaintext) % block_size)
    padding = bytes([padding_length] * padding_length)
    return plaintext + padding

def unpadding_plaintext(padded_plaintext):
    padding_length = padded_plaintext[-1]
    return padded_plaintext[:-padding_length]

# --- Streamlit App ---

def main():
    st.title("Encryption App")

    # Plaintext Input
    plaintext = st.text_input("Enter Plaintext:", "Type your message here")
    plaintext = plaintext.encode('utf-8') # Encode plaintext to bytes

    # Encryption Algorithm Selection
    algorithm = st.selectbox(
        "Select Encryption Algorithm:",
        (
            "Caesar Cipher",
            "Monoalphabetic Cipher",
            "Playfair Cipher",
            "Hill Cipher",
            "Polyalphabetic Cipher",
            "One-Time Pad",
            "Rail Fence",
            "Row Column Transposition",
            "AES",
            "DES", 
            "Twofish", 
            "Blowfish", 
            "IDEA", 
            "RSA", 
            "ECC"
        ),
    )

    # Algorithm-Specific Inputs (if needed)
    if algorithm == "Caesar Cipher":
        shift = st.slider("Shift Value:", min_value=1, max_value=25, value=3)
    elif algorithm == "Monoalphabetic Cipher":
        key = st.text_input("Key (26 unique lowercase letters):", "zyxwvutsrqponmlkjihgfedcba")
        if len(key) != 26 or len(set(key)) != 26 or not key.islower() or not key.isalpha():
            st.error("Invalid key! Must be 26 unique lowercase letters.")
            return
    elif algorithm == "Playfair Cipher":
        key = st.text_input("Key (Remove duplicate letters & J):", "monarchy")  
    elif algorithm == "Hill Cipher":
        key_matrix_str = st.text_input("Key Matrix (e.g., 6 24 1; 13 16 10; 20 17 15):", "6 24 1; 13 16 10; 20 17 15")
        try:
            key_matrix = [[int(x) for x in row.split()] for row in key_matrix_str.split(";")]
        except:
            st.error("Invalid key matrix format!")
            return
    elif algorithm == "Polyalphabetic Cipher":
        key = st.text_input("Key (Word or Phrase):", "LEMON")
    elif algorithm == "One-Time Pad":
        key = st.text_input("Key (Same length as plaintext):", "")
        if len(key) != len(plaintext):
            st.error("Key length must match plaintext length for One-Time Pad.")
            return
    elif algorithm == "Rail Fence":
        rails = st.slider("Number of Rails:", min_value=2, max_value=5, value=3)
    elif algorithm == "Row Column Transposition":
        key = st.text_input("Key (Numbers, e.g., 3142):", "3142")
        if not key.isdigit():
            st.error("Key must contain only digits!")
            return
    elif algorithm == 'AES':
        key = Fernet.generate_key()  # Generate a new key for AES
        st.write("Generated AES Key:", key)
    elif algorithm == 'DES':
        key = Fernet.generate_key()[:8]  # Generate a new key for DES
        st.write("Generated DES Key:", key)
    elif algorithm == 'RSA':
        st.write('Generating RSA keys...')
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        st.write('Done!')

        # Serialize the keys for display (you can save them if needed)
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        st.write("Private Key:", private_pem.decode('utf-8'))
        st.write("Public Key:", public_pem.decode('utf-8'))
    # ... (Add inputs for Twofish, Blowfish, IDEA, ECC) ...

    # --- Encrypt and Display --- 
    if st.button("Encrypt"):
        if algorithm == "Caesar Cipher":
            ciphertext = caesar_cipher(plaintext.decode('utf-8'), shift) # Decode plaintext here
        elif algorithm == "Monoalphabetic Cipher":
            ciphertext = monoalphabetic_cipher(plaintext.decode('utf-8'), key)
        elif algorithm == "Playfair Cipher":
            ciphertext = playfair_cipher(plaintext.decode('utf-8'), key)
        elif algorithm == "Hill Cipher":
            ciphertext = hill_cipher(plaintext.decode('utf-8'), key_matrix) 
        elif algorithm == "Polyalphabetic Cipher":
            ciphertext = polyalphabetic_cipher(plaintext.decode('utf-8'), key) 
        elif algorithm == "One-Time Pad":
            ciphertext = one_time_pad(plaintext.decode('utf-8'), key)
        elif algorithm == "Rail Fence":
            ciphertext = rail_fence(plaintext.decode('utf-8'), rails)
        elif algorithm == "Row Column Transposition":
            ciphertext = row_column_transposition(plaintext.decode('utf-8'), key)
        elif algorithm == "AES":
            ciphertext = aes_encrypt(plaintext, key)
        elif algorithm == "DES":
            ciphertext = des_encrypt(plaintext, key)
        elif algorithm == "Twofish":
            ciphertext = twofish_encrypt(plaintext, key) # Implement Twofish
        elif algorithm == "Blowfish":
            ciphertext = blowfish_encrypt(plaintext, key) # Implement Blowfish
        elif algorithm == "IDEA":
            ciphertext = idea_encrypt(plaintext, key) # Implement IDEA
        elif algorithm == "RSA":
            ciphertext = rsa_encrypt(plaintext, public_key) 
        elif algorithm == "ECC":
            ciphertext = ecc_encrypt(plaintext, key) # Implement ECC

        st.write("Ciphertext:", ciphertext)

if __name__ == "__main__":
    main()
