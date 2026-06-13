"""
Script para diagnosticar similitud de rostros
Muestra qué tan similar es una foto de prueba con cada rostro registrado
"""
import cv2
import numpy as np
from facial_recognition import FacialRecognitionSystem
import sys

system = FacialRecognitionSystem()

print("=" * 60)
print("DIAGNÓSTICO DE SIMILITUD DE ROSTROS")
print("=" * 60)

# Mostrar rostros registrados
print(f"\nRostros registrados en BD: {list(system.stored_faces.keys())}")
print(f"Total: {len(system.stored_faces)}")

# Pedir ruta de imagen de prueba
test_image_path = input("\nRuta de imagen de prueba: ").strip()

if not test_image_path:
    print("Error: Debes proporcionar una ruta")
    sys.exit(1)

test_image = cv2.imread(test_image_path)
if test_image is None:
    print(f"Error: No se pudo cargar {test_image_path}")
    sys.exit(1)

# Extraer características
test_features = system.extract_face_features(test_image)
if test_features is None:
    print("Error: No se detectó rostro en la imagen de prueba")
    sys.exit(1)

print(f"\n✓ Características extraídas de la imagen de prueba")

# Comparar con cada rostro registrado
print("\n" + "=" * 60)
print("SIMILITUDES:")
print("=" * 60)

similarities = {}
for person_name, stored_features in system.stored_faces.items():
    similarity = system.compare_faces(test_features, stored_features)
    similarities[person_name] = similarity
    status = "✓" if similarity > 0.56 else "✗"
    print(f"{status} {person_name:20s}: {similarity:.4f}")

# Mejor coincidencia
best_match = max(similarities, key=similarities.get)
best_score = similarities[best_match]

print("\n" + "=" * 60)
print(f"Mejor coincidencia: {best_match} ({best_score:.4f})")
print(f"Threshold: 0.56")
if best_score > 0.56:
    print(f"Resultado: ✓ IDENTIFICADO")
else:
    print(f"Resultado: ✗ NO IDENTIFICADO")
print("=" * 60)
