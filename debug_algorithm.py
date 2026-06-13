#!/usr/bin/env python3
"""
Debug detallado del algoritmo de comparación
Muestra cada componente de la similitud
"""

from facial_recognition import FacialRecognitionSystem
import cv2
import numpy as np
import os

def debug_comparison():
    system = FacialRecognitionSystem()
    
    # Pedir imagen de prueba
    test_path = input("📷 Ingresa ruta de imagen de prueba: ").strip()
    if not os.path.exists(test_path):
        print(f"❌ Archivo no encontrado")
        return
    
    # Extraer características de prueba
    test_image = cv2.imread(test_path)
    if test_image is None:
        print("❌ No se pudo leer la imagen")
        return
    
    test_features = system.extract_face_features(test_image)
    if test_features is None:
        print("❌ No se detectó rostro")
        return
    
    print(f"\n✓ Características de prueba extraídas: {len(test_features)}")
    print(f"  Media: {np.mean(test_features):.4f}")
    print(f"  Std: {np.std(test_features):.4f}")
    print(f"  Min: {np.min(test_features):.4f}")
    print(f"  Max: {np.max(test_features):.4f}")
    
    print("\n" + "="*80)
    print("ANÁLISIS DETALLADO DE COMPARACIÓN")
    print("="*80)
    
    for person, stored_features in system.stored_faces.items():
        print(f"\n👤 {person}")
        print(f"  Características almacenadas: {len(stored_features)}")
        print(f"  Media: {np.mean(stored_features):.4f}")
        print(f"  Std: {np.std(stored_features):.4f}")
        
        # L2 Normalization
        test_norm = test_features / (np.linalg.norm(test_features) + 1e-6)
        stored_norm = stored_features / (np.linalg.norm(stored_features) + 1e-6)
        
        # Componente 1: Similitud Coseno
        cosine_sim = np.dot(test_norm, stored_norm)
        
        # Componente 2: Similitud Euclidiana (sobre características normalizadas)
        euclidean_dist = np.linalg.norm(test_norm - stored_norm)
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # Componente 3: Similitud Manhattan (sobre características normalizadas)
        manhattan_dist = np.sum(np.abs(test_norm - stored_norm))
        manhattan_sim = 1 / (1 + manhattan_dist / 2)
        
        # Similitud combinada
        similarity = (cosine_sim * 0.50 + euclidean_sim * 0.40 + manhattan_sim * 0.10)
        
        print(f"\n  COMPONENTES:")
        print(f"    1. Coseno (50%):     {cosine_sim:.4f} → {cosine_sim * 0.50:.4f}")
        print(f"    2. Euclidiana (40%): {euclidean_sim:.4f} → {euclidean_sim * 0.40:.4f}")
        print(f"    3. Manhattan (10%):  {manhattan_sim:.4f} → {manhattan_sim * 0.10:.4f}")
        print(f"  " + "-"*40)
        print(f"  SIMILITUD TOTAL: {similarity:.4f}")
        print(f"  Threshold: 0.56")
        print(f"  Estado: {'✅ RECONOCIDO' if similarity > 0.56 else '❌ RECHAZADO'}")

if __name__ == "__main__":
    debug_comparison()
