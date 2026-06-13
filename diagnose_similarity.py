#!/usr/bin/env python3
"""
Script de diagnóstico detallado para similitud facial
Muestra todos los scores y ayuda a calibrar el threshold
"""

from facial_recognition import FacialRecognitionSystem
import os
from pathlib import Path

def diagnose_similarity():
    system = FacialRecognitionSystem()
    
    print("=" * 70)
    print("DIAGNÓSTICO DE SIMILITUD FACIAL")
    print("=" * 70)
    
    # Cargar base de datos
    db = system.stored_faces
    if not db or len(db) == 0:
        print("❌ Base de datos vacía. Registra rostros primero.")
        return
    
    print(f"\n📊 Rostros registrados: {len(db)}")
    for person in db:
        print(f"  • {person}")
    
    # Pedir imagen de prueba
    test_path = input("\n📷 Ingresa ruta de imagen de prueba: ").strip()
    if not os.path.exists(test_path):
        print(f"❌ Archivo no encontrado: {test_path}")
        return
    
    # Extraer características
    import cv2
    image = cv2.imread(test_path)
    if image is None:
        print("❌ No se pudo leer la imagen")
        return
    
    test_features = system.extract_face_features(image)
    if test_features is None:
        print("❌ No se detectó rostro en la imagen")
        return
    
    print(f"\n✓ Rostro detectado. Características extraídas: {len(test_features)}")
    
    # Comparar con todos
    print("\n" + "=" * 70)
    print("RESULTADOS DE COMPARACIÓN")
    print("=" * 70)
    
    results = []
    for person, features in db.items():
        similarity = system.compare_faces(test_features, features)
        results.append((person, similarity))
        
        # Color según score
        if similarity > 0.65:
            status = "🟢 ALTO"
        elif similarity > 0.56:
            status = "🟡 MEDIO (threshold actual)"
        elif similarity > 0.40:
            status = "🟠 BAJO"
        else:
            status = "🔴 MUY BAJO"
        
        print(f"{person:20} | Score: {similarity:.4f} | {status}")
    
    # Estadísticas
    scores = [s for _, s in results]
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS")
    print("=" * 70)
    print(f"Score máximo: {max(scores):.4f}")
    print(f"Score mínimo: {min(scores):.4f}")
    print(f"Promedio:     {sum(scores)/len(scores):.4f}")
    
    # Análisis
    high_scores = [s for s in scores if s > 0.56]
    if len(high_scores) > 1:
        print(f"\n⚠️  ALERTA: {len(high_scores)} rostros con score > 0.56")
        print("   → Threshold 0.56 sigue siendo DEMASIADO BAJO")
        print(f"   → Recomendación: Usar threshold {max(scores)-0.05:.2f}")
    elif len(high_scores) == 1:
        print(f"\n✓ Solo 1 rostro con score > 0.56 (CORRECTO)")
    else:
        print(f"\n❌ Ningún rostro reconocido con threshold 0.56")
        print(f"   → Posible problema con características extraídas")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    diagnose_similarity()
