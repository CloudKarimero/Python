from facial_recognition import FacialRecognitionSystem
import os
from pathlib import Path
import json

def test_recognition():
    """Prueba el reconocimiento contra imágenes de prueba"""
    system = FacialRecognitionSystem()
    test_path = 'test_images'
    
    if not os.path.exists(test_path):
        print(f"Error: Crea la carpeta '{test_path}' con imágenes de prueba")
        return
    
    print("=" * 60)
    print("PRUEBAS DE RECONOCIMIENTO FACIAL")
    print("=" * 60)
    
    test_images = [f for f in os.listdir(test_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not test_images:
        print(f"⚠️  No hay imágenes en {test_path}")
        return
    
    print(f"Encontradas {len(test_images)} imágenes de prueba\n")
    
    results = []
    correct = 0
    
    for idx, img_file in enumerate(test_images, 1):
        img_path = os.path.join(test_path, img_file)
        
        print(f"[{idx}/{len(test_images)}] Analizando: {img_file}")
        result = system.identify_face(img_path)
        
        print(f"  Resultado: {result['resultado']}")
        print(f"  Confianza: {result['confianza']}")
        print(f"  Spoofing: {result['spoofing']}")
        print()
        
        results.append({
            "imagen": img_file,
            "resultado": result['resultado'],
            "confianza": result['confianza'],
            "spoofing": result['spoofing'],
            "timestamp": result['timestamp']
        })
        
        if "IDENTIFICADO" in result['resultado']:
            correct += 1
    
    # Generar reporte
    print("=" * 60)
    print("REPORTE FINAL")
    print("=" * 60)
    print(f"Total de pruebas: {len(test_images)}")
    print(f"Identificadas correctamente: {correct}")
    print(f"Porcentaje de acierto: {(correct/len(test_images)*100):.1f}%")
    print("=" * 60)
    
    # Guardar resultados en JSON
    with open('resultados_prueba.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Resultados guardados en 'resultados_prueba.json'")

if __name__ == "__main__":
    test_recognition()