from facial_recognition import FacialRecognitionSystem
import os
from pathlib import Path

def register_training_images():
    """Registra todas las imágenes de entrenamiento"""
    system = FacialRecognitionSystem()
    training_path = 'training_images'
    
    if not os.path.exists(training_path):
        print(f"Error: Crea la carpeta '{training_path}' con subcarpetas de personas")
        return
    
    print("=" * 60)
    print("REGISTRANDO IMÁGENES DE ENTRENAMIENTO")
    print("=" * 60)
    
    total_registered = 0
    
    # Recorrer cada persona en training_images
    for person_folder in os.listdir(training_path):
        person_path = os.path.join(training_path, person_folder)
        
        if not os.path.isdir(person_path):
            continue
        
        print(f"\n📁 Procesando: {person_folder}")
        images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            print(f"  ⚠️  No hay imágenes en {person_folder}")
            continue
        
        print(f"  Encontradas {len(images)} imágenes")
        
        # Registrar todas las imágenes de la persona
        for img_file in images:
            img_path = os.path.join(person_path, img_file)
            success = system.register_face(img_path, person_folder)
            if success:
                total_registered += 1
                print(f"  ✓ {img_file}")
            else:
                print(f"  ✗ {img_file} (no se detectó rostro)")
    
    print("\n" + "=" * 60)
    print(f"✓ Total registradas: {total_registered} imágenes")
    print("=" * 60)

if __name__ == "__main__":
    register_training_images()