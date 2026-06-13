"""
Script para registrar automáticamente todas las fotos de entrenamiento
y promediar características de múltiples fotos por persona
"""
from facial_recognition import FacialRecognitionSystem
import os

system = FacialRecognitionSystem()
training_path = 'training_images'

print("=" * 60)
print("REGISTRO AUTOMÁTICO DE ROSTROS")
print("=" * 60)

if not os.path.exists(training_path):
    print(f"Error: La carpeta '{training_path}' no existe")
    exit()

total_registered = 0

for person_folder in os.listdir(training_path):
    person_path = os.path.join(training_path, person_folder)
    
    if not os.path.isdir(person_path):
        continue
    
    images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print(f"\n⚠️  {person_folder}: No hay imágenes")
        continue
    
    print(f"\n📁 Registrando: {person_folder}")
    print(f"   Encontradas {len(images)} imágenes")
    
    for img_file in images:
        img_path = os.path.join(person_path, img_file)
        if system.register_face(img_path, person_folder):
            total_registered += 1
            print(f"   ✓ {img_file}")
        else:
            print(f"   ✗ {img_file} (error)")

print("\n" + "=" * 60)
print(f"✓ Total registradas: {total_registered} imágenes")
print(f"✓ Rostros únicos en BD: {len(system.stored_faces)}")
for person_name in system.stored_faces.keys():
    print(f"   - {person_name}")
print("=" * 60)
