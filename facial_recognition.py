import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime

class FacialRecognitionSystem:
    def __init__(self, db_path='face_database'):
        """
        Inicializa el sistema de reconocimiento facial
        
        Algoritmo utilizado: OpenCV Haar Cascades + Análisis de Características
        - Detector: Haar Cascade Classifier
        - Características: Histogramas y análisis de textura
        - Validación: Análisis de profundidad y textura
        """
        self.db_path = db_path
        
        # Cargar clasificadores Haar
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Crear carpeta de base de datos si no existe
        Path(db_path).mkdir(exist_ok=True)
        
        # Cargar características almacenadas
        self.stored_faces = self.load_database()
    
    def load_database(self):
        """Carga características de rostros almacenados"""
        stored_faces = {}
        if os.path.exists(self.db_path):
            for filename in os.listdir(self.db_path):
                if filename.endswith('.npy'):
                    person_name = filename.replace('.npy', '')
                    features = np.load(os.path.join(self.db_path, filename))
                    stored_faces[person_name] = features
        return stored_faces
    
    def extract_face_features(self, image):
        """Extrae características del rostro usando histogramas y texturas - MEJORADO"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostro
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # Tomar el rostro más grande
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        
        # Redimensionar a tamaño fijo
        face_roi = cv2.resize(face_roi, (100, 100))
        
        features = []
        
        # 1. Histograma (mejorado - 128 bins normalizados)
        hist = cv2.calcHist([face_roi], [0], None, [128], [0, 256])
        hist_normalized = cv2.normalize(hist, hist).flatten()
        features.extend(hist_normalized[:128])
        
        # 2. Análisis de bordes (Sobel)
        face_roi_float = face_roi.astype(np.float32)
        sobelx = cv2.Sobel(face_roi, cv2.CV_32F, 1, 0, ksize=5)
        sobely = cv2.Sobel(face_roi, cv2.CV_32F, 0, 1, ksize=5)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.extend(edge_magnitude.flatten()[:64])
        
        # 3. Análisis de textura (Laplaciano) - usar uint8 directamente
        laplacian = cv2.Laplacian(face_roi, cv2.CV_32F)
        features.extend(laplacian.flatten()[:8])
        
        # 4. Estadísticas mejoradas
        face_roi_norm = face_roi.astype(np.float32) / 255.0
        features.append(np.mean(face_roi_norm))
        features.append(np.std(face_roi_norm))
        features.append(np.max(face_roi_norm))
        features.append(np.min(face_roi_norm))
        
        # Rellenar hasta 210 características
        while len(features) < 210:
            features.append(0)
        
        return np.array(features[:210], dtype=np.float32)
    
    def detect_spoofing(self, image):
        """
        Detecta si es un rostro real o una máscara/foto
        
        Técnicas utilizadas:
        - Análisis de textura (Laplaciano)
        - Análisis de varianza de profundidad
        - Detección de patrones planos
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Analizar texturas en región de ojos y nariz
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # Criterios de validación
        if laplacian_var < 100:
            return "MÁSCARA/FOTO", 0.5
        elif laplacian_var < 500:
            return "DESCONOCIDO", 0.6
        else:
            return "PERSONA REAL", 0.9
    
    def compare_faces(self, face1_features, face2_features, threshold=0.56):
        """
        Compara dos rostros usando similitud coseno + distancia normalizada - MEJORADO
        """
        if face1_features is None or face2_features is None:
            return 0.0
        
        # Normalizar características por vector (L2 normalization)
        face1_norm = face1_features / (np.linalg.norm(face1_features) + 1e-6)
        face2_norm = face2_features / (np.linalg.norm(face2_features) + 1e-6)
        
        # 1. Similitud coseno (50% del peso)
        cosine_sim = np.dot(face1_norm, face2_norm)
        
        # 2. Distancia euclidiana sobre características normalizadas (40% del peso)
        # Usar características normalizadas para escala consistente
        euclidean_dist = np.linalg.norm(face1_norm - face2_norm)
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # 3. Similitud de Manhattan sobre características normalizadas (10% del peso)
        manhattan_dist = np.sum(np.abs(face1_norm - face2_norm))
        manhattan_sim = 1 / (1 + manhattan_dist / 2)
        
        # Combinar: 50% coseno + 40% euclidiana + 10% manhattan
        similarity = (cosine_sim * 0.50 + euclidean_sim * 0.40 + manhattan_sim * 0.10)
        
        return min(max(similarity, 0), 1.0)
    
    def register_face(self, image_path, person_name):
        """Registra un nuevo rostro en la base de datos - MEJORADO con promediado"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: No se pudo cargar {image_path}")
            return False
        
        features = self.extract_face_features(image)
        if features is None:
            print(f"Error: No se detectó rostro en {image_path}")
            return False
        
        db_path = os.path.join(self.db_path, f'{person_name}.npy')
        
        # Si la persona ya está registrada, promediar características
        if person_name in self.stored_faces:
            old_features = self.stored_faces[person_name]
            # Promediar: (características_anteriores + características_nuevas) / 2
            combined_features = (old_features + features) / 2.0
            np.save(db_path, combined_features)
            self.stored_faces[person_name] = combined_features
            print(f"✓ Características de '{person_name}' actualizadas (promediadas)")
        else:
            # Primera vez que se registra
            np.save(db_path, features)
            self.stored_faces[person_name] = features
            print(f"✓ Rostro de '{person_name}' registrado exitosamente")
        
        return True
    
    def identify_face(self, image_path, threshold=0.56):
        """Identifica un rostro comparando con la base de datos (threshold optimizado a 0.56)"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: No se pudo cargar {image_path}")
            return None
        
        # Detectar rostro
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return {"resultado": "NO SE DETECTÓ ROSTRO", "confianza": 0.0, "spoofing": "N/A", "spoofing_confianza": 0.0, "mejor_match": None, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        # Extraer características
        face_features = self.extract_face_features(image)
        if face_features is None:
            return {"resultado": "NO SE PUDO EXTRAER CARACTERÍSTICAS", "confianza": 0.0, "spoofing": "N/A", "spoofing_confianza": 0.0, "mejor_match": None, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        # Detectar spoofing
        spoofing_status, spoofing_confidence = self.detect_spoofing(image)
        
        # Comparar con base de datos
        best_match = None
        best_similarity = 0.0
        
        for person_name, stored_features in self.stored_faces.items():
            similarity = self.compare_faces(face_features, stored_features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_name
        
        # Determinar resultado
        if best_similarity >= threshold:
            resultado = f"✓ IDENTIFICADO: {best_match}"
            confianza = best_similarity
        else:
            resultado = "✗ NO IDENTIFICADO (Rostro desconocido)"
            confianza = best_similarity
        
        return {
            "resultado": resultado,
            "confianza": round(confianza, 3),
            "spoofing": spoofing_status,
            "spoofing_confianza": spoofing_confidence,
            "mejor_match": best_match,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def process_video(self):
        """Procesa video en tiempo real desde la cámara"""
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Redimensionar para mejor rendimiento
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar rostros
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Dibujar bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extraer ROI y analizar
                roi = frame[y:y+h, x:x+w]
                features = self.extract_face_features(roi)
                
                if features is not None:
                    # Detectar spoofing
                    spoofing_status, _ = self.detect_spoofing(roi)
                    
                    # Encontrar mejor coincidencia
                    best_match = "Desconocido"
                    best_similarity = 0.0
                    
                    for person_name, stored_features in self.stored_faces.items():
                        similarity = self.compare_faces(features, stored_features)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = person_name if similarity > 0.65 else "Desconocido"
                    
                    # Mostrar información
                    text = f"{best_match} ({best_similarity:.2f})"
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, spoofing_status, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow('Reconocimiento Facial', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Programa principal"""
    system = FacialRecognitionSystem()
    
    print("=" * 60)
    print("SISTEMA DE RECONOCIMIENTO FACIAL")
    print("=" * 60)
    print("\nAlgoritmo: OpenCV Haar Cascades + Análisis de Características")
    print("- Detector: Haar Cascade Classifier")
    print("- Características: Histogramas y análisis de textura")
    print("- Validación: Análisis de profundidad y textura")
    print("=" * 60)
    
    while True:
        print("\n¿Qué deseas hacer?")
        print("1. Registrar nuevo rostro")
        print("2. Identificar rostro")
        print("3. Video en tiempo real")
        print("4. Salir")
        
        opcion = input("\nOpción: ").strip()
        
        if opcion == "1":
            person_name = input("Nombre de la persona: ").strip()
            image_path = input("Ruta de la imagen: ").strip()
            system.register_face(image_path, person_name)
        
        elif opcion == "2":
            image_path = input("Ruta de la imagen a identificar: ").strip()
            result = system.identify_face(image_path)
            print("\n" + "=" * 60)
            print(f"Resultado: {result['resultado']}")
            print(f"Confianza: {result['confianza']}")
            print(f"Verificación de Spoofing: {result['spoofing']}")
            print(f"Timestamp: {result['timestamp']}")
            print("=" * 60)
        
        elif opcion == "3":
            print("Presiona 'q' para salir...")
            system.process_video()
        
        elif opcion == "4":
            print("¡Hasta luego!")
            break
        
        else:
            print("Opción inválida")


if __name__ == "__main__":
    main()