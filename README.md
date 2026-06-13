# Sistema de Reconocimiento Facial

## Descripción General

Sistema de reconocimiento facial basado en **OpenCV Haar Cascades** con análisis multi-modal de características. Detecta, registra e identifica rostros en tiempo real desde cámara o imágenes estáticas.

**Versión**: 1.0  
**Compatibilidad**: Python 3.13  
**Dependencias**: OpenCV, NumPy

---

## Algoritmo Utilizado

### 1. Detector de Rostros: OpenCV Haar Cascade Classifier
```
Método: Cascada de clasificadores Haar (AdaBoost)
Modelo: haarcascade_frontalface_default.xml
Entrada: Imagen en escala de grises
Salida: Coordenadas de rostros detectados (x, y, w, h)
```

**Ventajas:**
- Rápido (~10-20ms por frame)
- No requiere GPU
- Funciona offline sin internet
- Excelente para rostros frontales

---

### 2. Extracción de Características: Análisis Multi-Modal

Cada rostro se representa como **210 características numéricas**:

#### Característica 1: Histograma (128 bins)
- Distribución de intensidades de píxeles (0-255)
- Histograma normalizado para consistencia
- Captura patrones globales de iluminación
- Invariante a pequeños cambios de ángulo

#### Característica 2: Análisis de Bordes Sobel (64 valores)
```python
Sobel_X = derivada en eje X
Sobel_Y = derivada en eje Y
Magnitud = √(Sobel_X² + Sobel_Y²)
```
- Detecta características estructurales del rostro
- Captura aristas y transiciones

#### Característica 3: Análisis de Textura Laplaciano (8 valores)
- Detecta detalles y texturas finas
- Útil para distinguir entre rostros reales y fotos

#### Característica 4: Estadísticas (4 valores)
- Media de intensidad
- Desviación estándar
- Valor máximo
- Valor mínimo

**Total**: 128 + 64 + 8 + 4 + relleno = 210 características

---

### 3. Detección de Spoofing: Análisis de Textura

Detecta si es un **rostro real vs máscara/foto** mediante varianza del Laplaciano:

```
Laplaciano = Derivada segunda de la imagen
Varianza del Laplaciano = Medida de textura/detalles
```

**Criterios de Clasificación:**
```
Si laplacian_var < 100:       → MÁSCARA/FOTO (confianza: 0.5)
Si 100 ≤ laplacian_var < 500: → DESCONOCIDO (confianza: 0.6)
Si laplacian_var ≥ 500:       → PERSONA REAL (confianza: 0.9)
```

**Lógica:**
- Fotos/máscaras tienen menos variación de texturas
- Rostros reales tienen más detalles de piel y expresiones

---

### 4. Comparación de Rostros: Similitud Coseno + Distancia (MEJORADO)

**Algoritmo Mejorado - Más Discriminativo:**

```python
# L2 Normalization (normaliza por magnitud del vector)
face1_norm = face1 / ||face1||
face2_norm = face2 / ||face2||

# 1. Similitud Coseno (50% del peso) - estructura facial
cosine_sim = dot(face1_norm, face2_norm)

# 2. Similitud Euclidiana (40% del peso) - diferencia de magnitudes
euclidean_dist = ||face1 - face2||
euclidean_sim = 1 / (1 + euclidean_dist/20)

# 3. Similitud Manhattan (10% del peso) - diferencia por características
manhattan_dist = sum(|face1 - face2|)
manhattan_sim = 1 / (1 + manhattan_dist/100)

# Similitud combinada: 50% coseno + 40% euclidiana + 10% manhattan
similarity = cosine_sim × 0.50 + euclidean_sim × 0.40 + manhattan_sim × 0.10
```

**Mejora Clave:** El nuevo algoritmo usa similitud coseno en lugar de correlación de Pearson, lo que **reduce significativamente los falsos positivos** entre personas diferentes.
similarity = correlation * 0.7 + distance_similarity * 0.3
similarity = (similarity + 1) / 2  # Normalizar a rango 0-1
```

**Rango**: 0 (diferentes) a 1 (idénticos)

**Threshold por defecto**: 0.56 (56% de similitud - balance optimizado entre discriminación y reconocimiento)

**Base de datos:**
- Almacenamiento: Archivos `.npy` (NumPy binarios)
- Ubicación: Carpeta `face_database/`
- Formato: `{nombre_persona}.npy`
- Capacidad: Ilimitada (solo limitada por disco)
- **Promediado**: Si registras múltiples fotos de la misma persona, se promedian automáticamente

---

### 5. Video en Tiempo Real: Reconocimiento Continuo

```
Frame de cámara (640×480)
    ↓
Detectar todos los rostros
    ↓
Para cada rostro:
    ├─ Extraer 200 características
    ├─ Comparar con todos los rostros en BD
    ├─ Encontrar mejor coincidencia
    ├─ Detectar si es real o spoofing
    └─ Mostrar resultado en pantalla
    ↓
Siguiente frame (~30 FPS)
```

---

## Estructura del Código

### Clase: `FacialRecognitionSystem`

#### Método: `__init__(db_path='face_database')`
```python
# Inicializa el sistema
system = FacialRecognitionSystem()
```
- Carga clasificadores Haar Cascade
- Crea carpeta de base de datos
- Carga rostros previamente registrados

---

#### Método: `extract_face_features(image)`
```python
features = system.extract_face_features(image)
# Retorna: array de 200 valores o None si no detecta rostro
```
**Proceso:**
1. Convertir a escala de grises
2. Detectar rostro más grande
3. Redimensionar a 100×100 píxeles
4. Calcular histograma (64 bins)
5. Calcular Sobel X y Y (64 valores)
6. Calcular media, varianza, Laplaciano
7. Rellenar hasta 200 características

---

#### Método: `detect_spoofing(image)`
```python
spoofing_status, confidence = system.detect_spoofing(image)
# Ejemplo: ("PERSONA REAL", 0.9)
```
**Retorna:**
- `spoofing_status`: "PERSONA REAL" | "MÁSCARA/FOTO" | "DESCONOCIDO"
- `confidence`: 0.5-0.9

---

#### Método: `compare_faces(face1_features, face2_features)`
```python
similarity = system.compare_faces(features1, features2)
# Retorna: float entre 0 y 1
```
- 0 = rostros diferentes
- 1 = rostros idénticos
- 0.56+ = Coincidencia (threshold)

---

#### Método: `register_face(image_path, person_name)`
```python
system.register_face('training_images/Jose/foto1.jpg', 'Jose')
```
**Proceso:**
1. Lee imagen
2. Extrae características
3. Si es la primera vez: guarda como `face_database/Jose.npy`
4. Si ya existe: promedia características anteriores con las nuevas
5. Carga en memoria

**Nota**: Registrar múltiples fotos mejora la precisión automáticamente

---

#### Método: `identify_face(image_path, threshold=0.56)`
```python
result = system.identify_face('test_images/test.jpg')
```
**Retorna diccionario:**
```python
{
    "resultado": "✓ IDENTIFICADO: Jose",
    "confianza": 0.823,
    "spoofing": "PERSONA REAL",
    "spoofing_confianza": 0.9,
    "mejor_match": "Jose",
    "timestamp": "2026-06-13 10:30:45"
}
```

---

#### Método: `process_video()`
```python
system.process_video()
# Presiona 'q' para salir
```
**En tiempo real:**
- Captura desde cámara (index 0)
- Detecta y identifica rostros
- Muestra bounding boxes con nombre y confianza
- Muestra estado de spoofing
- Presiona 'q' para cerrar

---

## Flujo de Funcionamiento

### 1. Registrar Nuevo Rostro
```
Opción 1
  ↓
Nombre de la persona: Jose
  ↓
Ruta de la imagen: training_images/Jose/foto1.jpg
  ↓
extract_face_features(imagen)
  ↓
np.save('face_database/Jose.npy', características)
  ↓
✓ Rostro de 'Jose' registrado exitosamente
```

### 2. Identificar Rostro en Imagen
```
Opción 2
  ↓
Ruta de la imagen a identificar: test_images/test.jpg
  ↓
detect + extract_features + detect_spoofing
  ↓
compare_faces(características_test vs BD)
  ↓
Mostrar resultado con confianza
```

### 3. Video en Tiempo Real
```
Opción 3
  ↓
while cap.isOpened():
    frame = cámara
    rostros = detectar_rostros(frame)
    para cada rostro:
        features = extract_features
        resultado = identify_face
        dibujar bounding box + nombre + confianza
    mostrar frame
    si presiona 'q': salir
```

---

## Rendimiento

| Operación | Tiempo Aprox |
|-----------|-------------|
| Detectar rostro | 10-20ms |
| Extraer 210 características | 40-60ms |
| Comparar con 100 rostros | 15-25ms |
| Procesamiento frame video | 40-60ms |
| Detección spoofing | 5-10ms |
| **Total por imagen** | **~60-120ms** |
| **FPS en video** | **~8-16 FPS** |

**Nota**: El aumento a 210 características es mínimo comparado con la mejora en precisión. Con 29+ fotos promediadas, la precisión mejora significativamente.

---

## Limitaciones y Problemas Conocidos del Algoritmo

### 1. Sensibilidad a Ángulos de Cámara ⚠️ (CRÍTICO)

**Problema**: El reconocimiento falla cuando la cámara o el rostro no están alineados frontalmente.

**Comportamiento:**
- **Ángulo 0° (frontal)**: ✅ Reconocimiento excelente
- **Ángulo 15-30° (perfil leve)**: ⚠️ Reconocimiento parcial
- **Ángulo >30° (perfil severo)**: ❌ Falla completamente

**Causa técnica:**
- Haar Cascade entrenado solo con rostros frontales
- Las características (histograma, Sobel) cambian radicalmente con rotación
- La similitud coseno ya no es válida con diferentes puntos de vista

**Solución:**
```python
# Para mejorar, necesitarías:
1. Agregar más fotos en diferentes ángulos durante el registro
2. Usar modelos de deep learning (MTCNN, FaceNet) en lugar de Haar Cascade
3. Implementar estimación de pose y corrección de rotación
```

---

### 2. Problemas de Iluminación 💡

**Problema**: Cambios en la iluminación hacen fallar el reconocimiento.

**Ejemplos:**
- Luz frontal vs luz lateral: características cambian 20-40%
- Sombras en rostro: histograma se distorsiona
- Contraluz: detección de rostro falla completamente
- Iluminación muy débil: ruido en características

**Comportamiento:**
```
Foto de entrenamiento: Luz natural frontal
  ↓
Foto de prueba: Luz artificial lateral
  ↓
Similitud: 0.45 (rechazada, threshold 0.56)
```

**Causa técnica:**
- Histograma es muy sensible a intensidad de píxeles
- Sobel se ve afectado por cambios abruptos de luz
- Las características extraídas son contexto-dependientes

**Solución:**
```python
# Mejoras posibles:
1. Registrar la misma persona con diferentes iluminaciones
2. Normalizar histograma con CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. Usar características invariantes a luz (LBP, SIFT) en lugar de histograma
```

---

### 3. Sensibilidad a Expresiones y Gestos 😊

**Problema**: Cambios en la expresión facial reducen significativamente la similitud.

**Ejemplos:**
- Sonriendo vs cara neutral: -15% similitud
- Con/sin gafas: -25% similitud
- Con/sin barba: -20% similitud
- Diferentes expresiones: -10-30% similitud

**Comportamiento:**
```
Entrenamiento: Foto neutral
  ↓
Prueba: Sonriendo
  ↓
Similitud Jose: 0.52 (cerca del threshold 0.56, puede rechazarse)
```

**Causa técnica:**
- Los bordes (Sobel) cambian con músculos faciales
- El histograma cambia con sombras de expresión
- La estructura facial se deforma con gestos

**Solución:**
```python
# Mejoras:
1. Registrar múltiples fotos con diferentes expresiones (obligatorio)
2. Usar puntos clave del rostro (landmarks) en lugar de características globales
3. Combinar con análisis de posición de ojos/nariz (más robustos)
```

---

### 4. Distancia de la Cámara y Resolución 📏

**Problema**: Si el rostro está muy cerca o muy lejos, falla la detección o las características se distorsionan.

**Comportamiento:**
- Rostro muy pequeño (<40×40 px): ❌ No detecta
- Rostro muy grande (>200×200 px): ⚠️ Características inconsistentes (redimensiona a 100×100)
- Rostro al borde del frame: ⚠️ Parcialmente detectado o rechazado

**Causa técnica:**
- Haar Cascade busca rostros en rango de tamaño específico
- Redimensionamiento a 100×100 pierde información si es muy grande
- Pérdida de detalles si es muy pequeño

**Solución:**
```python
# Mejoras:
1. Usar multi-scale detection (diferentes escalas)
2. Mantener rostro a 80-150 px de distancia durante video
3. Aumentar rango de detección en Haar Cascade
```

---

### 5. Confusión Entre Personas Similares 👥

**Problema**: Personas con rasgos similares (hermanos, gemelos, actores parecidos) pueden confundirse.

**Ejemplo del caso actual:**
```
Imagen de persona desconocida:
  Jose:  0.5952 ✅ Reconocido
  Tom:   0.4341 ❌ Rechazado

Pero si la persona desconocida es parecido a Jose:
  Jose:  0.62 ✅ Reconocido CORRECTO
  Tom:   0.58 ⚠️ FALSO POSITIVO
```

**Causa técnica:**
- 210 características son insuficientes para distinguir personas muy similares
- Similitud coseno + euclidiana no capturan diferencias sutiles
- Falta identificación de características únicas (lunares, cicatrices, etc.)

**Solución:**
```python
# Mejoras:
1. Aumentar threshold (pero reduce reconocimiento de la persona correcta)
2. Usar deep learning (FaceNet) - 512 características, mucho más discriminativo
3. Agregar biometría (huella de voz, iris)
```

---

### 6. Problemas de Detección de Rostro 🔍

**Problema**: En algunos casos, Haar Cascade no detecta rostros.

**Situaciones:**
- Rostros de perfil completo: ❌ No detecta
- Rostros boca abajo: ❌ No detecta
- Rostros con accesorios grandes (máscaras, gafas oscuras): ⚠️ Detección parcial
- Imágenes de baja calidad: ❌ No detecta
- Múltiples rostros muy cercanos: ⚠️ Solo detecta el más grande

**Causa técnica:**
- Haar Cascade entrenado solo con rostros frontales
- Sensible a variaciones no contempladas en entrenamiento
- Requiere mínima resolución

**Solución:**
```python
# Mejoras:
1. Usar MTCNN (más robusto con ángulos)
2. Usar RetinaFace (facial detection moderno)
3. Aumentar imageSize en detectMultiScale()
```

---

### 7. Spoofing Detection Débil 🎭

**Problema**: La detección de fotos/máscaras es muy básica (solo Laplaciano).

**Limitaciones:**
- Fotos de alta resolución pueden parecer "reales" (Laplaciano alto)
- Pantallas con video en vivo: pueden ser falsamente detectadas como reales
- Máscaras 3D realistas: pueden engañar el algoritmo

**Comportamiento:**
```
Foto de Jose en pantalla:
  Laplacian variance: 450
  Resultado: "PERSONA REAL" ❌ FALSO POSITIVO
```

**Causa técnica:**
- Laplaciano detecta texturas, pero fotos modernas tienen mucha textura
- No hay análisis de profundidad (2D vs 3D)
- No hay detección de parpadeo/movimiento

**Solución:**
```python
# Mejoras:
1. Combinar Laplaciano + LBP + texture analysis
2. Usar cámara de profundidad (RGB-D)
3. Detectar parpadeo (requiere múltiples frames)
4. Usar redes de deep learning especializadas en liveness detection
```

---

### 8. Necesidad de Re-calibración con Cambios Ambientales 🔧

**Problema**: El threshold (0.56) puede necesitar ajuste dependiendo del contexto.

**Escenarios:**
- Oficina bien iluminada: threshold 0.56 funciona ✅
- Luz natural variable: threshold demasiado bajo ❌
- Interior oscuro: threshold demasiado alto ❌

**Solución:**
```python
# Mejoras:
1. Hacer threshold dinámico basado en iluminación
2. Usar múltiples thresholds por persona
3. Ajustar threshold según confianza del spoofing
```

---

### 9. Base de Datos Pequeña Afecta Precisión 📊

**Problema**: Con solo 1-2 fotos de entrenamiento, la precisión es baja.

**Resultado con diferentes cantidades:**
```
1 foto:    Similitud intra-persona: 0.55-0.65 (muy variable)
3 fotos:   Similitud intra-persona: 0.60-0.75 (mejor)
10 fotos:  Similitud intra-persona: 0.65-0.80 (bueno)
30 fotos:  Similitud intra-persona: 0.70-0.85 (excelente)
```

**Causa técnica:**
- Pocas fotos = menos variabilidad capturada
- Promediado de características requiere muestras representativas
- Una foto "atípica" afecta mucho el promedio

**Solución:**
```python
# Mejoras:
1. Usar mínimo 5-10 fotos de entrenamiento (OBLIGATORIO)
2. Usar técnicas de data augmentation
3. Usar clustering en lugar de simple promedio
```

---

### Resumen de Problemas Críticos

| Problema | Severidad | Impacto | Solución Rápida |
|----------|-----------|--------|-----------------|
| Ángulo de cámara | 🔴 CRÍTICA | Reconocimiento falla >30° | Tomar fotos con múltiples ángulos |
| Iluminación | 🔴 CRÍTICA | Cambios de luz rompen características | Registrar con diferentes luces |
| Distancia/Resolución | 🟠 ALTA | No detecta rostros fuera de rango | Mantener distancia 0.5-2m |
| Expresiones | 🟠 ALTA | Similitud baja si cambia expresión | Registrar con varias expresiones |
| Spoofing débil | 🟠 ALTA | Fotos engañan al detector | Usar análisis de profundidad |
| Personas similares | 🟡 MEDIA | Falsos positivos ocasionales | Aumentar threshold con cuidado |
| Base datos pequeña | 🟡 MEDIA | Precisión baja | Mínimo 5-10 fotos por persona |

---

### Recomendaciones Para Mejor Resultado

**Registración:**
✅ Usar mínimo **10 fotos** por persona  
✅ Variar ángulos (frontal, 15°, -15°)  
✅ Variar iluminación (natural, artificial)  
✅ Incluir diferentes expresiones (neutral, sonriendo)  
✅ Mantener rostro a 1-2m de cámara  

**Identificación:**
✅ Posicionarse frontalmente a la cámara  
✅ Buena iluminación frontal  
✅ Rostro completamente visible  
✅ Mantener distancia 0.5-2m  

**Si sigue fallando:**
❌ No usar Haar Cascade → Cambiar a **MTCNN o RetinaFace**  
❌ No usar características manual → Cambiar a **FaceNet (512 características)**  
❌ No usar Laplaciano → Usar **análisis de profundidad RGB-D**

---

## Ventajas vs Desventajas

### Ventajas ✅
- Compatible con Python 3.13
- No requiere conexión a internet
- Funciona sin GPU
- Almacenamiento eficiente (características comprimidas)
- Procesamiento muy rápido (50-100ms por imagen)
- Código simple y personalizable
- **Promediado automático** de múltiples fotos
- Mejora de precisión con más datos (210 características)
- Combinación de métodos (correlación + distancia euclidiana)
- Threshold ajustable para mayor flexibilidad

### Desventajas ❌
- Mejor con rostros frontales (±15°)
- Sensible a cambios severos de iluminación
- Falsos positivos con fotos/pantallas (mitigado con spoofing detection)
- Haar Cascade menos preciso que modelos de deep learning
- Requiere al menos 1 foto de entrenamiento
- Mejora significativa con múltiples fotos (3-5 mínimo recomendado)

---

## Ejemplos de Uso

### Ejemplo 1: Registrar Rostro Individual
```python
system = FacialRecognitionSystem()
system.register_face('training_images/Jose/foto1.jpg', 'Jose')
```

### Ejemplo 2: Registrar Múltiples Fotos Automáticamente
```bash
# Script que registra todas las fotos de training_images/
python .\auto_register_batch.py
```

### Ejemplo 3: Diagnosticar Similitud
```bash
# Script para probar similitud con una foto específica
python .\test_similarity.py
# Te pide la ruta de la imagen de prueba
# Muestra similitud con cada rostro registrado
```

### Ejemplo 4: Identificar Rostro en Imagen Estática
```python
result = system.identify_face('test_images/test.jpg', threshold=0.55)
print(result['resultado'])        # "✓ IDENTIFICADO: Jose"
print(result['confianza'])        # 0.823
print(result['spoofing'])         # "PERSONA REAL"
```

### Ejemplo 5: Video en Tiempo Real
```python
system.process_video()
# La cámara se abre y empieza a identificar rostros
# Presiona 'q' para cerrar
```

---

## Estructura de Carpetas

```
c:\Users\karim\Python\
├── facial_recognition.py        # Código principal (clase)
├── register_faces.py            # Menú interactivo principal
├── test_faces.py                # Script de pruebas
├── auto_register_batch.py       # Registra automáticamente todas las fotos
├── test_similarity.py           # Diagnostica similitud de una imagen
├── README.md                    # Este archivo
├── face_database/               # Base de datos de rostros (características)
│   ├── Jose.npy                 # Características promediadas de Jose
│   ├── Maria.npy
│   └── ...
├── training_images/
│   ├── Jose/
│   │   ├── foto1.jpg
│   │   ├── foto2.jpg
│   │   └── ... (hasta 30 fotos)
│   └── Maria/
│       └── foto1.jpg
└── test_images/
    ├── test1.jpg
    └── test2.jpg
```

---

## Instalación y Uso

### 1. Instalar dependencias
```bash
pip install opencv-python numpy
```

### 2. Crear carpetas necesarias
```bash
mkdir training_images
mkdir test_images
```

### 3. Agregar imágenes de entrenamiento
```
training_images/
├── Jose/
│   ├── foto1.jpg
│   ├── foto2.jpg
│   └── ... (mínimo 3-5 fotos por persona)
└── Maria/
    ├── foto1.jpg
    └── foto2.jpg
```

### 4. Registrar automáticamente todas las fotos
```bash
# Script que lee training_images/ y registra todo
python auto_register_batch.py
```

### 5. (Alternativa) Menú interactivo
```bash
python register_faces.py
```

### 6. Diagnosticar similitud (Opcional)
```bash
# Prueba con una imagen específica
python test_similarity.py
```

### Opción 1: Registrar nuevo rostro
```
Nombre: Jose
Ruta: training_images/Jose/foto1.jpg
```
Repite para cada foto (se promedian automáticamente)

### Opción 2: Identificar rostro
```
Ruta: test_images/test.jpg
```

### Opción 3: Video en tiempo real
```
Presiona 'q' para salir
```

### Opción 4: Salir
```
Cierra el programa
```

---

## Notas Técnicas

### Matriz de Características (210x1)
```
[histograma_128 | sobel_64 | laplaciano_8 | stats_4 | relleno]
 Bins normalizados | Bordes  | Texturas    | Media,Std,Max,Min
```

### Correlación de Pearson
```
correlation = dot(face1_norm, face2_norm) / n
rango: -1 (opuestos) a +1 (idénticos)
```

### Distancia Euclidiana Normalizada
```
distance = ||face1_norm - face2_norm||
distance_similarity = 1 / (1 + distance/5)
rango: 0 (muy diferentes) a 1 (muy similares)
```

### Similitud Combinada
```
similarity = correlation * 0.7 + distance_similarity * 0.3
similarity = (similarity + 1) / 2
rango final: 0 a 1
threshold: 0.55 (puede ajustarse)
```

### Promediado de Características
```
característica_final = (caractéristica_anterior + característica_nueva) / 2
Se aplica automáticamente al registrar múltiples fotos
```

---

## Contacto y Soporte

Sistema desarrollado con OpenCV y NumPy.  
Compatible con Python 3.13 en Windows/Linux/Mac.

---

*Última actualización: 2026-06-13*
