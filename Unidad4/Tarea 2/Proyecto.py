import cv2
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time

class ReconocimientoEmociones:
    def __init__(self):
        self.modelo = None
        self.label_encoder = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Mapeo de carpetas en inglés a emociones en español
        self.carpetas_ingleses = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.emociones_espanol = ['Enojado', 'Desprecio', 'Asco', 'Miedo', 'Feliz', 'Neutral', 'Triste', 'Sorpresa']
        
        # Diccionario para convertir inglés a español
        self.traduccion_emociones = dict(zip(self.carpetas_ingleses, self.emociones_espanol))
        self.traduccion_inversa = dict(zip(self.emociones_espanol, self.carpetas_ingleses))
        
        self.colores_emociones = {
            'Enojado': (0, 0, 255),     # Rojo
            'Desprecio': (128, 0, 128),  # Púrpura
            'Asco': (0, 255, 0),        # Verde
            'Miedo': (255, 0, 255),     # Magenta
            'Feliz': (0, 255, 255),     # Amarillo
            'Neutral': (255, 255, 255), # Blanco
            'Triste': (255, 0, 0),      # Azul
            'Sorpresa': (0, 165, 255)   # Naranja
        }
        
    def cargar_imagenes_desde_carpetas(self, ruta_dataset):
        """Carga imágenes desde carpetas organizadas por emociones (en inglés)"""
        print("Cargando imágenes del dataset...")
        imagenes = []
        etiquetas = []
        
        # Buscar carpetas en inglés
        for carpeta_ingles in self.carpetas_ingleses:
            ruta_emocion = os.path.join(ruta_dataset, carpeta_ingles)
            if not os.path.exists(ruta_emocion):
                print(f"Advertencia: No se encontró la carpeta {carpeta_ingles}")
                continue
            
            # Obtener la traducción al español
            emocion_espanol = self.traduccion_emociones[carpeta_ingles]
            contador = 0
            
            for archivo in os.listdir(ruta_emocion):
                if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    ruta_imagen = os.path.join(ruta_emocion, archivo)
                    try:
                        img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img_redimensionada = cv2.resize(img, (48, 48))
                            imagenes.append(img_redimensionada)
                            etiquetas.append(emocion_espanol)  # Guardar en español
                            contador += 1
                    except Exception as e:
                        print(f"Error al cargar {archivo}: {e}")
            
            print(f"{carpeta_ingles} -> {emocion_espanol}: {contador} imágenes cargadas")
        
        # Mostrar estadísticas del dataset
        contador_etiquetas = Counter(etiquetas)
        print("\nDistribución del dataset:")
        for emocion, cantidad in contador_etiquetas.items():
            print(f"   {emocion}: {cantidad} imágenes")
        
        return np.array(imagenes), np.array(etiquetas)
    
    def crear_modelo(self, num_clases):
        """Crea la arquitectura del modelo CNN"""
        modelo = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_clases, activation='softmax')
        ])
        
        modelo.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return modelo
    
    def generar_matriz_confusion(self, X_val, y_val):
        """Genera y visualiza la matriz de confusión"""
        print("\nGenerando matriz de confusión...")
        
        # Realizar predicciones
        predicciones = self.modelo.predict(X_val, verbose=0)
        y_pred = np.argmax(predicciones, axis=1)
        
        # Generar matriz de confusión
        cm = confusion_matrix(y_val, y_pred)
        
        # Obtener nombres de clases en español
        nombres_clases = self.label_encoder.inverse_transform(range(len(self.label_encoder.classes_)))
        
        # Crear figura con mayor tamaño
        plt.figure(figsize=(12, 10))
        
        # Crear heatmap
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=nombres_clases,
                   yticklabels=nombres_clases,
                   cbar_kws={'label': 'Número de Predicciones'})
        
        plt.title('Matriz de Confusión - Reconocimiento de Emociones\n', fontsize=16, fontweight='bold')
        plt.xlabel('Predicción del Modelo', fontsize=12, fontweight='bold')
        plt.ylabel('Etiqueta Real', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar matriz de confusión
        plt.savefig('matriz_confusion.png', dpi=300, bbox_inches='tight')
        print("Matriz de confusión guardada como 'matriz_confusion.png'")
        
        # Mostrar matriz
        plt.show()
        
        return cm, nombres_clases
    
    def generar_reporte_clasificacion(self, X_val, y_val):
        """Genera reporte detallado de clasificación"""
        print("\nGenerando reporte de clasificación...")
        
        # Realizar predicciones
        predicciones = self.modelo.predict(X_val, verbose=0)
        y_pred = np.argmax(predicciones, axis=1)
        
        # Obtener nombres de clases
        nombres_clases = self.label_encoder.inverse_transform(range(len(self.label_encoder.classes_)))
        
        # Generar reporte
        reporte = classification_report(y_val, y_pred, 
                                      target_names=nombres_clases,
                                      output_dict=True)
        
        # Mostrar reporte en consola
        print("\n" + "="*60)
        print("REPORTE DE CLASIFICACIÓN")
        print("="*60)
        print(classification_report(y_val, y_pred, target_names=nombres_clases))
        
        # Guardar reporte en archivo
        with open('reporte_clasificacion.txt', 'w', encoding='utf-8') as f:
            f.write("REPORTE DE CLASIFICACIÓN - RECONOCIMIENTO DE EMOCIONES\n")
            f.write("="*60 + "\n\n")
            f.write(classification_report(y_val, y_pred, target_names=nombres_clases))
            
            # Agregar métricas generales
            f.write(f"\n\nMÉTRICAS GENERALES:\n")
            f.write(f"- Precisión Global: {reporte['accuracy']:.4f} ({reporte['accuracy']:.2%})\n")
            f.write(f"- Macro Avg Precision: {reporte['macro avg']['precision']:.4f}\n")
            f.write(f"- Macro Avg Recall: {reporte['macro avg']['recall']:.4f}\n")
            f.write(f"- Macro Avg F1-Score: {reporte['macro avg']['f1-score']:.4f}\n")
            f.write(f"- Weighted Avg Precision: {reporte['weighted avg']['precision']:.4f}\n")
            f.write(f"- Weighted Avg Recall: {reporte['weighted avg']['recall']:.4f}\n")
            f.write(f"- Weighted Avg F1-Score: {reporte['weighted avg']['f1-score']:.4f}\n")
        
        print("Reporte guardado como 'reporte_clasificacion.txt'")
        
        return reporte
    
    def generar_grafico_precision_por_clase(self, reporte_clasificacion):
        """Genera gráfico de precisión por clase"""
        print("\nGenerando gráfico de precisión por clase...")
        
        # Extraer datos del reporte
        clases = []
        precision = []
        recall = []
        f1_score = []
        
        for clase, metricas in reporte_clasificacion.items():
            if clase not in ['accuracy', 'macro avg', 'weighted avg']:
                clases.append(clase)
                precision.append(metricas['precision'])
                recall.append(metricas['recall'])
                f1_score.append(metricas['f1-score'])
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(clases))
        width = 0.25
        
        bars1 = ax.bar(x - width, precision, width, label='Precisión', color='skyblue', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', color='lightcoral', alpha=0.8)
        bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', color='lightgreen', alpha=0.8)
        
        # Personalizar gráfico
        ax.set_xlabel('Emociones', fontweight='bold')
        ax.set_ylabel('Puntuación', fontweight='bold')
        ax.set_title('Métricas de Rendimiento por Emoción', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(clases, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # Agregar valores en las barras
        def agregar_valores(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        agregar_valores(bars1)
        agregar_valores(bars2)
        agregar_valores(bars3)
        
        plt.tight_layout()
        plt.savefig('metricas_por_clase.png', dpi=300, bbox_inches='tight')
        print("Gráfico guardado como 'metricas_por_clase.png'")
        plt.show()
    
    def analisis_completo_modelo(self, X_val, y_val):
        """Realiza análisis completo del modelo incluyendo matriz de confusión"""
        print("\nIniciando análisis completo del modelo...")
        
        # 1. Generar matriz de confusión
        cm, nombres_clases = self.generar_matriz_confusion(X_val, y_val)
        
        # 2. Generar reporte de clasificación
        reporte = self.generar_reporte_clasificacion(X_val, y_val)
        
        # 3. Generar gráfico de precisión por clase
        self.generar_grafico_precision_por_clase(reporte)
        
        # 4. Análisis adicional de la matriz de confusión
        self.analizar_matriz_confusion(cm, nombres_clases)
        
        print("\nAnálisis completo finalizado!")
        print("Archivos generados:")
        print("   - matriz_confusion.png")
        print("   - reporte_clasificacion.txt")
        print("   - metricas_por_clase.png")
        print("   - analisis_matriz_confusion.txt")
    
    def analizar_matriz_confusion(self, cm, nombres_clases):
        """Analiza la matriz de confusión y genera insights"""
        print("\nAnalizando matriz de confusión...")
        
        # Calcular métricas por clase
        total_predicciones = np.sum(cm)
        precision_por_clase = np.diag(cm) / np.sum(cm, axis=0)
        recall_por_clase = np.diag(cm) / np.sum(cm, axis=1)
        
        # Encontrar las confusiones más comunes
        confusiones = []
        for i in range(len(nombres_clases)):
            for j in range(len(nombres_clases)):
                if i != j and cm[i][j] > 0:
                    confusiones.append((nombres_clases[i], nombres_clases[j], cm[i][j]))
        
        # Ordenar confusiones por frecuencia
        confusiones.sort(key=lambda x: x[2], reverse=True)
        
        # Guardar análisis
        with open('analisis_matriz_confusion.txt', 'w', encoding='utf-8') as f:
            f.write("ANÁLISIS DETALLADO DE LA MATRIZ DE CONFUSIÓN\n")
            f.write("="*50 + "\n\n")
            
            f.write("PRECISIÓN POR CLASE:\n")
            f.write("-"*30 + "\n")
            for i, clase in enumerate(nombres_clases):
                f.write(f"{clase}: {precision_por_clase[i]:.4f} ({precision_por_clase[i]:.2%})\n")
            
            f.write(f"\nRECALL POR CLASE:\n")
            f.write("-"*30 + "\n")
            for i, clase in enumerate(nombres_clases):
                f.write(f"{clase}: {recall_por_clase[i]:.4f} ({recall_por_clase[i]:.2%})\n")
            
            f.write(f"\nCONFUSIONES MÁS COMUNES:\n")
            f.write("-"*30 + "\n")
            for i, (real, predicha, frecuencia) in enumerate(confusiones[:10]):
                porcentaje = (frecuencia / total_predicciones) * 100
                f.write(f"{i+1}. {real} -> {predicha}: {frecuencia} veces ({porcentaje:.2f}%)\n")
            
            # Encontrar mejor y peor clase
            mejor_clase_idx = np.argmax(precision_por_clase)
            peor_clase_idx = np.argmin(precision_por_clase)
            
            f.write(f"\nCLASE CON MEJOR RENDIMIENTO:\n")
            f.write(f"- {nombres_clases[mejor_clase_idx]}: {precision_por_clase[mejor_clase_idx]:.4f}\n")
            
            f.write(f"\nCLASE CON PEOR RENDIMIENTO:\n")
            f.write(f"- {nombres_clases[peor_clase_idx]}: {precision_por_clase[peor_clase_idx]:.4f}\n")
        
        print("Análisis detallado guardado como 'analisis_matriz_confusion.txt'")

    def entrenar_modelo(self, ruta_dataset):
        """Entrena el modelo con el dataset proporcionado"""
        print("Iniciando entrenamiento del modelo...")
        
        # Cargar datos
        imagenes, etiquetas = self.cargar_imagenes_desde_carpetas(ruta_dataset)
        
        if len(imagenes) == 0:
            print("Error: No se encontraron imágenes para entrenar")
            return False
        
        # Preprocesar datos
        imagenes = imagenes.astype('float32') / 255.0
        imagenes = np.expand_dims(imagenes, axis=-1)
        
        # Codificar etiquetas
        self.label_encoder = LabelEncoder()
        etiquetas_codificadas = self.label_encoder.fit_transform(etiquetas)
        
        # Dividir dataset
        X_train, X_val, y_train, y_val = train_test_split(
            imagenes, etiquetas_codificadas, test_size=0.2, random_state=42, stratify=etiquetas_codificadas
        )
        
        print(f"Datos de entrenamiento: {len(X_train)} imágenes")
        print(f"Datos de validación: {len(X_val)} imágenes")
        
        # Crear y entrenar modelo
        self.modelo = self.crear_modelo(len(np.unique(etiquetas_codificadas)))
        
        print("\nArquitectura del modelo:")
        self.modelo.summary()
        
        # Callbacks para mejorar el entrenamiento
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy'),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, monitor='val_accuracy')
        ]
        
        # Entrenar
        print("\nComenzando entrenamiento...")
        historia = self.modelo.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar modelo
        val_loss, val_accuracy = self.modelo.evaluate(X_val, y_val, verbose=0)
        print(f"\nEntrenamiento completado!")
        print(f"Precisión en validación: {val_accuracy:.2%}")
        
        # Guardar modelo y encoder
        self.guardar_modelo()
        
        # NUEVO: Realizar análisis completo con matriz de confusión
        self.analisis_completo_modelo(X_val, y_val)
        
        return True
    
    def guardar_modelo(self):
        """Guarda el modelo entrenado y el label encoder"""
        try:
            self.modelo.save('modelo_emociones.h5')
            with open('label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print("Modelo guardado como 'modelo_emociones.h5'")
            print("Label encoder guardado como 'label_encoder.pkl'")
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")
    
    def cargar_modelo(self):
        """Carga un modelo previamente entrenado"""
        try:
            if os.path.exists('modelo_emociones.h5') and os.path.exists('label_encoder.pkl'):
                self.modelo = keras.models.load_model('modelo_emociones.h5')
                with open('label_encoder.pkl', 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print("Modelo cargado exitosamente")
                return True
            else:
                print("No se encontraron archivos del modelo entrenado")
                return False
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False
    
    def evaluar_modelo_cargado(self, ruta_dataset):
        """Evalúa un modelo ya cargado con matriz de confusión"""
        if self.modelo is None:
            print("Error: No hay modelo cargado")
            return False
        
        print("Evaluando modelo cargado...")
        
        # Cargar datos de prueba
        imagenes, etiquetas = self.cargar_imagenes_desde_carpetas(ruta_dataset)
        
        if len(imagenes) == 0:
            print("Error: No se encontraron imágenes para evaluar")
            return False
        
        # Preprocesar datos
        imagenes = imagenes.astype('float32') / 255.0
        imagenes = np.expand_dims(imagenes, axis=-1)
        
        # Codificar etiquetas (usar el encoder existente)
        try:
            etiquetas_codificadas = self.label_encoder.transform(etiquetas)
        except ValueError as e:
            print(f"Error: Hay etiquetas desconocidas en el dataset: {e}")
            return False
        
        # Dividir para tener conjunto de validación
        _, X_val, _, y_val = train_test_split(
            imagenes, etiquetas_codificadas, test_size=0.3, random_state=42, stratify=etiquetas_codificadas
        )
        
        # Evaluar
        val_loss, val_accuracy = self.modelo.evaluate(X_val, y_val, verbose=0)
        print(f"Precisión en evaluación: {val_accuracy:.2%}")
        
        # Generar análisis completo
        self.analisis_completo_modelo(X_val, y_val)
        
        return True
    
    def predecir_emocion(self, rostro):
        """Predice la emoción de un rostro detectado"""
        if self.modelo is None:
            return "Sin modelo", 0.0
        
        try:
            # Preprocesar rostro
            rostro_redimensionado = cv2.resize(rostro, (48, 48))
            rostro_normalizado = rostro_redimensionado.astype('float32') / 255.0
            rostro_expandido = np.expand_dims(np.expand_dims(rostro_normalizado, axis=-1), axis=0)
            
            # Predecir
            prediccion = self.modelo.predict(rostro_expandido, verbose=0)
            indice_emocion = np.argmax(prediccion)
            confianza = np.max(prediccion)
            
            emocion = self.label_encoder.inverse_transform([indice_emocion])[0]
            
            return emocion, confianza
        except Exception as e:
            print(f"Error en predicción: {e}")
            return "Error", 0.0
    
    def reconocimiento_tiempo_real(self):
        """Inicia el reconocimiento de emociones en tiempo real"""
        if self.modelo is None:
            print("Error: Primero debes cargar o entrenar un modelo")
            return
        
        print("Iniciando reconocimiento en tiempo real...")
        print("Presiona 'q' para salir, 's' para capturar imagen")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: No se pudo acceder a la cámara")
            return
        
        # Variables para estadísticas
        contador_frames = 0
        tiempo_inicio = time.time()
        emociones_detectadas = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame de la cámara")
                break
            
            # Voltear frame horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar rostros
            rostros = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in rostros:
                # Extraer región del rostro
                rostro = gray[y:y+h, x:x+w]
                
                # Predecir emoción
                emocion, confianza = self.predecir_emocion(rostro)
                
                # Obtener color de la emoción
                color = self.colores_emociones.get(emocion, (255, 255, 255))
                
                # Dibujar rectángulo y texto
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Texto de la emoción y confianza
                texto = f"{emocion}: {confianza:.1%}"
                cv2.putText(frame, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Guardar para estadísticas
                if confianza > 0.5:  # Solo emociones con alta confianza
                    emociones_detectadas.append(emocion)
            
            # Mostrar información adicional
            tiempo_transcurrido = time.time() - tiempo_inicio
            fps = contador_frames / tiempo_transcurrido if tiempo_transcurrido > 0 else 0
            
            # Información en pantalla
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Rostros: {len(rostros)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar emoción más frecuente
            if emociones_detectadas:
                emocion_frecuente = Counter(emociones_detectadas).most_common(1)[0][0]
                cv2.putText(frame, f"Emocion predominante: {emocion_frecuente}", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Reconocimiento de Emociones', frame)
            contador_frames += 1
            
            # Controles de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Guardar captura
                nombre_archivo = f"captura_{int(time.time())}.jpg"
                cv2.imwrite(nombre_archivo, frame)
                print(f"Imagen guardada como {nombre_archivo}")
        
        # Mostrar estadísticas finales
        cap.release()
        cv2.destroyAllWindows()
        
        if emociones_detectadas:
            print("\nEstadísticas de la sesión:")
            contador_emociones = Counter(emociones_detectadas)
            for emocion, cantidad in contador_emociones.most_common():
                porcentaje = (cantidad / len(emociones_detectadas)) * 100
                print(f"   {emocion}: {cantidad} veces ({porcentaje:.1f}%)")
        
        print("Sesión de reconocimiento finalizada")

def menu_principal():
    """Menú principal del programa"""
    reconocedor = ReconocimientoEmociones()
    
    print("="*60)
    print("SISTEMA DE RECONOCIMIENTO DE EMOCIONES FACIALES")
    print("="*60)
    
    while True:
        print("\n¿Qué deseas hacer?")
        print("1. Entrenar un nuevo modelo")
        print("2. Cargar modelo existente")
        print("3. Iniciar reconocimiento en tiempo real")
        print("4. Evaluar modelo cargado (con matriz de confusión)")
        print("5. Salir")
        
        opcion = input("\nSelecciona una opción (1-5): ").strip()
        
        if opcion == '1':
            ruta_dataset = input("\nIngresa la ruta de tu dataset (carpeta con subcarpetas de emociones): ").strip()
            if os.path.exists(ruta_dataset):
                if reconocedor.entrenar_modelo(ruta_dataset):
                    print("\n¡Modelo entrenado exitosamente!")
                    print("Análisis completo generado con matriz de confusión")
                else:
                    print("\nError en el entrenamiento")
            else:
                print("\nError: La ruta especificada no existe")
        
        elif opcion == '2':
            if reconocedor.cargar_modelo():
                print("\n¡Modelo cargado exitosamente!")
            else:
                print("\nNo se pudo cargar el modelo")
        
        elif opcion == '3':
            reconocedor.reconocimiento_tiempo_real()
        
        elif opcion == '4':
            if reconocedor.modelo is None:
                print("\nPrimero debes cargar un modelo (opción 2)")
                continue
            
            ruta_dataset = input("\nIngresa la ruta de tu dataset para evaluación: ").strip()
            if os.path.exists(ruta_dataset):
                if reconocedor.evaluar_modelo_cargado(ruta_dataset):
                    print("\n¡Evaluación completada con matriz de confusión!")
                else:
                    print("\nError en la evaluación")
            else:
                print("\nError: La ruta especificada no existe")
        
        elif opcion == '5':
            print("\nHasta luego!")
            break
        
        else:
            print("\nOpción no válida. Por favor, selecciona 1, 2, 3, 4 o 5.")

if __name__ == "__main__":
    # Verificar dependencias
    try:
        import tensorflow
        print(f"TensorFlow versión: {tensorflow.__version__}")
    except ImportError:
        print("Error: TensorFlow no está instalado")
        print("Instala con: pip install tensorflow")
        exit()
    
    try:
        import sklearn
        print(f"Scikit-learn versión: {sklearn.__version__}")
    except ImportError:
        print("Error: Scikit-learn no está instalado")
        print("Instala con: pip install scikit-learn")
        exit()
    try:
        import seaborn
        print(f"Seaborn versión: {seaborn.__version__}")
    except ImportError:
        print("Error: Seaborn no está instalado")
        print("Instala con: pip install seaborn")
        exit()
    
    # Ejecutar programa principal
    menu_principal()