import re
import math
from collections import Counter

# Lista para almacenar las palabras vacías (stopwords)
STOPWORDS = ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'a', 'ante', 
             'bajo', 'con', 'de', 'desde', 'en', 'entre', 'hacia', 'hasta', 'para', 'por', 
             'según', 'sin', 'sobre', 'tras', 'que', 'es', 'son', 'está', 'están']

class ClasificadorBayesianoSpam:
    def __init__(self):
        self.vocabulario = set()
        self.probabilidad_spam = 0
        self.probabilidad_no_spam = 0
        self.probabilidad_palabras_spam = {}
        self.probabilidad_palabras_no_spam = {}
        self.idf = {}
        self.umbral = 0.5  # Umbral para clasificación
        
    def preprocesar_texto(self, texto):
        """
        Preprocesa el texto: convierte a minúsculas, elimina caracteres especiales,
        elimina stopwords y tokeniza.
        """
        # Convertir a minúsculas
        texto = texto.lower()
        
        # Eliminar caracteres especiales
        texto = re.sub(r'[^\w\s]', '', texto)
        
        # Tokenizar
        tokens = texto.split()
        
        # Eliminar stopwords
        tokens = [token for token in tokens if token not in STOPWORDS]
        
        return tokens
    
    def calcular_frecuencia_terminos(self, tokens):
        """
        Calcula la frecuencia de términos (TF) en un documento.
        """
        contador = Counter()
        for token in tokens:
            contador[token] += 1
        return contador
    
    def calcular_idf(self, corpus_tokens):
        """
        Calcula la frecuencia de documentos inversa (IDF) para cada término en el corpus.
        """
        # Número total de documentos
        num_documentos = len(corpus_tokens)
        
        # Contar documentos donde aparece cada término
        documentos_por_termino = {}
        for doc_tokens in corpus_tokens:
            terminos_unicos = set(doc_tokens)
            for termino in terminos_unicos:
                if termino in documentos_por_termino:
                    documentos_por_termino[termino] += 1
                else:
                    documentos_por_termino[termino] = 1
        
        # Calcular IDF para cada término
        idf = {}
        for termino, num_docs in documentos_por_termino.items():
            idf[termino] = math.log(num_documentos / (1 + num_docs))
        
        return idf
    
    def calcular_tfidf(self, tf, idf):
        """
        Calcula el valor TF-IDF para cada término en un documento.
        """
        tfidf = {}
        for termino, frecuencia in tf.items():
            if termino in idf:
                tfidf[termino] = frecuencia * idf[termino]
        return tfidf
    
    def entrenar(self, correos, etiquetas):
        """
        Entrena el clasificador con los correos y sus etiquetas.
        """
        # Eliminar duplicados
        correos_unicos = []
        etiquetas_unicas = []
        correos_vistos = set()
        
        for i, correo in enumerate(correos):
            if correo not in correos_vistos:
                correos_vistos.add(correo)
                correos_unicos.append(correo)
                etiquetas_unicas.append(etiquetas[i])
        
        correos = correos_unicos
        etiquetas = etiquetas_unicas
        
        # Preprocesar todos los correos
        corpus_tokens = [self.preprocesar_texto(correo) for correo in correos]
        
        # Calcular vocabulario completo
        for tokens in corpus_tokens:
            self.vocabulario.update(tokens)
        
        # Calcular IDF
        self.idf = self.calcular_idf(corpus_tokens)
        
        # Dividir en spam y no spam
        spam_indices = [i for i, etiqueta in enumerate(etiquetas) if etiqueta == 1]
        no_spam_indices = [i for i, etiqueta in enumerate(etiquetas) if etiqueta == 0]
        
        spam_tokens = [corpus_tokens[i] for i in spam_indices]
        no_spam_tokens = [corpus_tokens[i] for i in no_spam_indices]
        
        # Calcular probabilidades previas
        total_correos = len(correos)
        num_spam = len(spam_indices)
        num_no_spam = len(no_spam_indices)
        
        self.probabilidad_spam = num_spam / total_correos
        self.probabilidad_no_spam = num_no_spam / total_correos
        
        # Contar frecuencias de términos en spam y no spam
        contador_spam = Counter()
        contador_no_spam = Counter()
        
        for tokens in spam_tokens:
            contador_spam.update(tokens)
        
        for tokens in no_spam_tokens:
            contador_no_spam.update(tokens)
        
        # Calcular probabilidades de características
        total_palabras_spam = sum(contador_spam.values())
        total_palabras_no_spam = sum(contador_no_spam.values())
        
        # Usar suavizado de Laplace (agregar 1 a cada conteo)
        alpha = 1
        for palabra in self.vocabulario:
            self.probabilidad_palabras_spam[palabra] = (contador_spam[palabra] + alpha) / (total_palabras_spam + alpha * len(self.vocabulario))
            self.probabilidad_palabras_no_spam[palabra] = (contador_no_spam[palabra] + alpha) / (total_palabras_no_spam + alpha * len(self.vocabulario))
    
    def clasificar(self, correo):
        """
        Clasifica un correo como spam (1) o no spam (0).
        """
        tokens = self.preprocesar_texto(correo)
        tf = self.calcular_frecuencia_terminos(tokens)
        tfidf = self.calcular_tfidf(tf, self.idf)
        
        # Calcular probabilidad de las características dado spam
        log_prob_spam = math.log(self.probabilidad_spam) if self.probabilidad_spam > 0 else -float('inf')
        log_prob_no_spam = math.log(self.probabilidad_no_spam) if self.probabilidad_no_spam > 0 else -float('inf')
        
        # Multiplicar por las probabilidades de cada palabra
        for palabra, tfidf_valor in tfidf.items():
            if palabra in self.vocabulario:
                # Usar el valor TF-IDF como peso
                log_prob_spam += math.log(self.probabilidad_palabras_spam[palabra]) * tfidf_valor
                log_prob_no_spam += math.log(self.probabilidad_palabras_no_spam[palabra]) * tfidf_valor
        
        # Convertir de logaritmos a probabilidades
        try:
            prob_spam = math.exp(log_prob_spam)
            prob_no_spam = math.exp(log_prob_no_spam)
        except OverflowError:
            # En caso de desbordamiento, ajustamos los valores
            if log_prob_spam > log_prob_no_spam:
                return 1, 1.0
            else:
                return 0, 0.0
        
        # Normalizar
        probabilidad_posterior = prob_spam / (prob_spam + prob_no_spam) if (prob_spam + prob_no_spam) > 0 else 0.5
        
        # Clasificar según la probabilidad posterior
        return 1 if probabilidad_posterior > self.umbral else 0, probabilidad_posterior
    
    def evaluar(self, correos_test, etiquetas_test):
        """
        Evalúa el modelo con un conjunto de prueba.
        """
        predicciones = []
        probabilidades = []
        
        for correo in correos_test:
            prediccion, probabilidad = self.clasificar(correo)
            predicciones.append(prediccion)
            probabilidades.append(probabilidad)
        
        # Calcular métricas
        verdaderos_positivos = sum(1 for pred, real in zip(predicciones, etiquetas_test) if pred == 1 and real == 1)
        falsos_positivos = sum(1 for pred, real in zip(predicciones, etiquetas_test) if pred == 1 and real == 0)
        verdaderos_negativos = sum(1 for pred, real in zip(predicciones, etiquetas_test) if pred == 0 and real == 0)
        falsos_negativos = sum(1 for pred, real in zip(predicciones, etiquetas_test) if pred == 0 and real == 1)
        
        # Evitar división por cero
        precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos) if (verdaderos_positivos + falsos_positivos) > 0 else 0
        recuperacion = verdaderos_positivos / (verdaderos_positivos + falsos_negativos) if (verdaderos_positivos + falsos_negativos) > 0 else 0
        accuracy = (verdaderos_positivos + verdaderos_negativos) / len(etiquetas_test) if len(etiquetas_test) > 0 else 0
        
        # F1-score
        f1 = 2 * (precision * recuperacion) / (precision + recuperacion) if (precision + recuperacion) > 0 else 0
        
        return {
            'precision': accuracy,
            'precision_score': precision,
            'recuperacion': recuperacion,
            'f1_score': f1,
            'predicciones': predicciones,
            'probabilidades': probabilidades
        }

def leer_correos_desde_txt(ruta_archivo):
    """
    Lee los correos desde un archivo de texto con formato:
    De: remitente
    Asunto: asunto
    Texto del correo
    ---
    """
    correos = []
    etiquetas = []
    datos_completos = []
    
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
            contenido = archivo.read()
            
            # Dividir por el separador de correos (línea con '---')
            bloques_correo = contenido.split('\n---\n')
            
            for bloque in bloques_correo:
                if not bloque.strip():
                    continue
                
                lineas = bloque.strip().split('\n')
                if len(lineas) >= 3:  # Debe tener al menos remitente, asunto y texto
                    # El formato esperado es: De: xxx\nAsunto: xxx\nTexto\nTexto...
                    remitente_linea = lineas[0]
                    asunto_linea = lineas[1]
                    
                    # Extraer remitente y asunto
                    remitente = remitente_linea.replace('De: ', '')
                    asunto = asunto_linea.replace('Asunto: ', '')
                    
                    # El resto es el texto del correo
                    texto = '\n'.join(lineas[2:])
                    
                    # Extraer la parte base del remitente (antes del @)
                    remitente_base = remitente.split('@')[0] if '@' in remitente else remitente
                    
                    # Extraer palabras clave del asunto
                    asunto_base = ' '.join(re.findall(r'\b\w{4,}\b', asunto.lower()))
                    
                    # Combinar remitente, asunto y texto para el clasificador
                    correo_completo = f"{remitente} {asunto} {texto}"
                    correos.append(correo_completo)
                    
                    # Guardar datos completos para mostrar después
                    datos_completos.append({
                        'remitente': remitente,
                        'asunto': asunto,
                        'texto': texto,
                        'remitente_base': remitente_base,
                        'asunto_base': asunto_base
                    })
                    
                    # Etiquetar manualmente para el entrenamiento basado en palabras clave
                    # Esta parte debería ser reemplazada por etiquetas reales si están disponibles
                    es_spam = 0
                    palabras_spam = ["whatsapp", "oferta", "gratis", "premio", "ganador", "urgente", 
                                    "lotería", "millones", "exclusivo", "increíble", "scam", "virus",
                                    "phishing", "malicious"]
                    
                    if any(palabra in correo_completo.lower() for palabra in palabras_spam):
                        es_spam = 1
                        
                    etiquetas.append(es_spam)
    
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
    
    return correos, etiquetas, datos_completos

# Esta función ya no se usa para dividir los datos
def dividir_datos(correos, etiquetas, datos_completos, proporcion_test=0.2):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    # Calcular el número de elementos para el conjunto de prueba
    num_test = int(len(correos) * proporcion_test)
    
    # Dividir los datos
    X_train = correos[:-num_test]
    X_test = correos[-num_test:]
    y_train = etiquetas[:-num_test]
    y_test = etiquetas[-num_test:]
    datos_test = datos_completos[-num_test:]
    
    return X_train, X_test, y_train, y_test, datos_test

def main():
    print("=== DETECTOR DE SPAM CON CLASIFICADOR BAYESIANO ===")
    print("Implementa preprocesamiento, extracción de características y algoritmo bayesiano")
    
    try:
        ruta_archivo = input("Ingrese la ruta del archivo de correos: ")
        if not ruta_archivo:
            ruta_archivo = "D:\Descargas\\PRUEBASSPAM.txt"  # Nombre del archivo por defecto
        
        print(f"Utilizando archivo: {ruta_archivo}")
        
        correos, etiquetas, datos_completos = leer_correos_desde_txt(ruta_archivo)
        
        if correos:
            print(f"\nSe encontraron {len(correos)} correos en el archivo.")
            
            # MODIFICACIÓN: Ya no dividimos los datos, usamos todos los correos
            # para entrenar y evaluar
            
            # Entrenar el clasificador con todos los correos
            print("\nEntrenando el clasificador Bayesiano...")
            clasificador = ClasificadorBayesianoSpam()
            clasificador.entrenar(correos, etiquetas)
            
            # Evaluar todos los correos
            print("\nEvaluando el clasificador en todos los correos...")
            resultados = clasificador.evaluar(correos, etiquetas)
            
            print("\n=== RESULTADOS DE EVALUACIÓN ===")
            print(f"Precisión (accuracy): {resultados['precision']:.4f}")
            print(f"Recuperación (recall): {resultados['recuperacion']:.4f}")
            print(f"Precisión (precision): {resultados['precision_score']:.4f}")
            print(f"F1-Score: {resultados['f1_score']:.4f}")
            
            # Mostrar resultados de clasificación con formato similar al original
            print("\n=== RESULTADOS DE CLASIFICACIÓN ===\n")
            print(f"{'N°':<3} | {'RESULTADO':<8} | {'REMITENTE':<25} | {'ASUNTO':<30} | {'RAZÓN':<65}")
            print("-" * 135)
            
            # MODIFICACIÓN: Usamos datos_completos directamente en lugar de datos_test
            for i, (pred, prob, dato) in enumerate(zip(resultados['predicciones'], resultados['probabilidades'], datos_completos), 1):
                resultado = "SPAM" if pred == 1 else "NO SPAM"
                
                # Truncar strings largos para mejor visualización
                remitente_display = dato['remitente'][:25]
                asunto_display = dato['asunto'][:30]
                
                # Determinar razón
                if pred == 1:
                    razon_display = f"Probabilidad de spam: {prob:.4f}"
                else:
                    razon_display = f"Probabilidad de no spam: {1-prob:.4f}"
                
                print(f"{i:<3} | {resultado:<8} | {remitente_display:<25} | {asunto_display:<30} | {razon_display:<65}")
            
            # Contar resultados
            spam_count = sum(1 for pred in resultados['predicciones'] if pred == 1)
            no_spam_count = sum(1 for pred in resultados['predicciones'] if pred == 0)
            
            print("\n=== RESUMEN ===")
            print(f"Total de correos analizados: {len(resultados['predicciones'])}")
            print(f"Correos clasificados como SPAM: {spam_count}")
            print(f"Correos clasificados como NO SPAM: {no_spam_count}")
            print(f"Porcentaje de SPAM: {spam_count/len(resultados['predicciones'])*100:.2f}%")
            
            # Mostrar detalles
            # MODIFICACIÓN: Pasamos datos_completos en lugar de datos_test
            mostrar_detalles(resultados, datos_completos, clasificador)
        else:
            print("No se encontraron correos en el archivo o el formato es incorrecto.")
    
    except Exception as e:
        print(f"Error al procesar los correos: {e}")

def mostrar_detalles(resultados, datos_completos, clasificador):
    print("\n=== ANÁLISIS DETALLADO ===")
    print("¿Desea ver el análisis detallado de algún correo? (S/N): ", end='')
    respuesta = input().strip().upper()
    
    if respuesta == 'S':
        while True:
            print("\nIngrese el número del correo (o 0 para salir): ", end='')
            try:
                num = int(input().strip())
                if num == 0:
                    break
                
                if 1 <= num <= len(resultados['predicciones']):
                    idx = num - 1
                    dato = datos_completos[idx]
                    prediccion = resultados['predicciones'][idx]
                    probabilidad = resultados['probabilidades'][idx]
                    
                    print("\n=== DETALLE DEL CORREO ===")
                    print(f"Remitente: {dato['remitente']}")
                    print(f"Asunto: {dato['asunto']}")
                    print(f"Clasificación: {'SPAM' if prediccion == 1 else 'NO SPAM'}")
                    print(f"Probabilidad de ser spam: {probabilidad:.4f}")
                    
                    # Mostrar análisis bayesiano
                    tokens = clasificador.preprocesar_texto(f"{dato['remitente']} {dato['asunto']} {dato['texto']}")
                    tf = clasificador.calcular_frecuencia_terminos(tokens)
                    tfidf = clasificador.calcular_tfidf(tf, clasificador.idf)
                    
                    print("\nAnálisis de características:")
                    print(f"- Probabilidad previa de spam: {clasificador.probabilidad_spam:.4f}")
                    print(f"- Probabilidad previa de no spam: {clasificador.probabilidad_no_spam:.4f}")
                    
                    print("\nPalabras más relevantes (TF-IDF):")
                    palabras_importantes = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:5]
                    for palabra, valor in palabras_importantes:
                        prob_spam = clasificador.probabilidad_palabras_spam.get(palabra, 0)
                        prob_no_spam = clasificador.probabilidad_palabras_no_spam.get(palabra, 0)
                        print(f"- {palabra}: TF-IDF={valor:.4f}, P(palabra|Spam)={prob_spam:.4f}, P(palabra|NoSpam)={prob_no_spam:.4f}")
                    
                    print("\nPrimeras 150 caracteres del mensaje:")
                    print(dato['texto'][:150] + "...")
                else:
                    print("Número de correo inválido. Intente de nuevo.")
            except ValueError:
                print("Por favor, ingrese un número válido.")

if __name__ == "__main__":
    main()