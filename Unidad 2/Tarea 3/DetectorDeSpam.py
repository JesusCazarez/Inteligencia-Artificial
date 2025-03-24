import re


# Lista de sitios web conocidos por distribuir malware (ejemplo)
sitios_malware = ["malware.com", "virus.net", "scam.org", "phishing.com", "malicious.site"]

# Remitentes confiables (para razonamiento no monótono)
remitentes_confiables = ["soporte@empresa.com", "sistema@banco.com", "notificaciones@amazon.com"]

# Dominios confiables (para razonamiento monótono)
dominios_conocidos = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "empresa.com", 
                      "banco.com", "amazon.com", "spotify.com", "netflix.com", "universidad.edu"]

# Palabras que suelen aparecer en mensajes legítimos (para razonamiento no monótono)
palabras_legitimas = ["factura", "confirmación", "recordatorio", "actualización", "seguridad", 
                     "cuenta", "reserva", "pedido", "cliente", "reunión"]

# Función para aplicar las reglas monótonas (no cambian con nueva información)
def aplicar_reglas_monotonas(texto, remitente, asunto):
    razones_spam = []
    
    # Regla 1: Si contiene la palabra clave "Whatsapp" (Regla monótona)
    if "whatsapp" in texto.lower():
        razones_spam.append("Regla 1 (M): Contiene la palabra clave 'Whatsapp'")
    
    # Regla 2: Si contiene enlace a sitio web conocido por distribuir malware (Regla monótona)
    for sitio in sitios_malware:
        if sitio in texto.lower():
            razones_spam.append(f"Regla 2 (M): Contiene enlace a sitio malicioso '{sitio}'")
            break
    
    # Regla 5: Si está mal escrito o tiene errores gramaticales (versión monótona)
    # Simplificación: verificamos patrones que SIEMPRE indican spam
    if texto.count('!!!') > 2 or texto.count('$$$') > 0:
        razones_spam.append("Regla 5 (M): Patrones claros de errores gramaticales")
    
    # Las reglas monótonas son definitivas - si alguna se cumple, es spam
    if razones_spam:
        return True, razones_spam
    
    return False, []

# Función para aplicar reglas no monótonas (pueden cambiar con nueva información)
def aplicar_reglas_no_monotonas(texto, remitente, asunto, remitente_base, asunto_base):
    razones_spam = []
    excepciones = []
    
    # Regla 3: Si tiene un remitente desconocido (regla no monótona)
    es_remitente_conocido = False
    for dominio in dominios_conocidos:
        if dominio in remitente.lower():
            es_remitente_conocido = True
            break
    
    if not es_remitente_conocido and "@" in remitente:
        razones_spam.append("Regla 3 (NM): Remitente desconocido")
    
    # Regla 4: Si tiene un asunto demasiado bueno para ser verdad (regla no monótona)
    palabras_sospechosas = ["gratis", "ganar", "premio", "millones", "oferta", "urgente", 
                           "dinero", "lotería", "ganador", "exclusivo", "increíble"]
    
    for palabra in palabras_sospechosas:
        if palabra in asunto.lower():
            razones_spam.append(f"Regla 4 (NM): Asunto sospechoso (contiene '{palabra}')")
            break
    
    # Versión no monótona de Regla 5: Símbolos excesivos pero no concluyente
    simbolos = len(re.findall(r'[!@#$%^&*()_+{}|:"<>?~]', texto))
    if simbolos > len(texto) * 0.05 and simbolos <= len(texto) * 0.1:  # Entre 5% y 10% son símbolos
        razones_spam.append("Regla 5 (NM): Cantidad sospechosa de símbolos")
    
    # --- EXCEPCIONES (razonamiento no monótono) ---
    
    # Excepción 1: Si el remitente está en la lista de confiables, se anula la regla 3
    if remitente in remitentes_confiables:
        excepciones.append("Excepción 1: Remitente está en lista de confianza")
    
    # Excepción 2: Si contiene palabras típicas de comunicaciones legítimas
    palabras_legitimas_encontradas = [palabra for palabra in palabras_legitimas if palabra in texto.lower()]
    if len(palabras_legitimas_encontradas) >= 3:
        excepciones.append(f"Excepción 2: Contiene múltiples palabras de comunicación legítima: {', '.join(palabras_legitimas_encontradas[:3])}")
    
    # Excepción 3: Comunicación esperada/regular
    if remitente_base in texto.lower() or asunto_base in texto.lower():
        excepciones.append("Excepción 3: El correo hace referencia al remitente o asunto en el contenido")
    
    return razones_spam, excepciones

def tomar_decision_final(es_spam_monotono, razones_monotonas, razones_no_monotonas, excepciones):
    # Caso 1: Si las reglas monótonas indican spam, es definitivamente spam
    if es_spam_monotono:
        return 1, razones_monotonas
    
    # Caso 2: Si hay razones no monótonas pero hay excepciones que las contradicen
    if razones_no_monotonas and excepciones:
        # Si hay más excepciones que razones no monótonas, no es spam
        if len(excepciones) >= len(razones_no_monotonas):
            return 0, ["No es spam: Las excepciones superan a las reglas no monótonas"] + excepciones
        else:
            return 1, razones_no_monotonas + ["(con excepciones: " + "; ".join(excepciones) + ")"]
    
    # Caso 3: Si hay razones no monótonas y no hay excepciones, es spam
    if razones_no_monotonas and not excepciones:
        return 1, razones_no_monotonas
    
    # Caso 4: No hay razones para considerar spam
    return 0, ["No es spam: No cumple ninguna regla de spam"]

def leer_correos_desde_txt(ruta_archivo):
    correos = []
    
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
                    
                    correos.append({
                        'remitente': remitente,
                        'asunto': asunto,
                        'texto': texto,
                        'remitente_base': remitente_base,
                        'asunto_base': asunto_base
                    })
    
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
    
    return correos

def clasificar_correos(correos):
    resultados = []
    spam_count = 0
    no_spam_count = 0
    
    print("\n=== RESULTADOS DE CLASIFICACIÓN ===\n")
    print(f"{'N°':<3} | {'RESULTADO':<8} | {'REMITENTE':<25} | {'ASUNTO':<30} | {'RAZÓN':<65}")
    print("-" * 135)
    
    for i, correo in enumerate(correos, 1):
        # Aplicar razonamiento monótono (conclusiones fijas)
        es_spam_monotono, razones_monotonas = aplicar_reglas_monotonas(
            correo['texto'], correo['remitente'], correo['asunto'])
        
        # Aplicar razonamiento no monótono (conclusiones revisables)
        razones_no_monotonas, excepciones = aplicar_reglas_no_monotonas(
            correo['texto'], correo['remitente'], correo['asunto'], 
            correo['remitente_base'], correo['asunto_base'])
        
        # Tomar decisión final considerando ambos tipos de razonamiento
        es_spam, razones_finales = tomar_decision_final(
            es_spam_monotono, razones_monotonas, razones_no_monotonas, excepciones)
        
        if es_spam:
            resultado = "SPAM"
            spam_count += 1
        else:
            resultado = "NO SPAM"
            no_spam_count += 1
        
        # Truncar strings largos para mejor visualización
        remitente_display = correo['remitente'][:25]
        asunto_display = correo['asunto'][:30]
        # Tomar solo la primera razón para mostrar
        razon_display = razones_finales[0][:65] if razones_finales else "Sin razón específica"
        
        print(f"{i:<3} | {resultado:<8} | {remitente_display:<25} | {asunto_display:<30} | {razon_display:<65}")
        
        resultados.append({
            'correo': correo,
            'es_spam': es_spam,
            'razones': razones_finales,
            'reglas_monotonas': razones_monotonas,
            'reglas_no_monotonas': razones_no_monotonas,
            'excepciones': excepciones
        })
    
    print("\n=== RESUMEN ===")
    print(f"Total de correos analizados: {len(correos)}")
    print(f"Correos clasificados como SPAM: {spam_count}")
    print(f"Correos clasificados como NO SPAM: {no_spam_count}")
    print(f"Porcentaje de SPAM: {spam_count/len(correos)*100:.2f}%")
    
    return resultados

def mostrar_detalles(resultados):
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
                
                if 1 <= num <= len(resultados):
                    resultado = resultados[num-1]
                    print("\n=== DETALLE DEL CORREO ===")
                    print(f"Remitente: {resultado['correo']['remitente']}")
                    print(f"Asunto: {resultado['correo']['asunto']}")
                    print(f"Clasificación: {'SPAM' if resultado['es_spam'] else 'NO SPAM'}")
                    
                    print("\nReglas monótonas aplicadas:")
                    if resultado['reglas_monotonas']:
                        for regla in resultado['reglas_monotonas']:
                            print(f"- {regla}")
                    else:
                        print("- Ninguna regla monótona activada")
                    
                    print("\nReglas no monótonas aplicadas:")
                    if resultado['reglas_no_monotonas']:
                        for regla in resultado['reglas_no_monotonas']:
                            print(f"- {regla}")
                    else:
                        print("- Ninguna regla no monótona activada")
                    
                    print("\nExcepciones aplicadas:")
                    if resultado['excepciones']:
                        for excepcion in resultado['excepciones']:
                            print(f"- {excepcion}")
                    else:
                        print("- Ninguna excepción aplicada")
                    
                    print("\nRazones finales de la decisión:")
                    for razon in resultado['razones']:
                        print(f"- {razon}")
                    
                    print("\nPrimeras 150 caracteres del mensaje:")
                    print(resultado['correo']['texto'][:150] + "...")
                else:
                    print("Número de correo inválido. Intente de nuevo.")
            except ValueError:
                print("Por favor, ingrese un número válido.")

# Función principal
def main():
    print("=== DETECTOR DE SPAM CON RAZONAMIENTO MONÓTONO Y NO MONÓTONO ===")
    print("Basado en las 5 reglas del sistema de clasificación")
    
    try:
        ruta_archivo = "D:\Descargas\\PRUEBASSPAM.txt"  # Nombre del archivo por defecto
        print(f"Utilizando archivo: {ruta_archivo}")
        
        correos = leer_correos_desde_txt(ruta_archivo)
        
        if correos:
            print(f"\nSe encontraron {len(correos)} correos en el archivo.")
            resultados = clasificar_correos(correos)
            mostrar_detalles(resultados)
        else:
            print("No se encontraron correos en el archivo o el formato es incorrecto.")
    
    except Exception as e:
        print(f"Error al procesar los correos: {e}")

if __name__ == "__main__":
    main()