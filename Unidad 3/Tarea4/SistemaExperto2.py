"""
Sistema Experto para Diagnóstico de Fallas Vehiculares - Versión Simplificada
Este programa implementa un sistema experto básico para diagnosticar fallas vehiculares
utilizando una interfaz gráfica simple y un motor de inferencia basado en reglas.

VERSIÓN CORREGIDA: Permite seleccionar síntomas y características simultáneamente.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import os
from datetime import datetime

class SistemaExpertoVehicular:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Experto para Diagnóstico de Fallas Vehiculares")
        self.root.geometry("900x600")
        
        # Variables
        self.current_vehicle = None
        
        # Cargar base de conocimiento
        self.knowledge_base = self.load_knowledge_base()
        
        # Crear interfaz
        self.create_interface()
    
    def load_knowledge_base(self):
        """Carga la base de conocimiento desde un archivo JSON o crea una por defecto"""
        try:
            if os.path.exists("base_conocimiento.json"):
                with open("base_conocimiento.json", "r", encoding="utf-8") as file:
                    return json.load(file)
            else:
                # Crear base de conocimiento por defecto
                kb = self.create_default_knowledge_base()
                with open("base_conocimiento.json", "w", encoding="utf-8") as file:
                    json.dump(kb, file, indent=4, ensure_ascii=False)
                return kb
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la base de conocimiento: {e}")
            return self.create_default_knowledge_base()
    
    def create_default_knowledge_base(self):
        """Crea una base de conocimiento por defecto"""
        return {
            "rules": [
                {
                    "id": "R1",
                    "symptoms": ["no_enciende"],
                    "characteristics": ["bateria_cargada", "sin_sonido_llave"],
                    "diagnosis": "falla_motor_arranque",
                    "confidence": 0.9
                },
                {
                    "id": "R2",
                    "symptoms": ["no_enciende"],
                    "characteristics": ["sonido_click", "luces_funcionan"],
                    "diagnosis": "falla_solenoide_arranque",
                    "confidence": 0.85
                },
                {
                    "id": "R3",
                    "symptoms": ["enciende_y_se_apaga"],
                    "characteristics": ["olor_combustible"],
                    "diagnosis": "sistema_combustible_inundado",
                    "confidence": 0.8
                },
                {
                    "id": "R4",
                    "symptoms": ["vibracion_excesiva"],
                    "characteristics": ["ruido_metalico"],
                    "diagnosis": "soportes_motor_dañados",
                    "confidence": 0.7
                },
                {
                    "id": "R5",
                    "symptoms": ["humo_escape"],
                    "characteristics": ["humo_blanco", "olor_dulce"],
                    "diagnosis": "daño_junta_culata",
                    "confidence": 0.9
                },
                {
                    "id": "R6",
                    "symptoms": ["ruido_al_frenar"],
                    "characteristics": ["ruido_agudo", "vibracion_pedal"],
                    "diagnosis": "pastillas_freno_desgastadas",
                    "confidence": 0.9
                },
                {
                    "id": "R7",
                    "symptoms": ["luces_parpadean"],
                    "characteristics": ["bateria_pierde_carga"],
                    "diagnosis": "alternador_defectuoso",
                    "confidence": 0.9
                },
                {
                    "id": "R8",
                    "symptoms": ["pedal_freno_esponjoso"],
                    "characteristics": ["nivel_liquido_disminuye"],
                    "diagnosis": "fuga_sistema_hidraulico_frenos",
                    "confidence": 0.9
                },
                {
                    "id": "R9",
                    "symptoms": ["rendimiento_combustible_bajo"],
                    "characteristics": ["aceleracion_irregular"],
                    "diagnosis": "filtro_combustible_obstruido",
                    "confidence": 0.75
                },
                {
                    "id": "R10",
                    "symptoms": ["dificultad_cambiar_velocidades"],
                    "characteristics": ["ruido_al_engranar"],
                    "diagnosis": "embrague_desgastado",
                    "confidence": 0.85
                }
            ],
            "symptoms": [
                {"id": "no_enciende", "name": "No enciende", "description": "El vehículo no arranca"},
                {"id": "enciende_y_se_apaga", "name": "Enciende y se apaga", "description": "El motor enciende pero se apaga inmediatamente"},
                {"id": "vibracion_excesiva", "name": "Vibración excesiva", "description": "El vehículo presenta vibración anormal"},
                {"id": "humo_escape", "name": "Humo en el escape", "description": "Sale humo del escape del vehículo"},
                {"id": "ruido_al_frenar", "name": "Ruido al frenar", "description": "Se escucha ruido cuando se aplican los frenos"},
                {"id": "luces_parpadean", "name": "Luces parpadean", "description": "Las luces del vehículo parpadean intermitentemente"},
                {"id": "pedal_freno_esponjoso", "name": "Pedal de freno esponjoso", "description": "El pedal de freno se siente esponjoso o blando"},
                {"id": "rendimiento_combustible_bajo", "name": "Bajo rendimiento de combustible", "description": "El vehículo consume más combustible de lo normal"},
                {"id": "dificultad_cambiar_velocidades", "name": "Dificultad para cambiar velocidades", "description": "Cuesta cambiar de velocidad o se siente resistencia"}
            ],
            "characteristics": [
                {"id": "bateria_cargada", "name": "Batería cargada", "description": "La batería tiene carga suficiente"},
                {"id": "sin_sonido_llave", "name": "Sin sonido al girar llave", "description": "No se escucha ningún sonido al girar la llave"},
                {"id": "sonido_click", "name": "Sonido de click", "description": "Se escucha un click al girar la llave"},
                {"id": "luces_funcionan", "name": "Luces funcionan", "description": "Las luces del vehículo funcionan correctamente"},
                {"id": "olor_combustible", "name": "Olor a combustible", "description": "Se percibe olor a combustible en o alrededor del vehículo"},
                {"id": "ruido_metalico", "name": "Ruido metálico", "description": "Se escucha un ruido metálico en el motor"},
                {"id": "humo_blanco", "name": "Humo blanco", "description": "El humo que sale del escape es blanco"},
                {"id": "olor_dulce", "name": "Olor dulce", "description": "Se percibe un olor dulce (como anticongelante)"},
                {"id": "ruido_agudo", "name": "Ruido agudo", "description": "Se escucha un ruido agudo o chillido"},
                {"id": "vibracion_pedal", "name": "Vibración en el pedal", "description": "Se siente vibración en el pedal de freno"},
                {"id": "bateria_pierde_carga", "name": "Batería pierde carga", "description": "La batería pierde carga rápidamente"},
                {"id": "nivel_liquido_disminuye", "name": "Nivel de líquido disminuye", "description": "El nivel de líquido de frenos disminuye con el tiempo"},
                {"id": "aceleracion_irregular", "name": "Aceleración irregular", "description": "El vehículo acelera de forma irregular o entrecortada"},
                {"id": "ruido_al_engranar", "name": "Ruido al engranar", "description": "Se escucha ruido al intentar engranar una marcha"}
            ],
            "diagnoses": [
                {
                    "id": "falla_motor_arranque",
                    "name": "Falla en el motor de arranque",
                    "description": "El motor de arranque no funciona correctamente",
                    "severity": "media",
                    "repair_cost": 250,
                    "repair_time": 2,
                    "recommendations": [
                        "Revisar conexiones eléctricas del motor de arranque",
                        "Verificar si el motor de arranque gira libremente",
                        "Posible reemplazo del motor de arranque"
                    ]
                },
                {
                    "id": "falla_solenoide_arranque",
                    "name": "Falla en el solenoide del arranque",
                    "description": "El solenoide del arranque no funciona correctamente",
                    "severity": "baja",
                    "repair_cost": 150,
                    "repair_time": 1.5,
                    "recommendations": [
                        "Revisar conexiones eléctricas del solenoide",
                        "Verificar voltaje en el solenoide",
                        "Posible reemplazo del solenoide"
                    ]
                },
                {
                    "id": "sistema_combustible_inundado",
                    "name": "Sistema de combustible inundado",
                    "description": "Hay exceso de combustible en el sistema",
                    "severity": "baja",
                    "repair_cost": 50,
                    "repair_time": 0.5,
                    "recommendations": [
                        "Esperar unos minutos antes de intentar arrancar de nuevo",
                        "Presionar el acelerador a fondo al arrancar",
                        "Revisar sistema de inyección si es recurrente"
                    ]
                },
                {
                    "id": "soportes_motor_dañados",
                    "name": "Soportes de motor dañados",
                    "description": "Los soportes que sostienen el motor están dañados",
                    "severity": "media",
                    "repair_cost": 300,
                    "repair_time": 3,
                    "recommendations": [
                        "Inspeccionar todos los soportes del motor",
                        "Reemplazar los soportes dañados",
                        "Verificar alineación del motor después de la reparación"
                    ]
                },
                {
                    "id": "daño_junta_culata",
                    "name": "Daño en la junta de culata",
                    "description": "La junta entre el bloque del motor y la culata está dañada",
                    "severity": "alta",
                    "repair_cost": 1200,
                    "repair_time": 8,
                    "recommendations": [
                        "Reemplazar la junta de culata",
                        "Verificar planitud de culata",
                        "Revisar sistema de refrigeración",
                        "Verificar posibles causas de sobrecalentamiento"
                    ]
                },
                {
                    "id": "pastillas_freno_desgastadas",
                    "name": "Pastillas de freno desgastadas",
                    "description": "Las pastillas de freno están desgastadas y necesitan reemplazo",
                    "severity": "media",
                    "repair_cost": 200,
                    "repair_time": 1.5,
                    "recommendations": [
                        "Reemplazar las pastillas de freno",
                        "Verificar el estado de los discos",
                        "Comprobar el funcionamiento de los calibradores"
                    ]
                },
                {
                    "id": "alternador_defectuoso",
                    "name": "Alternador defectuoso",
                    "description": "El alternador no está cargando correctamente la batería",
                    "severity": "alta",
                    "repair_cost": 400,
                    "repair_time": 2.5,
                    "recommendations": [
                        "Verificar voltaje de salida del alternador",
                        "Revisar correas y conexiones",
                        "Reemplazar el alternador si es necesario"
                    ]
                },
                {
                    "id": "fuga_sistema_hidraulico_frenos",
                    "name": "Fuga en el sistema hidráulico de frenos",
                    "description": "Hay una fuga en el sistema hidráulico de frenos",
                    "severity": "alta",
                    "repair_cost": 300,
                    "repair_time": 2.5,
                    "recommendations": [
                        "Localizar la fuga en el sistema",
                        "Reemplazar componentes dañados",
                        "Purgar el sistema de frenos",
                        "Verificar funcionamiento"
                    ]
                },
                {
                    "id": "filtro_combustible_obstruido",
                    "name": "Filtro de combustible obstruido",
                    "description": "El filtro de combustible está obstruido y restringe el flujo",
                    "severity": "baja",
                    "repair_cost": 100,
                    "repair_time": 1,
                    "recommendations": [
                        "Reemplazar el filtro de combustible",
                        "Revisar calidad del combustible utilizado",
                        "Verificar si hay contaminación en el tanque"
                    ]
                },
                {
                    "id": "embrague_desgastado",
                    "name": "Embrague desgastado",
                    "description": "El disco de embrague está desgastado y requiere reemplazo",
                    "severity": "media",
                    "repair_cost": 800,
                    "repair_time": 6,
                    "recommendations": [
                        "Reemplazar kit de embrague completo",
                        "Verificar volante de inercia",
                        "Revisar sistema hidráulico del embrague"
                    ]
                }
            ],
            "severity_values": {
                "baja": 1,
                "media": 3,
                "alta": 5
            }
        }
    
    def create_interface(self):
        """Crea la interfaz gráfica"""
        # Marco para información del vehículo
        vehicle_frame = ttk.LabelFrame(self.root, text="Información del Vehículo")
        vehicle_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Grid para campos
        ttk.Label(vehicle_frame, text="Marca:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.make_entry = ttk.Entry(vehicle_frame, width=15)
        self.make_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(vehicle_frame, text="Modelo:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.model_entry = ttk.Entry(vehicle_frame, width=15)
        self.model_entry.grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(vehicle_frame, text="Año:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.year_entry = ttk.Entry(vehicle_frame, width=15)
        self.year_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(vehicle_frame, text="Kilometraje:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.mileage_entry = ttk.Entry(vehicle_frame, width=15)
        self.mileage_entry.grid(row=1, column=3, padx=5, pady=5)
        
        ttk.Button(
            vehicle_frame, 
            text="Guardar Información", 
            command=self.save_vehicle_info
        ).grid(row=1, column=4, padx=5, pady=5)
        
        # Panel con dos columnas para síntomas y características
        panel_frame = ttk.Frame(self.root)
        panel_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Marco para síntomas
        symptoms_frame = ttk.LabelFrame(panel_frame, text="Síntomas")
        symptoms_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Listbox con scrollbar para síntomas (usando Checkbuttons para mejorar selección múltiple)
        self.symptoms_var = {}
        symptoms_canvas = tk.Canvas(symptoms_frame)
        symptoms_scrollbar = ttk.Scrollbar(symptoms_frame, orient="vertical", command=symptoms_canvas.yview)
        symptoms_scrollable_frame = ttk.Frame(symptoms_canvas)
        
        symptoms_scrollable_frame.bind(
            "<Configure>",
            lambda e: symptoms_canvas.configure(scrollregion=symptoms_canvas.bbox("all"))
        )
        
        symptoms_canvas.create_window((0, 0), window=symptoms_scrollable_frame, anchor="nw")
        symptoms_canvas.configure(yscrollcommand=symptoms_scrollbar.set)
        
        # Añadir checkbuttons para síntomas
        for i, symptom in enumerate(self.knowledge_base["symptoms"]):
            self.symptoms_var[symptom["id"]] = tk.BooleanVar()
            ttk.Checkbutton(
                symptoms_scrollable_frame, 
                text=symptom["name"],
                variable=self.symptoms_var[symptom["id"]]
            ).grid(row=i, column=0, sticky="w", padx=5, pady=2)
        
        symptoms_canvas.pack(side="left", fill="both", expand=True)
        symptoms_scrollbar.pack(side="right", fill="y")
        
        # Marco para características
        characteristics_frame = ttk.LabelFrame(panel_frame, text="Características")
        characteristics_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Listbox con scrollbar para características (usando Checkbuttons)
        self.characteristics_var = {}
        char_canvas = tk.Canvas(characteristics_frame)
        char_scrollbar = ttk.Scrollbar(characteristics_frame, orient="vertical", command=char_canvas.yview)
        char_scrollable_frame = ttk.Frame(char_canvas)
        
        char_scrollable_frame.bind(
            "<Configure>",
            lambda e: char_canvas.configure(scrollregion=char_canvas.bbox("all"))
        )
        
        char_canvas.create_window((0, 0), window=char_scrollable_frame, anchor="nw")
        char_canvas.configure(yscrollcommand=char_scrollbar.set)
        
        # Añadir checkbuttons para características
        for i, characteristic in enumerate(self.knowledge_base["characteristics"]):
            self.characteristics_var[characteristic["id"]] = tk.BooleanVar()
            ttk.Checkbutton(
                char_scrollable_frame, 
                text=characteristic["name"],
                variable=self.characteristics_var[characteristic["id"]]
            ).grid(row=i, column=0, sticky="w", padx=5, pady=2)
        
        char_canvas.pack(side="left", fill="both", expand=True)
        char_scrollbar.pack(side="right", fill="y")
        
        # Botones
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(
            button_frame, 
            text="Diagnosticar", 
            command=self.diagnose
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(
            button_frame, 
            text="Limpiar Selección", 
            command=self.clear_selection
        ).pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Marco para resultados
        results_frame = ttk.LabelFrame(self.root, text="Resultados del Diagnóstico")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Área de texto para resultados con scrollbar
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Mensaje inicial
        self.results_text.insert(tk.END, "Introduzca la información del vehículo y seleccione los síntomas observados para realizar un diagnóstico.")
    
    def save_vehicle_info(self):
        """Guarda la información del vehículo"""
        make = self.make_entry.get().strip()
        model = self.model_entry.get().strip()
        year_str = self.year_entry.get().strip()
        mileage_str = self.mileage_entry.get().strip()
        
        # Validar campos
        if not all([make, model, year_str, mileage_str]):
            messagebox.showwarning("Datos incompletos", "Por favor complete todos los campos del vehículo.")
            return
        
        try:
            year = int(year_str)
            mileage = int(mileage_str)
            
            if year < 1900 or year > 2030:
                messagebox.showwarning("Año inválido", "Por favor ingrese un año válido.")
                return
            
            if mileage < 0:
                messagebox.showwarning("Kilometraje inválido", "El kilometraje no puede ser negativo.")
                return
            
        except ValueError:
            messagebox.showwarning("Datos inválidos", "El año y el kilometraje deben ser números.")
            return
        
        # Guardar la información
        self.current_vehicle = {
            "make": make,
            "model": model,
            "year": year,
            "mileage": mileage
        }
        
        messagebox.showinfo("Información guardada", 
                           f"Información del vehículo guardada:\n{make} {model} {year}\nKilometraje: {mileage} km")
    
    def clear_selection(self):
        """Limpia la selección de síntomas y características"""
        # Desmarcar todos los checkbuttons
        for var in self.symptoms_var.values():
            var.set(False)
        
        for var in self.characteristics_var.values():
            var.set(False)
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Introduzca la información del vehículo y seleccione los síntomas observados para realizar un diagnóstico.")
    
    def diagnose(self):
        """Realiza el diagnóstico basado en los síntomas y características seleccionados"""
        if not self.current_vehicle:
            messagebox.showwarning("Vehículo no definido", "Por favor ingrese la información del vehículo primero.")
            return
        
        # Obtener síntomas seleccionados
        selected_symptoms = [symptom_id for symptom_id, var in self.symptoms_var.items() if var.get()]
        
        # Obtener características seleccionadas
        selected_characteristics = [char_id for char_id, var in self.characteristics_var.items() if var.get()]
        
        if not selected_symptoms:
            messagebox.showwarning("Sin síntomas", "Por favor seleccione al menos un síntoma.")
            return
        
        # Realizar el diagnóstico
        results = self.inference_engine(selected_symptoms, selected_characteristics)
        
        # Mostrar resultados
        self.show_results(results, selected_symptoms, selected_characteristics)
    
    def inference_engine(self, selected_symptoms, selected_characteristics):
        """Motor de inferencia que aplica las reglas a los síntomas y características"""
        results = []
        
        # Para cada regla en la base de conocimiento
        for rule in self.knowledge_base["rules"]:
            # Verificar si todos los síntomas de la regla están en los síntomas seleccionados
            symptoms_match = all(symptom in selected_symptoms for symptom in rule["symptoms"])
            
            # Si no hay síntomas que coincidan, continuar con la siguiente regla
            if not symptoms_match:
                continue
            
            # Verificar si todas las características de la regla están en las características seleccionadas
            characteristics_match = all(char in selected_characteristics for char in rule["characteristics"])
            
            # Si tanto los síntomas como las características coinciden
            if symptoms_match and characteristics_match:
                # Buscar el diagnóstico correspondiente
                diagnosis = next((d for d in self.knowledge_base["diagnoses"] if d["id"] == rule["diagnosis"]), None)
                
                if diagnosis:
                    # Calcular puntuación
                    severity_value = self.knowledge_base["severity_values"].get(diagnosis["severity"], 1)
                    score = rule["confidence"] * (1 + (severity_value / 10))
                    
                    # Agregar a resultados
                    results.append({
                        "rule_id": rule["id"],
                        "diagnosis": diagnosis,
                        "confidence": rule["confidence"],
                        "score": score
                    })
        
        # Ordenar por puntuación (de mayor a menor)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
    
    def show_results(self, results, selected_symptoms, selected_characteristics):
        """Muestra los resultados del diagnóstico"""
        self.results_text.delete(1.0, tk.END)
        
        if not results:
            self.results_text.insert(tk.END, "No se encontraron diagnósticos que coincidan con los síntomas y características seleccionados.\n\n")
            self.results_text.insert(tk.END, "Recomendaciones:\n")
            self.results_text.insert(tk.END, "1. Intente seleccionar síntomas adicionales o diferentes.\n")
            self.results_text.insert(tk.END, "2. Verifique si las características seleccionadas son correctas.\n")
            self.results_text.insert(tk.END, "3. Consulte con un mecánico profesional para un diagnóstico más detallado.\n")
            return
        
        # Mostrar información del vehículo
        vehicle = self.current_vehicle
        self.results_text.insert(tk.END, f"RESULTADOS PARA: {vehicle['make']} {vehicle['model']} {vehicle['year']}\n")
        self.results_text.insert(tk.END, f"Kilometraje: {vehicle['mileage']} km\n")
        self.results_text.insert(tk.END, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # Mostrar síntomas seleccionados
        symptom_names = [s["name"] for s in self.knowledge_base["symptoms"] if s["id"] in selected_symptoms]
        self.results_text.insert(tk.END, "SÍNTOMAS SELECCIONADOS:\n")
        for name in symptom_names:
            self.results_text.insert(tk.END, f"- {name}\n")
        
        # Mostrar características seleccionadas si hay
        if selected_characteristics:
            char_names = [c["name"] for c in self.knowledge_base["characteristics"] if c["id"] in selected_characteristics]
            self.results_text.insert(tk.END, "\nCARACTERÍSTICAS SELECCIONADAS:\n")
            for name in char_names:
                self.results_text.insert(tk.END, f"- {name}\n")
        
        # Mostrar diagnósticos
        self.results_text.insert(tk.END, "\nDIAGNÓSTICOS POSIBLES:\n")
        
        for i, result in enumerate(results):
            diagnosis = result["diagnosis"]
            confidence = result["confidence"] * 100
            
            self.results_text.insert(tk.END, f"{i+1}. {diagnosis['name']} (Confianza: {confidence:.1f}%)\n")
            self.results_text.insert(tk.END, f"   Descripción: {diagnosis['description']}\n")
            self.results_text.insert(tk.END, f"   Severidad: {diagnosis['severity'].upper()}\n")
            self.results_text.insert(tk.END, f"   Costo estimado: ${diagnosis['repair_cost']}\n")
            self.results_text.insert(tk.END, f"   Tiempo estimado: {diagnosis['repair_time']} hora(s)\n")
            
            self.results_text.insert(tk.END, "   Recomendaciones:\n")
            for rec in diagnosis["recommendations"]:
                self.results_text.insert(tk.END, f"     - {rec}\n")
            
            self.results_text.insert(tk.END, "\n")
        
        # Mensaje final
        self.results_text.insert(tk.END, "NOTA: Este diagnóstico es una orientación basada en los síntomas proporcionados.\n")
        self.results_text.insert(tk.END, "Se recomienda consultar con un mecánico profesional para una evaluación completa.")

# Función principal
def main():
    root = tk.Tk()
    app = SistemaExpertoVehicular(root)
    root.mainloop()


if __name__ == "__main__":
    main()