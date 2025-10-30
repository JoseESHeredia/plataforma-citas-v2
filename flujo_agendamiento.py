import gspread
from google.oauth2.service_account import Credentials
import csv
import os # Lo mantenemos para la función de persistir
import json # Necesario para los Secrets

# ===== Constantes =====
# Apuntan a los archivos CSV de backup
PACIENTES_CSV = "data/Pacientes.csv"
CITAS_CSV = "data/Citas.csv"

# ===== Conexión a Google Sheets (Tarea S2-04) =====
try:
    alcances = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    # --- BLOQUE CORREGIDO (INDENTACIÓN) ---
    google_creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
    if not google_creds_json:
        print("Secret no encontrado, usando credenciales.json local...")
        cred = Credentials.from_service_account_file("credenciales.json", scopes=alcances)
    else:
        print("Cargando credenciales desde Secret...")
        cred_dict = json.loads(google_creds_json)
        cred = Credentials.from_service_account_info(cred_dict, scopes=alcances)
    # --- FIN CORRECCIÓN ---

    # Estas líneas van DESPUÉS del if/else, pero DENTRO del try
    cliente = gspread.authorize(cred)
    documento = cliente.open("Base de Datos Citas (Proyecto Voz y Chat)")
    pacientes_sheet = documento.worksheet("Pacientes")
    citas_sheet = documento.worksheet("Citas")

    print("✅ Conexión exitosa a Google Sheets.")
    
except Exception as e:
    print(f"❌ Error conectando a Google Sheets: {e}")
    print("Revisa 'credenciales.json' y los permisos de la hoja.")
    # Si falla la conexión, creamos placeholders para que el test no falle
    pacientes_sheet = None
    citas_sheet = None


# ===== Función para generar ID único (Modo Google Sheets) =====
def generar_id(prefijo, hoja):
    """
    Lee la columna 1 de una Google Sheet, encuentra el ID más alto
    y devuelve el siguiente ID formateado.
    """
    if hoja is None:
        return f"{prefijo}000" # Error
        
    try:
        # Asumiendo que la columna 1 (A) es 'ID_Paciente' o 'ID_Cita'
        ids = hoja.col_values(1)[1:]  # Ignorar encabezado
    except gspread.exceptions.APIError as e:
        print(f"Error leyendo la hoja para generar ID: {e}")
        return f"{prefijo}000"

    if not ids:
        return f"{prefijo}001"
    
    numeros = [int(id[len(prefijo):]) for id in ids if id.startswith(prefijo) and id[len(prefijo):].isdigit()]
    max_num = max(numeros) if numeros else 0
    return f"{prefijo}{max_num + 1:03d}"


# ===== Lógica de negocio (Médicos) - Sin cambios =====
def asignar_especialidad(medico):
    especialidades = {
        "Dr.Vega": "Endodoncia",
        "Dr.Perez": "Periodoncia",
        "Dra.Morales": "Ortodoncia",
        "Dr.Castro": "Protesis dental",
        "Dra.Paredes": "Cirugia Oral"
    }
    return especialidades.get(medico, "General")

def obtener_medicos():
    especialidades = {
        "Dr.Vega": "Endodoncia",
        "Dr.Perez": "Periodoncia",
        "Dra.Morales": "Ortodoncia",
        "Dr.Castro": "Protesis dental",
        "Dra.Paredes": "Cirugia Oral"
    }
    return list(especialidades.keys())


# ===== Guardar datos en CSV (Función de Backup) =====
def persistir_csv_backup(hoja_gspread, nombre_archivo_csv):
    """
    Descarga TODOS los datos de una Google Sheet y los
    sobrescribe en un archivo CSV local como backup.
    """
    if hoja_gspread is None:
        return
        
    try:
        datos = hoja_gspread.get_all_values()
        
        # Asegurarse de que el directorio 'data' exista
        os.makedirs(os.path.dirname(nombre_archivo_csv), exist_ok=True)
        
        with open(nombre_archivo_csv, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(datos)
        print(f"💾 Backup CSV guardado en {nombre_archivo_csv}")
    except Exception as e:
        print(f"❌ Error al crear backup CSV: {e}")


# ===== Comando Agendar (CORREGIDO para evitar duplicados) =====
def agendar(nombre, dni, telefono, email, fecha, hora, medico):
    """
    Valida datos, busca si el paciente ya existe por DNI,
    lo crea si no existe, y luego agenda la cita en Google Sheets.
    """

    # --- 1. Validaciones (igual que antes) ---
    try:
        dni_limpio = ''.join(filter(str.isdigit, str(dni)))
        if len(dni_limpio) != 8:
            raise ValueError(f"DNI debe tener 8 dígitos (dato: {dni})")
        dni_num = int(dni_limpio)
    except ValueError as e:
        print(f"❌ Error de validación: {e}")
        return f"Error: DNI debe tener 8 dígitos (recibido: {dni})."

    try:
        tel_limpio = ''.join(filter(str.isdigit, str(telefono)))
        if len(tel_limpio) != 9 or not tel_limpio.startswith("9"):
            raise ValueError(f"Teléfono debe tener 9 dígitos y empezar con 9 (dato: {telefono})")
        tel_num = int(tel_limpio)
    except ValueError as e:
        print(f"❌ Error de validación: {e}")
        return f"Error: Teléfono debe tener 9 dígitos y empezar con 9 (recibido: {telefono})."

    # --- 2. Verificar Conexión ---
    if pacientes_sheet is None or citas_sheet is None:
        return "Error: No hay conexión a Google Sheets. Revisa las credenciales."

    try:
        # --- 3. Buscar Paciente por DNI ---
        print(f"Buscando paciente con DNI: {dni_num}...")
        celda_paciente = pacientes_sheet.find(str(dni_num), in_column=3) # Columna 3 es 'DNI'

        if celda_paciente:
            # --- Paciente ENCONTRADO ---
            id_paciente = pacientes_sheet.cell(celda_paciente.row, 1).value # Columna 1 es 'ID_Paciente'
            nombre_existente = pacientes_sheet.cell(celda_paciente.row, 2).value
            print(f"✅ Paciente encontrado: {id_paciente} ({nombre_existente}). Usando ID existente.")
            # (Opcional: Podrías actualizar el teléfono/email si son diferentes)
            # pacientes_sheet.update_cell(celda_paciente.row, 4, tel_num)
            # pacientes_sheet.update_cell(celda_paciente.row, 5, email)

        else:
            # --- Paciente NO Encontrado: Crear uno nuevo ---
            print(f"Paciente con DNI {dni_num} no encontrado. Creando nuevo paciente...")
            id_paciente = generar_id("P", pacientes_sheet)
            fila_paciente = [id_paciente, nombre, dni_num, tel_num, email]
            pacientes_sheet.append_row(fila_paciente, value_input_option="USER_ENTERED")
            print(f"✅ Nuevo Paciente creado en GSheets: {id_paciente}")

        # --- 4. Crear Cita (usando el ID_Paciente encontrado o creado) ---
        id_cita = generar_id("C", citas_sheet)
        especialidad = asignar_especialidad(medico)
        # Estado inicial siempre es "Pendiente" (con mayúscula inicial)
        fila_cita = [id_cita, id_paciente, fecha, hora, medico, especialidad, "Pendiente"]
        citas_sheet.append_row(fila_cita, value_input_option="USER_ENTERED")
        print(f"✅ Cita agendada en GSheets: {id_cita} para paciente {id_paciente}")

        # --- 5. Guardar CSV (Backup) ---
        persistir_csv_backup(pacientes_sheet, PACIENTES_CSV)
        persistir_csv_backup(citas_sheet, CITAS_CSV)

        return f"¡Éxito! Cita {id_cita} agendada para el paciente {id_paciente} en Google Sheets."

    except Exception as e:
        print(f"❌ Error durante el agendamiento: {e}")
        return f"Error al procesar la cita en Google Sheets: {e}"

# ===== Función "Leer" (Read) - (Tarea S2-04) =====
def consultar_citas(dni):
    """
    Busca citas en Google Sheets por DNI del paciente.
    """
    if pacientes_sheet is None or citas_sheet is None:
        return "Error: No hay conexión a Google Sheets."

    try:
        # 1. Buscar el ID del Paciente usando el DNI
        # Columna 3 es 'DNI'
        celda_paciente = pacientes_sheet.find(dni, in_column=3) 
        if not celda_paciente:
            return f"No se encontró ningún paciente con el DNI {dni}."
        
        # Columna 1 es 'ID_Paciente', Columna 2 es 'Nombre'
        id_paciente = pacientes_sheet.cell(celda_paciente.row, 1).value 
        nombre_paciente = pacientes_sheet.cell(celda_paciente.row, 2).value 

        # 2. Buscar todas las citas con ese ID de Paciente
        # Columna 2 es 'ID_Paciente'
        celdas_citas = citas_sheet.findall(id_paciente, in_column=2) 
        
        if not celdas_citas:
            return f"Paciente {nombre_paciente} ({id_paciente}) no tiene citas programadas."

        # 3. Formatear los resultados
        citas_encontradas = []
        encabezados = citas_sheet.row_values(1) # Obtener los títulos ('ID_Cita', 'Fecha', 'Estado')
        
        for celda in celdas_citas:
            datos_cita = citas_sheet.row_values(celda.row)
            # Convertir a un diccionario legible usando los encabezados correctos
            cita_dict = dict(zip(encabezados, datos_cita))
            citas_encontradas.append(cita_dict)
        
        print(f"✅ Citas encontradas para {id_paciente}: {len(citas_encontradas)}")
        return citas_encontradas

    except Exception as e:
        print(f"❌ Error durante la consulta de citas: {e}")
        return f"Error al consultar citas: {e}"

# ===== Función "Actualizar" (Cancel) - (Tarea S2-04) =====
def cancelar_cita(dni, fecha):
    """
    Busca una cita por DNI y fecha, y actualiza su estado a 'Cancelado'.
    """
    if pacientes_sheet is None or citas_sheet is None:
        return "Error: No hay conexión a Google Sheets."

    try:
        # 1. Buscar el ID del Paciente
        celda_paciente = pacientes_sheet.find(dni, in_column=3) # Columna 3 es 'DNI'
        if not celda_paciente:
            return f"No se encontró ningún paciente con el DNI {dni}."
        
        id_paciente = pacientes_sheet.cell(celda_paciente.row, 1).value # Columna 1 es 'ID_Paciente'
        
        # 2. Buscar la cita específica
        celdas_citas = citas_sheet.findall(id_paciente, in_column=2) # Columna 2 es 'ID_Paciente'
        
        fila_a_cancelar = None
        for celda in celdas_citas:
            datos_fila = citas_sheet.row_values(celda.row)
            fecha_cita = datos_fila[2] # Columna 3 es 'Fecha'
            estado_cita = datos_fila[6] # Columna 7 es 'Estado'
            
            # Comparamos la fecha y que esté 'Pendiente'
            # Usamos .lower() para ser flexibles
            if fecha_cita == fecha and estado_cita.lower() == "pendiente":
                fila_a_cancelar = celda.row
                break
        
        if fila_a_cancelar:
            # 3. Actualizar la celda de Estado (Columna 7) a 'cancelado' (minúscula)
            # para coincidir con tu GSheet
            citas_sheet.update_cell(fila_a_cancelar, 7, "cancelado") 
            print(f"✅ Cita en fila {fila_a_cancelar} actualizada a 'cancelado'.")
            return f"Éxito: La cita del {fecha} para el DNI {dni} ha sido cancelada."
        else:
            # Mensaje más claro si no se encuentra o ya está cancelada/confirmada
            return f"No se encontró una cita 'Pendiente' para el DNI {dni} en la fecha {fecha}."

    except Exception as e:
        print(f"❌ Error durante la cancelación de cita: {e}")
        return f"Error al cancelar la cita: {e}"

# ===== Función NUEVA: Buscar Paciente por DNI =====
def buscar_paciente_por_dni(dni):
    """
    Busca un paciente en Google Sheets por DNI.
    Devuelve un diccionario con sus datos si lo encuentra, o None si no.
    """
    if pacientes_sheet is None:
        print("❌ Error: Hoja de pacientes no disponible.")
        return None

    try:
        dni_str = str(dni).strip() # Asegurarse de que sea string
        print(f"Buscando paciente con DNI: {dni_str}...")
        celda_paciente = pacientes_sheet.find(dni_str, in_column=3) # Columna 3 es 'DNI'

        if celda_paciente:
            # Paciente encontrado, devolver sus datos
            id_paciente = pacientes_sheet.cell(celda_paciente.row, 1).value # Col 1: ID_Paciente
            nombre = pacientes_sheet.cell(celda_paciente.row, 2).value      # Col 2: Nombre
            telefono = pacientes_sheet.cell(celda_paciente.row, 4).value    # Col 4: Telefono
            email = pacientes_sheet.cell(celda_paciente.row, 5).value       # Col 5: Email
            print(f"✅ Paciente encontrado: {id_paciente} ({nombre})")
            return {
                "ID_Paciente": id_paciente,
                "Nombre": nombre,
                "DNI": dni_str, # Devolvemos el DNI buscado
                "Telefono": telefono,
                "Email": email
            }
        else:
            print(f"Paciente con DNI {dni_str} no encontrado.")
            return None
    except Exception as e:
        print(f"❌ Error buscando paciente por DNI: {e}")
        return None

# ===== TEST AUTOMÁTICO (CRUD Completo S2-04) - Usa encabezados MAYÚSCULAS =====
if __name__ == "__main__":
    """
    Esto se ejecuta solo si corres el archivo directamente
    (ej. 'python flujo_agendamiento.py')
    """
    print("\n📌 Iniciando test de CRUD Completo (Modo Sprint 2 - Google Sheets)...\n")
    
    # --- 1. Prueba de "Create" (Agendar) ---
    print("--- Probando CREAR Cita ---")
    dni_prueba = "98765432" # DNI para todas las pruebas
    fecha_prueba = "2025-10-30"
    
    mensaje_crear = agendar(
        nombre="Paciente de Prueba CRUD",
        dni=dni_prueba,
        telefono="911222333",
        email="crud@example.com",
        fecha=fecha_prueba,
        hora="14:00",
        medico="Dra.Paredes"
    )
    print(f"\nResultado CREAR: {mensaje_crear}")
    
    
    # --- 2. Prueba de "Read" (Consultar) ---
    print("\n--- Probando LEER Citas ---")
    print(f"Buscando citas para el DNI: {dni_prueba}...")
    
    citas_encontradas = consultar_citas(dni_prueba)
    
    if isinstance(citas_encontradas, list):
        print(f"Resultado LEER: Se encontraron {len(citas_encontradas)} citas.")
        for cita in citas_encontradas:
            # --- Usa encabezados con Mayúscula (ID_Cita, Fecha, Estado) ---
            print(f"  > Cita ID: {cita['ID_Cita']}, Fecha: {cita['Fecha']}, Estado: {cita['Estado']}")
    else:
        print(f"Resultado LEER: {citas_encontradas}")


    # --- 3. Prueba de "Update" (Cancelar) ---
    print("\n--- Probando ACTUALIZAR Cita (Cancelar) ---")
    print(f"Cancelando cita para DNI {dni_prueba} en fecha {fecha_prueba}...")
    
    mensaje_cancelar = cancelar_cita(dni_prueba, fecha_prueba)
    print(f"Resultado ACTUALIZAR: {mensaje_cancelar}")
    
    
    # --- 4. Verificación Final (Leer de nuevo) ---
    print("\n--- Verificando Cancelación (LEER de nuevo) ---")
    citas_verificacion = consultar_citas(dni_prueba)
    
    if isinstance(citas_verificacion, list):
        for cita in citas_verificacion:
            # --- Usa encabezados con Mayúscula ---
            if cita['Fecha'] == fecha_prueba:
                print(f"  > Cita ID: {cita['ID_Cita']}, Fecha: {cita['Fecha']}, ¡Nuevo Estado: {cita['Estado']}!")
    else:
        print(f"Resultado VERIFICACIÓN: {citas_verificacion}")

    print("\n\n✅ Test CRUD completado. Revisa Google Sheets para confirmar los cambios.")
