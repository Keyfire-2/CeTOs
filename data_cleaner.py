import re
import json
import pandas as pd
from datetime import datetime
import emoji
from typing import List, Dict, Any, Optional
import streamlit as st
import requests
from textblob import TextBlob
from textblob.sentiments import PatternAnalyzer
import numpy as np

class WhatsAppDataCleaner:
    def __init__(self):
        self.USUARIOS = [
            "Karla", "Lawrence Merry 🥰💖", "Brayan Ramos", "Trixxie 💜",
            "crisleoalvarez9", "Lupita 🦋", "Cesar", "Alejandro Sanz",
            "Alee💕🫶🏼", "Manu", "El Dogk🐾", "Charly", "Leonardo Espinoza",
            "Jack :3", "+52 55 2256 8704", "+52 55 1075 0633", "+52 55 8478 5889",
            "+52 56 4570 1929", "+52 55 5218 3296", "+52 56 2758 8572",
            "+52 56 4324 9103", "+52 55 3197 5514", "+52 55 1116 8705",
            "+52 55 7526 1770", "+52 55 3978 1636", "+52 56 5712 4390",
            "+52 56 1207 7882", "Mapachita", "mapachita"
        ]
        
        self.DAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
        
        # Expresiones regulares (mantenemos las originales)
        self.LINE_RE = re.compile(
            r"""^\s*
            (?P<fecha>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})
            [,\s]*
            (?P<hora>\d{1,2}:\d{2}(?::\d{2})?\s*[apAP\.m\s\u202f]{0,6})
            \s*[-–]\s*
            (?P<usuario>[^:]{1,120}?)\s*:\s*
            (?P<mensaje>.*)$
            """,
            re.UNICODE | re.VERBOSE
        )
        
        self.SYSTEM_RE = re.compile(
            r"""^\s*
            (?P<fecha>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})
            [,\s]*
            (?P<hora>\d{1,2}:\d{2}(?::\d{2})?\s*[apAP\.m\s\u202f]{0,6})
            \s*[-–]\s*
            (?P<mensaje_sistema>.+)$
            """,
            re.UNICODE | re.VERBOSE
        )
        
        self.URL_RE = re.compile(r'(https?://\S+)')
        self.MENTION_RE = re.compile(r'@(\w+)')
        
        # NUEVO: Diccionarios de sentimiento más completos
        self._initialize_sentiment_resources()

    def _initialize_sentiment_resources(self):
        """Inicializa recursos para análisis de sentimiento avanzado"""
        # Diccionario expandido de palabras en español
        self.POSITIVE_WORDS = {
            'bueno', 'excelente', 'genial', 'fantástico', 'maravilloso', 'increíble', 'perfecto',
            'gracias', 'amo', 'quiero', 'feliz', 'contento', 'alegre', 'encantado', 'satisfecho',
            'divertido', 'hermoso', 'precioso', 'bonito', 'agradable', 'estupendo', 'magnífico',
            'bravo', 'chévere', 'padre', 'guay', 'cool', 'bien', 'correcto', 'acertado',
            'amor', 'beso', 'abrazo', 'caricia', 'ternura', 'cariño', 'afecto',
            'éxito', 'triunfo', 'victoria', 'logro', 'avance', 'progreso', 'mejora',
            'risa', 'alegría', 'diversión', 'fiesta', 'celebración', 'felicitación'
        }
        
        self.NEGATIVE_WORDS = {
            'malo', 'terrible', 'horrible', 'fatal', 'pésimo', 'frustrante', 'decepcionante',
            'odio', 'detesto', 'aborrezco', 'triste', 'deprimido', 'desanimado', 'desesperado',
            'enojado', 'molesto', 'furioso', 'airado', 'irritado', 'enfadado', 'cabreado',
            'problema', 'error', 'fallo', 'defecto', 'imperfección', 'dificultad', 'obstáculo',
            'fracaso', 'derrota', 'pérdida', 'daño', 'perjuicio', 'estropeo', 'avería',
            'enfermo', 'dolor', 'sufrimiento', 'molestia', 'incomodidad', 'malestar',
            'miedo', 'temor', 'susto', 'pánico', 'ansiedad', 'angustia', 'preocupación'
        }
        
        # Intensificadores y negaciones
        self.INTENSIFIERS = {
            'muy': 1.5, 'mucho': 1.4, 'bastante': 1.3, 'realmente': 1.6, 
            'extremadamente': 2.0, 'totalmente': 1.7, 'completamente': 1.7,
            'super': 1.8, 'hyper': 2.0, 'super': 1.8, 'más': 1.3, 'tan': 1.4
        }
        
        self.NEGATIONS = {'no', 'ni', 'nunca', 'jamás', 'tampoco', 'nada', 'ningún'}

    # MANTENEMOS TODOS LOS MÉTODOS EXISTENTES HASTA enhance_messages_with_sentiment
    # [Todos los métodos anteriores se mantienen igual hasta enhance_messages_with_sentiment]
    
    def clean_time_string(self, time_str: str) -> str:
        """Limpia y normaliza la cadena de tiempo"""
        # ... (código original igual)
        if not time_str:
            return ""
        
        time_str = (time_str.replace("\u202f", " ")
                    .replace(".", "")
                    .replace(" ", " ")
                    .strip()
                    .lower())
        
        time_str = re.sub(r'\s+', ' ', time_str)
        time_str = re.sub(r'(?P<time>\d{1,2}:\d{2})\s*(?P<ap>[ap])m?', r'\g<time> \g<ap>m', time_str)
        time_str = re.sub(r'(?P<time>\d{1,2}:\d{2})\s*(?P<ap>[ap])\.m\.', r'\g<time> \g<ap>m', time_str)
        
        return time_str.strip()

    def clean_date_string(self, date_str: str) -> str:
        """Limpia y normaliza la cadena de fecha"""
        # ... (código original igual)
        if not date_str:
            return ""
        
        date_str = date_str.replace("-", "/").strip()
        
        if re.match(r'\d{1,2}/\d{1,2}/\d{2}$', date_str):
            parts = date_str.split('/')
            year = int(parts[2])
            if year < 100:
                year += 2000 if year < 50 else 1900
            date_str = f"{parts[0]}/{parts[1]}/{year}"
        
        return date_str

    def normalize_datetime(self, fecha: str, hora: str) -> tuple:
        """Normaliza fecha y hora a formato estándar"""
        # ... (código original igual)
        try:
            fecha_clean = self.clean_date_string(fecha)
            hora_clean = self.clean_time_string(hora)
            
            if not fecha_clean or not hora_clean:
                return f"{fecha} {hora}", "desconocido"
            
            formatos = [
                "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%y %H:%M", "%d/%m/%y %H:%M:%S",
                "%d/%m/%Y %I:%M %p", "%d/%m/%Y %I:%M:%S %p", "%d/%m/%y %I:%M %p", "%d/%m/%y %I:%M:%S %p",
                "%m/%d/%Y %H:%M", "%m/%d/%Y %I:%M %p",
            ]
            
            for fmt in formatos:
                try:
                    dt = datetime.strptime(f"{fecha_clean} {hora_clean}", fmt)
                    fecha_iso = dt.strftime("%Y-%m-%d %H:%M:%S")
                    dia_semana = self.DAYS_ES[dt.weekday()]
                    return fecha_iso, dia_semana
                except ValueError:
                    continue
            
            formatos_fecha = ["%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y"]
            for fmt in formatos_fecha:
                try:
                    dt = datetime.strptime(fecha_clean, fmt)
                    fecha_iso = dt.strftime("%Y-%m-%d 00:00:00")
                    dia_semana = self.DAYS_ES[dt.weekday()]
                    return fecha_iso, dia_semana
                except ValueError:
                    continue
                    
        except Exception as e:
            st.error(f"Error normalizando fecha: {e}")
        
        return f"{fecha} {hora}", "desconocido"

    def normalize_usuario(self, usuario: str) -> str:
        """Normaliza el nombre de usuario"""
        # ... (código original igual)
        if not usuario or usuario == "Unknown":
            return "Sistema"
        
        usuario = usuario.replace("\u202f", " ").strip()
        for member in self.USUARIOS:
            if usuario == member or usuario in member or member in usuario:
                return member
        return usuario

    def extract_emojis(self, texto: str) -> List[str]:
        """Extrae emojis del texto"""
        return [c for c in texto if c in emoji.EMOJI_DATA]

    def extract_urls(self, texto: str) -> List[str]:
        """Extrae URLs del texto"""
        return self.URL_RE.findall(texto)

    def extract_mentions(self, texto: str) -> List[str]:
        """Extrae menciones del texto"""
        return self.MENTION_RE.findall(texto)

    def determine_tipo(self, texto: str) -> str:
        """Determina el tipo de mensaje"""
        # ... (código original igual)
        if not texto:
            return "sistema"
        elif self.extract_urls(texto):
            return "enlace"
        elif self.extract_emojis(texto) and len(texto.strip()) <= 5 and all(c in emoji.EMOJI_DATA or c.isspace() for c in texto.strip()):
            return "emoji"
        elif any(keyword in texto.lower() for keyword in ["creó el grupo", "añadió", "uniste", "saliste", "eliminó"]):
            return "sistema"
        else:
            return "texto"

    def parse_system_message(self, line: str) -> Optional[Dict]:
        """Parsea mensajes del sistema"""
        # ... (código original igual)
        m = self.SYSTEM_RE.match(line)
        if m:
            fecha = m.group("fecha").strip()
            hora = m.group("hora").strip()
            contenido = m.group("mensaje_sistema").strip()
            fecha_hora, dia_semana = self.normalize_datetime(fecha, hora)
            
            return {
                "timestamp": fecha_hora,
                "user": "Sistema",
                "message": contenido,
                "sentiment": 0.0,
                "emotions": {},
                "message_type": "system",
                "words_count": len(contenido.split()),
                "emojis": self.extract_emojis(contenido),
                "urls": self.extract_urls(contenido),
                "mentions": self.extract_mentions(contenido),
                "day_of_week": dia_semana
            }
        return None

    def clean_whatsapp_file(self, file_content: str) -> List[Dict[str, Any]]:
        """Limpia y procesa archivo de WhatsApp"""
        # ... (código original igual)
        lines = file_content.split('\n')
        messages = []
        buffer = None
        
        for i, raw_line in enumerate(lines):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            
            # Intentar parsear como mensaje normal
            m = self.LINE_RE.match(line)
            if m:
                if buffer:
                    messages.append(buffer)
                    buffer = None
                
                fecha = m.group("fecha").strip()
                hora = m.group("hora").strip()
                usuario = self.normalize_usuario(m.group("usuario").strip())
                contenido = m.group("mensaje").strip()
                fecha_hora, dia_semana = self.normalize_datetime(fecha, hora)
                tipo = self.determine_tipo(contenido)
                
                message_obj = {
                    "timestamp": fecha_hora,
                    "user": usuario,
                    "message": contenido,
                    "sentiment": 0.0,  # Se calculará después
                    "emotions": {},    # Se calculará después
                    "message_type": tipo,
                    "words_count": len(contenido.split()),
                    "emojis": self.extract_emojis(contenido),
                    "urls": self.extract_urls(contenido),
                    "mentions": self.extract_mentions(contenido),
                    "day_of_week": dia_semana,
                    "has_media": "omitted" in contenido.lower() or "multimedia" in contenido.lower()
                }
                messages.append(message_obj)
                buffer = message_obj
                
            else:
                # Intentar parsear como mensaje del sistema
                system_msg = self.parse_system_message(line)
                if system_msg:
                    if buffer:
                        messages.append(buffer)
                    messages.append(system_msg)
                    buffer = None
                elif buffer:
                    # Continuación del mensaje anterior
                    buffer["message"] += "\n" + line.strip()
                    # Actualizar métricas
                    buffer["words_count"] = len(buffer["message"].split())
                    buffer["emojis"] = self.extract_emojis(buffer["message"])
                    buffer["urls"] = self.extract_urls(buffer["message"])
                    buffer["mentions"] = self.extract_mentions(buffer["message"])
                    buffer["message_type"] = self.determine_tipo(buffer["message"])
                else:
                    # Mensaje desconocido
                    message_obj = {
                        "timestamp": "",
                        "user": "Sistema",
                        "message": line.strip(),
                        "sentiment": 0.0,
                        "emotions": {},
                        "message_type": "system",
                        "words_count": len(line.split()),
                        "emojis": self.extract_emojis(line),
                        "urls": self.extract_urls(line),
                        "mentions": self.extract_mentions(line),
                        "day_of_week": "desconocido"
                    }
                    messages.append(message_obj)
                    buffer = message_obj
        
        if buffer:
            messages.append(buffer)
        
        return messages

    def load_json_file(self, file_content: str) -> List[Dict[str, Any]]:
        """Carga y procesa archivo JSON"""
        # ... (código original igual)
        try:
            data = json.loads(file_content)
            messages = []
            
            if isinstance(data, dict) and 'mensajes' in data:
                # Formato del limpiador anterior
                for msg in data['mensajes']:
                    cabecera = msg.get('cabecera', {})
                    contenido = msg.get('contenido', '')
                    
                    message_obj = {
                        "timestamp": cabecera.get('fecha_hora', ''),
                        "user": self.normalize_usuario(cabecera.get('usuario', '')),
                        "message": contenido,
                        "sentiment": 0.0,
                        "emotions": {},
                        "message_type": cabecera.get('tipo', 'texto'),
                        "words_count": len(contenido.split()),
                        "emojis": msg.get('extras', {}).get('emojis', []),
                        "urls": msg.get('extras', {}).get('urls', []),
                        "mentions": msg.get('extras', {}).get('menciones', []),
                        "day_of_week": cabecera.get('dia_semana', 'desconocido')
                    }
                    messages.append(message_obj)
            
            elif isinstance(data, list):
                # Formato array simple
                for msg in data:
                    if isinstance(msg, dict):
                        message_obj = {
                            "timestamp": msg.get('timestamp', ''),
                            "user": self.normalize_usuario(msg.get('user', '')),
                            "message": msg.get('message', ''),
                            "sentiment": msg.get('sentiment', 0.0),
                            "emotions": msg.get('emotions', {}),
                            "message_type": msg.get('message_type', 'texto'),
                            "words_count": msg.get('words_count', 0),
                            "emojis": msg.get('emojis', []),
                            "urls": msg.get('urls', []),
                            "mentions": msg.get('mentions', []),
                            "day_of_week": msg.get('day_of_week', 'desconocido')
                        }
                        messages.append(message_obj)
            
            return messages
            
        except json.JSONDecodeError as e:
            st.error(f"Error decodificando JSON: {e}")
            return []

    def process_uploaded_file(self, uploaded_file) -> List[Dict[str, Any]]:
        """Procesa archivo subido (TXT o JSON)"""
        # ... (código original igual)
        file_content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        
        if uploaded_file.name.endswith('.json'):
            return self.load_json_file(file_content)
        else:
            return self.clean_whatsapp_file(file_content)

    def get_cleaning_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Genera estadísticas del proceso de limpieza"""
        # ... (código original igual)
        if not messages:
            return {}
        
        # Crear DataFrame
        df = pd.DataFrame(messages)
        
        # Filtrar timestamps válidos para el cálculo del rango de fechas
        valid_timestamps = []
        for ts in df['timestamp']:
            if ts and isinstance(ts, str) and len(ts) > 10:  # Timestamp válido
                try:
                    # Intentar convertir a datetime
                    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    valid_timestamps.append(dt)
                except (ValueError, TypeError):
                    continue
        
        # Calcular rango de fechas solo si hay timestamps válidos
        date_range = {"start": "", "end": ""}
        if valid_timestamps:
            date_range = {
                "start": min(valid_timestamps).strftime("%Y-%m-%d %H:%M:%S"),
                "end": max(valid_timestamps).strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Calcular días de diferencia si hay al menos 2 fechas válidas
        days_diff = 0
        if len(valid_timestamps) >= 2:
            time_span = max(valid_timestamps) - min(valid_timestamps)
            days_diff = time_span.days
        
        stats = {
            'total_messages': len(messages),
            'unique_users': df['user'].nunique(),
            'date_range': date_range,
            'conversation_days': days_diff,
            'message_types': df['message_type'].value_counts().to_dict(),
            'users_message_count': df['user'].value_counts().to_dict(),
            'days_distribution': df['day_of_week'].value_counts().to_dict(),
            'total_words': df['words_count'].sum() if 'words_count' in df.columns else 0,
            'total_emojis': sum(len(msg.get('emojis', [])) for msg in messages),
            'total_urls': sum(len(msg.get('urls', [])) for msg in messages),
            'valid_timestamps': len(valid_timestamps),
            'messages_per_day': len(messages) / max(days_diff, 1) if days_diff > 0 else len(messages)
        }
        
        return stats

    # 🚀 NUEVOS MÉTODOS MEJORADOS PARA SENTIMIENTO

    def enhance_messages_with_sentiment(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mejora los mensajes con análisis de sentimiento AVANZADO"""
        enhanced_messages = []
        total_messages = len(messages)
        
        # Barra de progreso para análisis de sentimiento
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, msg in enumerate(messages):
            enhanced_msg = msg.copy()
            message_text = msg.get('message', '')
            
            # Actualizar progreso
            progress = (i + 1) / total_messages
            progress_bar.progress(progress)
            status_text.text(f"Analizando sentimiento... {i+1}/{total_messages}")
            
            # Análisis de sentimiento avanzado
            sentiment_score = self._calculate_advanced_sentiment(message_text)
            
            # Análisis de emociones mejorado
            emotions = self._detect_advanced_emotions(message_text, sentiment_score)
            
            # Análisis de emojis para sentimiento
            emoji_sentiment = self._analyze_emoji_sentiment(msg.get('emojis', []))
            
            # Combinar sentimientos (texto + emojis)
            final_sentiment = self._combine_sentiment_sources(sentiment_score, emoji_sentiment)
            
            enhanced_msg.update({
                'sentiment': final_sentiment,
                'emotions': emotions,
                'sentiment_components': {
                    'text_sentiment': sentiment_score,
                    'emoji_sentiment': emoji_sentiment,
                    'final_sentiment': final_sentiment
                }
            })
            
            enhanced_messages.append(enhanced_msg)
        
        # Limpiar barra de progreso
        progress_bar.empty()
        status_text.empty()
        
        return enhanced_messages

    def _calculate_advanced_sentiment(self, text: str) -> float:
        """Calcula sentimiento usando múltiples métodos y los combina"""
        if not text or len(text.strip()) < 2:
            return 0.0
        
        # Método 1: TextBlob (inglés/español)
        blob_score = self._textblob_sentiment(text)
        
        # Método 2: Diccionario de palabras (español)
        lexicon_score = self._lexicon_based_sentiment(text)
        
        # Método 3: Análisis de patrones (estructura de frase)
        pattern_score = self._pattern_based_sentiment(text)
        
        # Combinar scores ponderados
        final_score = (blob_score * 0.4 + lexicon_score * 0.4 + pattern_score * 0.2)
        
        return max(-1.0, min(1.0, final_score))

    def _textblob_sentiment(self, text: str) -> float:
        """Usa TextBlob para análisis de sentimiento multilingüe"""
        try:
            blob = TextBlob(text)
            # TextBlob detecta automáticamente el idioma
            return blob.sentiment.polarity
        except:
            return 0.0

    def _lexicon_based_sentiment(self, text: str) -> float:
        """Análisis basado en diccionario expandido en español"""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        
        positive_score = 0
        negative_score = 0
        total_meaningful_words = 0
        
        # Analizar cada palabra considerando contexto
        for i, word in enumerate(words):
            word_score = 0
            intensity = 1.0
            
            # Verificar si hay intensificador anterior
            if i > 0 and words[i-1] in self.INTENSIFIERS:
                intensity = self.INTENSIFIERS[words[i-1]]
            
            # Verificar si hay negación
            negation = any(neg in words[max(0, i-3):i] for neg in self.NEGATIONS)
            
            # Calcular score base de la palabra
            if word in self.POSITIVE_WORDS:
                word_score = 0.7 * intensity
                if negation:
                    word_score = -word_score * 0.5  # La negación invierte el sentimiento
            elif word in self.NEGATIVE_WORDS:
                word_score = -0.7 * intensity
                if negation:
                    word_score = -word_score * 0.5  # Doble negación
            
            # Aplicar score
            if word_score > 0:
                positive_score += word_score
                total_meaningful_words += 1
            elif word_score < 0:
                negative_score += word_score
                total_meaningful_words += 1
        
        # Calcular score final
        if total_meaningful_words == 0:
            return 0.0
        
        total_score = (positive_score + negative_score) / total_meaningful_words
        return max(-1.0, min(1.0, total_score))

    def _pattern_based_sentiment(self, text: str) -> float:
        """Analiza patrones de texto como exclamaciones, preguntas, etc."""
        score = 0.0
        
        # Exclamaciones positivas
        positive_exclamations = re.findall(r'!\s*[👍🎉🥳😊❤️]', text)
        score += len(positive_exclamations) * 0.2
        
        # Exclamaciones negativas
        negative_exclamations = re.findall(r'!\s*[👎😠💔😢]', text)
        score -= len(negative_exclamations) * 0.2
        
        # Preguntas (ligeramente negativas en contexto)
        questions = text.count('?')
        score -= questions * 0.05
        
        # Mayúsculas (énfasis emocional)
        uppercase_words = re.findall(r'\b[A-ZÁÉÍÓÚÑ]{3,}\b', text)
        if uppercase_words:
            # Si hay muchas mayúsculas, puede indicar fuerte emoción
            score += (len(uppercase_words) * 0.1) if len(uppercase_words) < 5 else 0.3
        
        # Palabras repetidas (énfasis emocional)
        repeated_words = re.findall(r'\b(\w+)\s+\1\b', text.lower())
        score += len(repeated_words) * 0.15
        
        return max(-0.5, min(0.5, score))

    def _analyze_emoji_sentiment(self, emojis: List[str]) -> float:
        """Analiza el sentimiento de los emojis"""
        if not emojis:
            return 0.0
        
        emoji_sentiment = 0.0
        emoji_weights = {
            # Positivos
            '😊': 0.8, '😂': 0.9, '🥰': 1.0, '❤️': 0.9, '👍': 0.7, '🎉': 0.8,
            '😍': 1.0, '✨': 0.6, '🥳': 0.9, '😎': 0.7, '💖': 0.9, '🙌': 0.8,
            # Negativos  
            '😢': -0.8, '😠': -0.9, '💔': -1.0, '👎': -0.7, '😞': -0.8, '😡': -1.0,
            '😰': -0.7, '😨': -0.8, '😓': -0.6, '💀': -0.9, '👻': -0.3,
            # Neutrales
            '😐': 0.0, '🤔': 0.0, '🙄': -0.2, '💭': 0.0
        }
        
        for emoji_char in emojis:
            emoji_sentiment += emoji_weights.get(emoji_char, 0.0)
        
        # Normalizar por cantidad de emojis
        return max(-1.0, min(1.0, emoji_sentiment / len(emojis)))

    def _combine_sentiment_sources(self, text_sentiment: float, emoji_sentiment: float) -> float:
        """Combina sentimiento de texto y emojis de forma inteligente"""
        # Los emojis tienen más peso cuando el texto es ambiguo
        if abs(text_sentiment) < 0.2:
            # Texto neutral, los emojis deciden
            return emoji_sentiment * 0.7 + text_sentiment * 0.3
        else:
            # Texto tiene sentimiento claro, combinar balanceadamente
            return (text_sentiment * 0.6 + emoji_sentiment * 0.4)

    def _detect_advanced_emotions(self, text: str, sentiment: float) -> Dict[str, float]:
        """Detección avanzada de emociones considerando contexto"""
        emotions = {
            'joy': 0.0, 'anger': 0.0, 'sadness': 0.0, 
            'surprise': 0.0, 'fear': 0.0, 'love': 0.0, 'neutral': 0.0
        }
        
        if not text:
            emotions['neutral'] = 1.0
            return emotions
        
        text_lower = text.lower()
        
        # Palabras clave expandidas y contextuales
        joy_patterns = [
            r'\b(jaja|jeje|jiji|juju)\b', r'\b(risa|reír|divertido|gracioso)\b',
            r'\b(feliz|alegre|contento|emocionado)\b', r'\b(celebrar|fiesta|éxito)\b'
        ]
        
        anger_patterns = [
            r'\b(enojo|molesto|furioso|cabreado|irritado)\b', 
            r'\b(odio|detesto|aborrezco|asco)\b', r'!\s*[😠🤬]'
        ]
        
        # ... (patrones expandidos para otras emociones)
        
        # Detección basada en patrones
        emotions['joy'] = self._detect_emotion_by_patterns(text_lower, joy_patterns)
        emotions['anger'] = self._detect_emotion_by_patterns(text_lower, anger_patterns)
        
        # Ajustar basado en sentimiento general
        if sentiment > 0.3:
            emotions['joy'] = max(emotions['joy'], 0.6)
        elif sentiment < -0.3:
            emotions['sadness'] = max(emotions['sadness'], 0.6)
        
        # Normalizar emociones
        total = sum(emotions.values())
        if total > 0:
            for emotion in emotions:
                emotions[emotion] /= total
        
        return emotions

    def _detect_emotion_by_patterns(self, text: str, patterns: List[str]) -> float:
        """Detecta emociones basado en patrones regex"""
        score = 0.0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            score += len(matches) * 0.2
        
        return min(1.0, score)

    # Mantenemos el método básico como fallback
    def _calculate_basic_sentiment(self, text: str) -> float:
        """Método básico mantenido para compatibilidad"""
        return self._lexicon_based_sentiment(text)

    def _detect_basic_emotions(self, text: str) -> Dict[str, float]:
        """Método básico mantenido para compatibilidad"""
        return self._detect_advanced_emotions(text, 0.0)