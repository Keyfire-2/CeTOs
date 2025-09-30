import pandas as pd
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import io
import streamlit as st
from data_cleaner import WhatsAppDataCleaner  # Integración con el nuevo sistema

class AdvancedDataPreprocessor:
    def __init__(self):
        self.whatsapp_patterns = [
            # Patrón para formato internacional
            re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4},?\s?\d{1,2}:\d{2}\s?[ap]m?)\s?[-–]\s?([^:]+):\s?(.+)'),
            # Patrón para formato español
            re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4},?\s?\d{1,2}:\d{2})\s?[-–]\s?([^:]+):\s?(.+)'),
            # Patrón para formato con segundos
            re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4},?\s?\d{1,2}:\d{2}:\d{2}\s?[ap]m?)\s?[-–]\s?([^:]+):\s?(.+)')
        ]
        
        self.media_indicators = [
            '<multimedia omitido>', 'image omitted', 'video omitted', 
            'audio omitted', 'document omitted', 'sticker omitted',
            'media omitted', 'archivo omitido', '📷', '🎥', '🎵', '📄'
        ]
        
        self.system_messages = [
            'creó el grupo', 'añadió', 'eliminó', 'saliste', 'uniste',
            'cambió', 'cambio', 'security code', 'código de seguridad'
        ]
    
    def process_whatsapp_file(self, file, use_advanced_cleaner: bool = True) -> List[Dict[str, Any]]:
        """Procesa archivo de WhatsApp con opción de usar el cleaner avanzado"""
        try:
            # Leer contenido del archivo
            content = file.getvalue().decode('utf-8', errors='ignore')
            
            # Opción de usar el data_cleaner avanzado
            if use_advanced_cleaner:
                cleaner = WhatsAppDataCleaner()
                return cleaner.clean_whatsapp_file(content)
            else:
                # Usar procesamiento legacy mejorado
                return self._process_with_legacy_method(content)
                
        except Exception as e:
            st.error(f"❌ Error procesando archivo: {str(e)}")
            raise Exception(f"Error procesando archivo: {str(e)}")
    
    def _process_with_legacy_method(self, content: str) -> List[Dict[str, Any]]:
        """Procesamiento legacy mejorado para compatibilidad"""
        messages = []
        lines = content.split('\n')
        buffer = None
        
        for line in lines:
            processed_message = self._parse_whatsapp_line_advanced(line)
            
            if processed_message:
                if buffer:
                    messages.append(buffer)
                buffer = processed_message
            elif buffer:
                # Continuación del mensaje anterior
                buffer['message'] += ' ' + line.strip()
                buffer = self._update_message_metrics(buffer)
            else:
                # Mensaje del sistema o no parseable
                system_msg = self._parse_system_message(line)
                if system_msg:
                    messages.append(system_msg)
        
        if buffer:
            messages.append(buffer)
        
        return self._enhance_messages_advanced(messages)
    
    def _parse_whatsapp_line_advanced(self, line: str) -> Optional[Dict[str, Any]]:
        """Parsea una línea de WhatsApp con múltiples patrones"""
        line = line.strip()
        if not line:
            return None
        
        for pattern in self.whatsapp_patterns:
            match = pattern.match(line)
            if match:
                timestamp_str, user, message = match.groups()
                
                # Limpiar y validar
                user = user.strip()
                message = message.strip()
                
                if not user or not message:
                    continue
                
                # Convertir timestamp
                timestamp = self._parse_timestamp_advanced(timestamp_str)
                
                return {
                    'timestamp': timestamp,
                    'user': user,
                    'message': message,
                    'original_line': line,
                    'message_type': self._classify_message_type(message)
                }
        
        return None
    
    def _parse_timestamp_advanced(self, timestamp_str: str) -> datetime:
        """Conversión mejorada de timestamp con más formatos"""
        try:
            # Limpiar y normalizar el string
            timestamp_str = timestamp_str.replace(',', '').strip()
            
            # Lista expandida de formatos
            formats = [
                '%m/%d/%y %I:%M %p',    # 12/25/23 2:30 PM
                '%d/%m/%y %I:%M %p',    # 25/12/23 2:30 PM
                '%m/%d/%Y %I:%M %p',    # 12/25/2023 2:30 PM
                '%d/%m/%Y %I:%M %p',    # 25/12/2023 2:30 PM
                '%Y-%m-%d %I:%M %p',    # 2023-12-25 2:30 PM
                '%m/%d/%y %H:%M',       # 12/25/23 14:30
                '%d/%m/%y %H:%M',       # 25/12/23 14:30
                '%m/%d/%Y %H:%M',       # 12/25/2023 14:30
                '%d/%m/%Y %H:%M',       # 25/12/2023 14:30
                '%Y-%m-%d %H:%M',       # 2023-12-25 14:30
                '%m/%d/%y %I:%M:%S %p', # 12/25/23 2:30:45 PM
                '%d/%m/%y %I:%M:%S %p', # 25/12/23 2:30:45 PM
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            
            # Fallback: intentar parsear con pandas
            try:
                return pd.to_datetime(timestamp_str).to_pydatetime()
            except:
                return datetime.now()
                
        except Exception:
            return datetime.now()
    
    def _classify_message_type(self, message: str) -> str:
        """Clasifica el tipo de mensaje"""
        message_lower = message.lower()
        
        # Verificar si es mensaje de sistema
        if any(indicator in message_lower for indicator in self.system_messages):
            return 'system'
        
        # Verificar si es multimedia
        if any(indicator in message_lower for indicator in self.media_indicators):
            return 'media'
        
        # Verificar si es enlace
        if re.search(r'http[s]?://', message_lower):
            return 'link'
        
        # Verificar si es principalmente emojis
        if self._is_emoji_message(message):
            return 'emoji'
        
        return 'text'
    
    def _is_emoji_message(self, message: str) -> bool:
        """Verifica si el mensaje es principalmente emojis"""
        # Contar caracteres que no son emojis ni espacios
        non_emoji_chars = len([c for c in message if c.isalnum() or c in ',.!?'])
        return len(message.strip()) > 0 and non_emoji_chars == 0
    
    def _parse_system_message(self, line: str) -> Optional[Dict[str, Any]]:
        """Parsea mensajes del sistema"""
        line = line.strip()
        if not line:
            return None
        
        # Buscar patrones de mensajes del sistema
        for pattern in self.whatsapp_patterns:
            match = pattern.match(line)
            if match:
                timestamp_str, _, message = match.groups()
                if any(indicator in message.lower() for indicator in self.system_messages):
                    return {
                        'timestamp': self._parse_timestamp_advanced(timestamp_str),
                        'user': 'Sistema',
                        'message': message.strip(),
                        'original_line': line,
                        'message_type': 'system'
                    }
        
        return None
    
    def _update_message_metrics(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Actualiza métricas cuando un mensaje continúa"""
        message['words_count'] = len(message['message'].split())
        message['characters_count'] = len(message['message'])
        message['message_type'] = self._classify_message_type(message['message'])
        return message
    
    def _enhance_messages_advanced(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mejora los mensajes con análisis avanzado"""
        enhanced_messages = []
        
        for msg in messages:
            enhanced_msg = msg.copy()
            
            # Métricas básicas
            enhanced_msg.update({
                'words_count': len(msg['message'].split()),
                'characters_count': len(msg['message']),
                'has_media': self._check_media_advanced(msg['message']),
                'is_link': 'http' in msg['message'].lower(),
                'is_question': '?' in msg['message'],
                'has_exclamation': '!' in msg['message'],
                'clean_message': self._clean_message_advanced(msg['message']),
                'day_of_week': msg['timestamp'].strftime('%A').lower() if isinstance(msg['timestamp'], datetime) else 'unknown',
                'hour': msg['timestamp'].hour if isinstance(msg['timestamp'], datetime) else 0
            })
            
            # Análisis de sentimiento básico
            enhanced_msg['sentiment'] = self._calculate_basic_sentiment(msg['message'])
            
            # Extraer emojis
            enhanced_msg['emojis'] = self._extract_emojis(msg['message'])
            
            # Extraer menciones
            enhanced_msg['mentions'] = self._extract_mentions(msg['message'])
            
            # Extraer URLs
            enhanced_msg['urls'] = self._extract_urls(msg['message'])
            
            enhanced_messages.append(enhanced_msg)
        
        return enhanced_messages
    
    def _check_media_advanced(self, message: str) -> bool:
        """Verificación mejorada de multimedia"""
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in self.media_indicators)
    
    def _clean_message_advanced(self, message: str) -> str:
        """Limpieza avanzada del mensaje para análisis"""
        # Remover URLs
        cleaned = re.sub(r'http[s]?://\S+', '', message)
        # Remover menciones
        cleaned = re.sub(r'@\w+', '', cleaned)
        # Remover caracteres especiales pero mantener letras acentuadas
        cleaned = re.sub(r'[^\w\sáéíóúñüÁÉÍÓÚÑÜ]', ' ', cleaned)
        # Remover espacios extra
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def _calculate_basic_sentiment(self, message: str) -> float:
        """Cálculo básico de sentimiento"""
        positive_words = ['bueno', 'excelente', 'genial', 'gracias', 'feliz', 'contento', 'amo', 'quiero']
        negative_words = ['malo', 'terrible', 'odio', 'triste', 'enojado', 'problema', 'error']
        
        message_lower = message.lower()
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        total_words = len(message.split())
        
        if total_words == 0:
            return 0.0
        
        score = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, score))
    
    def _extract_emojis(self, message: str) -> List[str]:
        """Extrae emojis del mensaje"""
        # Patrón simple para emojis (en una implementación real usarías una librería como emoji)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.findall(message)
    
    def _extract_mentions(self, message: str) -> List[str]:
        """Extrae menciones del mensaje"""
        return re.findall(r'@(\w+)', message)
    
    def _extract_urls(self, message: str) -> List[str]:
        """Extrae URLs del mensaje"""
        return re.findall(r'http[s]?://\S+', message)
    
    def validate_data_advanced(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validación avanzada de la calidad de los datos"""
        if not messages:
            return {'valid': False, 'error': 'No hay mensajes para analizar'}
        
        try:
            df = pd.DataFrame(messages)
            
            # Métricas básicas
            stats = {
                'total_messages': len(messages),
                'unique_users': df['user'].nunique(),
                'date_range': {
                    'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                    'end': df['timestamp'].max() if 'timestamp' in df.columns else None
                },
                'message_types': df['message_type'].value_counts().to_dict() if 'message_type' in df.columns else {},
                'users_with_few_messages': [],
                'quality_issues': []
            }
            
            # Análisis de usuarios
            user_counts = df['user'].value_counts()
            stats['users_with_few_messages'] = user_counts[user_counts < 3].index.tolist()
            
            # Verificar problemas de calidad
            if len(messages) < 10:
                stats['quality_issues'].append('Pocos mensajes para análisis significativo')
            
            if df['user'].nunique() < 2:
                stats['quality_issues'].append('Solo un usuario en la conversación')
            
            # Verificar distribución temporal
            if 'timestamp' in df.columns:
                time_span = stats['date_range']['end'] - stats['date_range']['start'] if stats['date_range']['start'] and stats['date_range']['end'] else None
                if time_span and time_span.days < 1:
                    stats['quality_issues'].append('Conversación muy corta en el tiempo')
            
            # Calcular score de calidad
            stats['valid'] = len(stats['quality_issues']) == 0
            stats['quality_score'] = self._calculate_advanced_quality_score(stats, df)
            stats['recommendations'] = self._generate_quality_recommendations(stats)
            
            return stats
            
        except Exception as e:
            return {'valid': False, 'error': f'Error en validación: {str(e)}'}
    
    def _calculate_advanced_quality_score(self, stats: Dict[str, Any], df: pd.DataFrame) -> float:
        """Calcula score de calidad avanzado"""
        score = 0.0
        max_score = 10.0
        
        # Puntos por cantidad de mensajes
        if stats['total_messages'] >= 100:
            score += 3.0
        elif stats['total_messages'] >= 50:
            score += 2.0
        elif stats['total_messages'] >= 20:
            score += 1.0
        
        # Puntos por número de usuarios
        if stats['unique_users'] >= 5:
            score += 2.0
        elif stats['unique_users'] >= 3:
            score += 1.0
        
        # Puntos por diversidad de tipos de mensaje
        if len(stats['message_types']) >= 3:
            score += 1.0
        
        # Puntos por span temporal (si hay datos de timestamp)
        if 'timestamp' in df.columns and not df['timestamp'].isna().all():
            time_span = df['timestamp'].max() - df['timestamp'].min()
            if time_span.days >= 7:
                score += 2.0
            elif time_span.days >= 1:
                score += 1.0
        
        # Penalizaciones
        score -= len(stats['users_with_few_messages']) * 0.5
        score -= len(stats['quality_issues']) * 0.5
        
        return max(0.0, min(max_score, score))
    
    def _generate_quality_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones para mejorar la calidad de los datos"""
        recommendations = []
        
        if stats['total_messages'] < 50:
            recommendations.append("📊 Considera analizar conversaciones más largas para mejores insights")
        
        if stats['unique_users'] < 3:
            recommendations.append("👥 La conversación tiene pocos participantes, considera grupos más activos")
        
        if stats['users_with_few_messages']:
            recommendations.append("🎯 Algunos usuarios tienen pocos mensajes, podrían ser participantes ocasionales")
        
        if not stats['quality_issues'] and stats['quality_score'] >= 7:
            recommendations.append("✅ Calidad de datos excelente para análisis")
        
        return recommendations

# Clase de compatibilidad con versión anterior
class DataPreprocessor(AdvancedDataPreprocessor):
    """Clase de compatibilidad con la versión anterior"""
    
    def validate_data(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Método de compatibilidad con la versión anterior"""
        result = self.validate_data_advanced(messages)
        
        # Mantener estructura compatible
        compatible_result = {
            'valid': result['valid'],
            'error': result.get('error', ''),
            'total_messages': result['total_messages'],
            'unique_users': result['unique_users'],
            'date_range': result['date_range'],
            'message_lengths': [len(msg.get('message', '')) for msg in messages],
            'users_with_few_messages': result['users_with_few_messages'],
            'quality_score': result['quality_score']
        }
        
        return compatible_result

# Función de utilidad para uso rápido
def preprocess_whatsapp_file(file, use_advanced: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Función de conveniencia para preprocesamiento rápido"""
    preprocessor = AdvancedDataPreprocessor()
    
    try:
        messages = preprocessor.process_whatsapp_file(file, use_advanced)
        validation = preprocessor.validate_data_advanced(messages)
        
        return messages, validation
        
    except Exception as e:
        st.error(f"❌ Error en preprocesamiento: {e}")
        return [], {'valid': False, 'error': str(e)}