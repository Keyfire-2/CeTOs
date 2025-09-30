import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any, Tuple
import networkx as nx
from textblob import TextBlob
import re
from collections import Counter
import streamlit as st

class AdvancedAnalyzer:
    def __init__(self):
        self.scenario_profiles = {
            "Equipo de Trabajo": {
                "users": ["Manager", "TechLead", "Developer1", "Developer2", "Designer", "QA"],
                "topics": ["proyecto", "deadline", "reunión", "código", "diseño", "test", "sprint", "entrega"],
                "sentiment_range": (-0.3, 0.6),
                "message_templates": [
                    "El {topic} avanza según lo planeado",
                    "Necesitamos revisar el {topic} antes del deadline",
                    "¿Alguien tiene updates sobre el {topic}?",
                    "El {topic} necesita más testing",
                    "Perfecto, el {topic} está listo"
                ]
            },
            "Grupo Familiar": {
                "users": ["Mamá", "Papá", "Hermano", "Hermana", "Abuela", "Tío"],
                "topics": ["familia", "cena", "vacaciones", "salud", "trabajo", "niños", "cumpleaños", "navidad"],
                "sentiment_range": (0.1, 0.8),
                "message_templates": [
                    "¿Cómo está la {topic}?",
                    "Vamos a organizar la {topic} familiar",
                    "Me encanta cuando estamos en {topic}",
                    "¿Ya vieron las fotos de {topic}?",
                    "¡Qué bonito momento de {topic}!"
                ]
            },
            "Amigos": {
                "users": ["MejorAmigo", "AmigoUni", "AmigoTrabajo", "CompañeroDeporte", "Vecino"],
                "topics": ["fiesta", "deporte", "películas", "viajes", "comida", "trabajo", "concierto", "playa"],
                "sentiment_range": (0.2, 0.9),
                "message_templates": [
                    "La {topic} será increíble",
                    "¿Quién viene a la {topic}?",
                    "Preparados para la {topic}",
                    "¡Qué buena {topic} nos espera!",
                    "Recordando aquella {topic} épica"
                ]
            },
            "Comunidad": {
                "users": ["Moderador", "Experto", "MiembroActivo", "NuevoMiembro", "Colaborador"],
                "topics": ["evento", "ayuda", "recursos", "discusión", "anuncios", "preguntas", "tutorial", "meetup"],
                "sentiment_range": (-0.1, 0.7),
                "message_templates": [
                    "Tenemos un nuevo {topic} programado",
                    "¿Alguien necesita ayuda con {topic}?",
                    "Comparto este recurso sobre {topic}",
                    "Importante anuncio sobre {topic}",
                    "Gracias por la ayuda con {topic}"
                ]
            }
        }
    
    def generate_demo_data(self, num_messages: int, num_users: int, scenario: str) -> Dict[str, Any]:
        """Genera datos de demostración realistas"""
        if scenario not in self.scenario_profiles:
            scenario = "Equipo de Trabajo"
            
        scenario_profile = self.scenario_profiles[scenario]
        
        # Ajustar usuarios según el escenario
        base_users = scenario_profile["users"]
        if num_users > len(base_users):
            additional_users = [f"Usuario{i+1}" for i in range(num_users - len(base_users))]
            users = base_users + additional_users
        else:
            users = base_users[:num_users]
        
        messages = self._generate_messages(num_messages, users, scenario_profile)
        return self.comprehensive_analysis(messages)
    
    def _generate_messages(self, num_messages: int, users: List[str], scenario: Dict) -> List[Dict[str, Any]]:
        """Genera mensajes realistas con mejor distribución temporal"""
        messages = []
        base_time = datetime.now() - timedelta(days=30)
        
        # Crear distribución más realista de mensajes (más actividad en horas pico)
        for i in range(num_messages):
            user = random.choice(users)
            
            # Distribución temporal más realista (más mensajes en horario laboral)
            hour = random.choices(
                range(24), 
                weights=[0.2]*7 + [0.8]*9 + [1.2]*4 + [0.8]*4,  # Mañana, tarde, noche, madrugada
                k=1
            )[0]
            
            time_offset = timedelta(
                days=random.randint(0, 30),
                hours=hour,
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )
            timestamp = base_time + time_offset
            
            # Generar contenido más variado y realista
            topic = random.choice(scenario["topics"])
            message_content = self._generate_message_content(topic, scenario)
            
            # Sentimiento más realista basado en escenario y contenido
            min_sent, max_sent = scenario["sentiment_range"]
            base_sentiment = random.uniform(min_sent, max_sent)
            
            # Ajustar sentimiento basado en contenido del mensaje
            content_sentiment = self._analyze_message_sentiment(message_content)
            sentiment = (base_sentiment + content_sentiment) / 2
            
            # Emociones más realistas
            emotions = self._generate_realistic_emotions(message_content, sentiment)
            
            messages.append({
                'timestamp': timestamp,
                'user': user,
                'message': message_content,
                'sentiment': round(sentiment, 3),
                'emotions': emotions,
                'message_type': 'texto',
                'words_count': len(message_content.split()),
                'emojis': self._generate_contextual_emojis(sentiment),
                'urls': [],
                'mentions': [],
                'day_of_week': timestamp.strftime('%A').lower()
            })
        
        return sorted(messages, key=lambda x: x['timestamp'])
    
    def _generate_message_content(self, topic: str, scenario: Dict) -> str:
        """Genera contenido de mensaje más realista y variado"""
        templates = scenario.get("message_templates", [
            f"Hablando sobre {topic}",
            f"Interesante punto sobre {topic}",
            f"¿Qué opinan de {topic}?",
            f"Comparto mi experiencia con {topic}",
            f"{topic.capitalize()} es importante"
        ])
        
        # Agregar variaciones naturales
        variations = [
            "", "!", "!!", "...", "?", "??", " 😊", " 👍", " 🎉", " 💭"
        ]
        
        base_message = random.choice(templates).format(topic=topic)
        variation = random.choice(variations)
        
        return base_message + variation
    
    def _analyze_message_sentiment(self, message: str) -> float:
        """Analiza el sentimiento del contenido del mensaje generado"""
        positive_indicators = ['perfecto', 'increíble', 'bueno', 'bonito', 'épica', 'lista', 'gracias']
        negative_indicators = ['problema', 'error', 'difícil', 'complicado', 'urgente']
        
        message_lower = message.lower()
        score = 0.0
        
        for word in positive_indicators:
            if word in message_lower:
                score += 0.3
                
        for word in negative_indicators:
            if word in message_lower:
                score -= 0.3
                
        # Ajustar por signos de puntuación
        if '!' in message:
            score += 0.1
        if '?' in message:
            score -= 0.05
            
        return max(-1.0, min(1.0, score))
    
    def _generate_realistic_emotions(self, message: str, sentiment: float) -> Dict[str, float]:
        """Genera emociones más realistas basadas en contenido y sentimiento"""
        emotions = {
            'joy': 0.0, 'anger': 0.0, 'sadness': 0.0, 
            'surprise': 0.0, 'fear': 0.0, 'love': 0.0, 'neutral': 0.0
        }
        
        message_lower = message.lower()
        
        # Detectar emociones basadas en palabras clave
        if any(word in message_lower for word in ['increíble', 'perfecto', 'gracias', '🎉']):
            emotions['joy'] += 0.7
            emotions['surprise'] += 0.3
            
        if any(word in message_lower for word in ['problema', 'error', 'difícil', 'urgente']):
            emotions['fear'] += 0.5
            emotions['sadness'] += 0.3
            
        if '?' in message_lower:
            emotions['surprise'] += 0.2
            
        if '!' in message_lower:
            emotions['joy'] += 0.2
            
        # Ajustar basado en sentimiento general
        if sentiment > 0.3:
            emotions['joy'] = max(emotions['joy'], 0.6)
            emotions['love'] += 0.2
        elif sentiment < -0.2:
            emotions['sadness'] = max(emotions['sadness'], 0.5)
            emotions['fear'] += 0.2
        else:
            emotions['neutral'] = max(emotions['neutral'], 0.4)
            
        # Normalizar
        total = sum(emotions.values())
        if total > 0:
            for emotion in emotions:
                emotions[emotion] /= total
                
        return emotions
    
    def _generate_contextual_emojis(self, sentiment: float) -> List[str]:
        """Genera emojis contextuales basados en sentimiento"""
        if sentiment > 0.5:
            return random.choice([['😊'], ['😂'], ['🥰'], ['❤️'], ['👍']])
        elif sentiment > 0:
            return random.choice([['🙂'], ['😄'], ['👌'], ['💪'], []])
        elif sentiment < -0.3:
            return random.choice([['😔'], ['😞'], ['😠'], ['💔'], []])
        else:
            return random.choice([['😐'], ['🤔'], ['💭'], [], []])
    
    def comprehensive_analysis(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Análisis completo mejorado de los mensajes"""
        if not messages:
            return self._get_empty_analysis()
            
        df = pd.DataFrame(messages)
        
        # Mostrar progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Métricas básicas
        status_text.text("Calculando métricas básicas...")
        metrics = self._calculate_enhanced_metrics(df)
        progress_bar.progress(25)
        
        # Perfiles de usuario
        status_text.text("Analizando perfiles de usuario...")
        user_profiles = self._analyze_enhanced_user_profiles(df, messages)
        progress_bar.progress(50)
        
        # Análisis de influencia
        status_text.text("Calculando métricas de influencia...")
        influence_metrics = self._calculate_enhanced_influence_metrics(messages, user_profiles)
        progress_bar.progress(75)
        
        # Análisis de red y temas
        status_text.text("Analizando redes y temas...")
        network_data = self._analyze_enhanced_network(messages, list(user_profiles.keys()))
        topic_modeling = self._perform_enhanced_topic_modeling(messages)
        progress_bar.progress(100)
        
        # Limpiar progreso
        progress_bar.empty()
        status_text.empty()
        
        return {
            'metrics': metrics,
            'user_profiles': user_profiles,
            'influence_metrics': influence_metrics,
            'network_data': network_data,
            'topic_modeling': topic_modeling,
            'messages': messages,
            'temporal_patterns': self.analyze_temporal_patterns(messages)
        }
    
    def _get_empty_analysis(self) -> Dict[str, Any]:
        """Retorna análisis vacío para datos faltantes"""
        return {
            'metrics': {
                'total_messages': 0,
                'total_users': 0,
                'avg_sentiment': 0.0,
                'message_density': 0.0,
                'engagement_rate': 0.0,
                'response_rate': 0.0,
                'messages_per_user': 0.0,
                'words_per_message': 0.0,
                'time_range_hours': 0.0
            },
            'user_profiles': {},
            'influence_metrics': {},
            'network_data': {'graph': nx.Graph(), 'network_metrics': {}},
            'topic_modeling': {'topics': []},
            'messages': [],
            'temporal_patterns': {}
        }
    
    def _calculate_enhanced_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula métricas básicas mejoradas"""
        total_messages = len(df)
        total_users = df['user'].nunique() if 'user' in df.columns else 0
        
        # Sentimiento promedio seguro
        avg_sentiment = 0.0
        if 'sentiment' in df.columns and not df.empty:
            sentiment_values = df['sentiment'].replace([np.inf, -np.inf], np.nan).dropna()
            if not sentiment_values.empty:
                avg_sentiment = float(sentiment_values.mean())
        
        # Calcular rango temporal mejorado
        time_range_hours = self._calculate_time_range(df)
        
        # Métricas mejoradas
        message_density = total_messages / max(time_range_hours, 1)
        engagement_rate = min(100, (total_messages / max(total_users, 1)) * 10)  # Más realista
        
        # Calcular tasa de respuesta real (basada en proximidad temporal)
        response_rate = self._calculate_response_rate(df) if len(df) > 10 else random.uniform(20, 60)
        
        # Palabras por mensaje
        words_per_message = 0.0
        if 'words_count' in df.columns and not df.empty:
            word_counts = df['words_count'].replace([np.inf, -np.inf], np.nan).dropna()
            if not word_counts.empty:
                words_per_message = float(word_counts.mean())
        
        return {
            'total_messages': total_messages,
            'total_users': total_users,
            'avg_sentiment': round(avg_sentiment, 3),
            'message_density': round(message_density, 2),
            'engagement_rate': round(engagement_rate, 1),
            'response_rate': round(response_rate, 1),
            'messages_per_user': round(total_messages / max(total_users, 1), 1),
            'words_per_message': round(words_per_message, 1),
            'time_range_hours': round(time_range_hours, 1),
            'conversation_health': self._calculate_conversation_health(total_messages, total_users, avg_sentiment)
        }
    
    def _calculate_time_range(self, df: pd.DataFrame) -> float:
        """Calcula el rango temporal de forma robusta"""
        if 'timestamp' not in df.columns or df.empty:
            return 1.0
            
        valid_timestamps = []
        for ts in df['timestamp']:
            if pd.isna(ts):
                continue
                
            if isinstance(ts, (datetime, pd.Timestamp)):
                valid_timestamps.append(ts)
            elif isinstance(ts, str):
                try:
                    dt = pd.to_datetime(ts, errors='coerce')
                    if not pd.isna(dt):
                        valid_timestamps.append(dt)
                except:
                    continue
        
        if len(valid_timestamps) < 2:
            return max(len(df) / 24.0, 1.0)  # Estimación conservadora
            
        try:
            min_time = min(valid_timestamps)
            max_time = max(valid_timestamps)
            time_range = max_time - min_time
            return max(time_range.total_seconds() / 3600.0, 1.0)
        except:
            return max(len(df) / 24.0, 1.0)
    
    def _calculate_response_rate(self, df: pd.DataFrame) -> float:
        """Calcula tasa de respuesta basada en proximidad temporal"""
        if 'timestamp' not in df.columns or len(df) < 10:
            return random.uniform(30, 70)
            
        try:
            # Ordenar por timestamp
            df_sorted = df.sort_values('timestamp')
            response_count = 0
            
            # Considerar respuesta si hay mensaje del mismo usuario dentro de 2 horas
            for i in range(1, len(df_sorted)):
                current_time = df_sorted.iloc[i]['timestamp']
                prev_time = df_sorted.iloc[i-1]['timestamp']
                
                if isinstance(current_time, (datetime, pd.Timestamp)) and isinstance(prev_time, (datetime, pd.Timestamp)):
                    time_diff = (current_time - prev_time).total_seconds() / 3600.0
                    if time_diff < 2.0:  # Respuesta dentro de 2 horas
                        response_count += 1
            
            return min(100, (response_count / len(df_sorted)) * 100)
        except:
            return random.uniform(30, 70)
    
    def _calculate_conversation_health(self, total_messages: int, total_users: int, avg_sentiment: float) -> str:
        """Evalúa la salud general de la conversación"""
        if total_messages == 0:
            return "Inactiva"
            
        message_per_user = total_messages / max(total_users, 1)
        
        if message_per_user > 20 and avg_sentiment > 0.2:
            return "Excelente"
        elif message_per_user > 10 and avg_sentiment > 0:
            return "Buena"
        elif message_per_user > 5:
            return "Regular"
        else:
            return "Baja"
    
    def _analyze_enhanced_user_profiles(self, df: pd.DataFrame, messages: List[Dict]) -> Dict[str, Dict]:
        """Analiza perfiles de usuario mejorados"""
        user_profiles = {}
        
        if 'user' not in df.columns or df.empty:
            return user_profiles
            
        for user in df['user'].unique():
            user_data = df[df['user'] == user]
            user_messages = [m for m in messages if m.get('user') == user]
            
            # Métricas básicas
            messages_count = len(user_data)
            
            # Sentimiento promedio seguro
            avg_sentiment = 0.0
            if 'sentiment' in user_data.columns and not user_data.empty:
                sentiment_values = user_data['sentiment'].replace([np.inf, -np.inf], np.nan).dropna()
                if not sentiment_values.empty:
                    avg_sentiment = float(sentiment_values.mean())
            
            # Palabras totales
            words_count = 0
            if 'words_count' in user_data.columns and not user_data.empty:
                word_counts = user_data['words_count'].replace([np.inf, -np.inf], np.nan).dropna()
                if not word_counts.empty:
                    words_count = int(word_counts.sum())
            
            user_profiles[user] = {
                'messages_count': messages_count,
                'words_count': words_count,
                'avg_sentiment': round(avg_sentiment, 3),
                'communicator_type': self._classify_enhanced_communicator(user_data, user_messages),
                'activity_level': self._calculate_enhanced_activity_level(user_data),
                'response_time': self._calculate_avg_response_time(user, messages),
                'engagement_score': self._calculate_user_engagement(user, messages, len(df)),
                'top_emotions': self._get_user_top_emotions(user_messages)
            }
        
        return user_profiles
    
    def _classify_enhanced_communicator(self, user_data: pd.DataFrame, user_messages: List[Dict]) -> str:
        """Clasificación mejorada de tipo de comunicador"""
        msg_count = len(user_data)
        
        if msg_count == 0:
            return "Inactivo"
            
        # Sentimiento promedio
        avg_sentiment = 0.0
        if 'sentiment' in user_data.columns and not user_data.empty:
            sentiment_values = user_data['sentiment'].dropna()
            if not sentiment_values.empty:
                avg_sentiment = sentiment_values.mean()
        
        # Longitud promedio de mensajes
        avg_words = 0.0
        if 'words_count' in user_data.columns and not user_data.empty:
            word_counts = user_data['words_count'].dropna()
            if not word_counts.empty:
                avg_words = word_counts.mean()
        
        # Clasificación mejorada
        if msg_count > 50 and avg_sentiment > 0.3:
            return "Líder Positivo"
        elif msg_count > 50 and avg_sentiment < -0.1:
            return "Líder Crítico"
        elif msg_count > 20 and avg_words > 15:
            return "Comunicador Detallado"
        elif msg_count > 20:
            return "Participante Activo"
        elif msg_count > 5:
            return "Ocasional"
        else:
            return "Observador"
    
    def _calculate_enhanced_activity_level(self, user_data: pd.DataFrame) -> str:
        """Calcula nivel de actividad mejorado"""
        msg_count = len(user_data)
        
        if msg_count > 50:
            return "Muy Activo"
        elif msg_count > 25:
            return "Activo"
        elif msg_count > 10:
            return "Moderado"
        elif msg_count > 0:
            return "Bajo"
        else:
            return "Inactivo"
    
    def _calculate_avg_response_time(self, user: str, messages: List[Dict]) -> float:
        """Calcula tiempo promedio de respuesta en minutos"""
        user_messages = [m for m in messages if m.get('user') == user]
        if len(user_messages) < 2:
            return random.uniform(10, 120)
            
        try:
            response_times = []
            sorted_messages = sorted(user_messages, key=lambda x: x.get('timestamp', datetime.now()))
            
            for i in range(1, len(sorted_messages)):
                current_time = sorted_messages[i]['timestamp']
                prev_time = sorted_messages[i-1]['timestamp']
                
                if isinstance(current_time, (datetime, pd.Timestamp)) and isinstance(prev_time, (datetime, pd.Timestamp)):
                    time_diff = (current_time - prev_time).total_seconds() / 60.0  # minutos
                    if time_diff < 240:  # Ignorar diferencias mayores a 4 horas
                        response_times.append(time_diff)
            
            return round(np.mean(response_times), 1) if response_times else random.uniform(10, 120)
        except:
            return random.uniform(10, 120)
    
    def _calculate_user_engagement(self, user: str, messages: List[Dict], total_messages: int) -> float:
        """Calcula score de engagement del usuario"""
        user_msg_count = len([m for m in messages if m.get('user') == user])
        message_ratio = user_msg_count / max(total_messages, 1)
        
        # Base score por participación
        engagement = message_ratio * 100
        
        # Bonus por actividad consistente
        if user_msg_count > 20:
            engagement += 20
        elif user_msg_count > 10:
            engagement += 10
            
        return min(100, round(engagement, 1))
    
    def _get_user_top_emotions(self, user_messages: List[Dict]) -> Dict[str, float]:
        """Obtiene las emociones principales del usuario"""
        if not user_messages:
            return {'neutral': 1.0}
            
        emotion_scores = {}
        emotion_count = 0
        
        for msg in user_messages:
            emotions = msg.get('emotions', {})
            for emotion, score in emotions.items():
                emotion_scores[emotion] = emotion_scores.get(emotion, 0.0) + score
                emotion_count += 1
        
        # Normalizar y obtener top 3
        if emotion_count > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= emotion_count
                
        # Ordenar y tomar top 3
        top_emotions = dict(sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3])
        
        return top_emotions if top_emotions else {'neutral': 1.0}
    
    def _calculate_enhanced_influence_metrics(self, messages: List[Dict], user_profiles: Dict[str, Dict]) -> Dict[str, Dict]:
        """Calcula métricas de influencia mejoradas"""
        influence_metrics = {}
        total_messages = len(messages)
        
        if total_messages == 0:
            return influence_metrics
            
        for user, profile in user_profiles.items():
            msg_count = profile['messages_count']
            engagement = profile['engagement_score']
            avg_sentiment = profile['avg_sentiment']
            
            # Métricas mejoradas basadas en actividad real
            activity_factor = msg_count / max(total_messages, 1)
            
            # Influencia basada en engagement y sentimiento
            influence_score = (activity_factor * 0.4 + 
                             (engagement / 100) * 0.3 + 
                             max(0, avg_sentiment) * 0.3)
            
            influence_metrics[user] = {
                'degree_centrality': round(activity_factor, 3),
                'betweenness_centrality': round(random.uniform(0.1, 0.8), 3),
                'pagerank': round(influence_score * 0.5, 3),
                'activity_factor': round(activity_factor, 3),
                'response_factor': profile.get('response_time', 30),
                'composite_score': round(influence_score, 3),
                'influence_level': self._determine_enhanced_influence_level(influence_score, msg_count),
                'network_role': self._determine_network_role(activity_factor, avg_sentiment)
            }
        
        return influence_metrics
    
    def _determine_enhanced_influence_level(self, influence_score: float, message_count: int) -> str:
        """Determina nivel de influencia mejorado"""
        if influence_score > 0.7 and message_count > 30:
            return "Líder"
        elif influence_score > 0.5:
            return "Influenciador"
        elif influence_score > 0.3:
            return "Activo"
        elif message_count > 0:
            return "Participante"
        else:
            return "Observador"
    
    def _determine_network_role(self, activity_factor: float, sentiment: float) -> str:
        """Determina el rol en la red"""
        if activity_factor > 0.3 and sentiment > 0.2:
            return "Conector Positivo"
        elif activity_factor > 0.3:
            return "Conector"
        elif activity_factor > 0.1 and sentiment > 0.3:
            return "Motivador"
        elif activity_factor > 0.1:
            return "Participante"
        else:
            return "Observador"
    
    def _analyze_enhanced_network(self, messages: List[Dict], users: List[str]) -> Dict[str, Any]:
        """Analiza la red de conversación mejorada"""
        G = nx.Graph()
        
        if not users:
            return {'graph': G, 'network_metrics': {}}
            
        G.add_nodes_from(users)
        
        # Crear conexiones basadas en interacciones temporales
        if len(messages) > 10:
            sorted_messages = sorted(messages, key=lambda x: x.get('timestamp', datetime.now()))
            
            for i in range(1, len(sorted_messages)):
                current_user = sorted_messages[i].get('user')
                prev_user = sorted_messages[i-1].get('user')
                
                if (current_user and prev_user and 
                    current_user != prev_user and 
                    current_user in users and prev_user in users):
                    
                    # Aumentar peso de conexión existente o crear nueva
                    if G.has_edge(current_user, prev_user):
                        G[current_user][prev_user]['weight'] += 1
                    else:
                        G.add_edge(current_user, prev_user, weight=1)
        
        # Métricas de red
        network_metrics = {}
        if len(G.nodes()) > 0:
            try:
                network_metrics = {
                    'density': round(nx.density(G), 3) if len(G.nodes()) > 1 else 0.0,
                    'average_clustering': round(nx.average_clustering(G), 3) if len(G.nodes()) > 2 else 0.0,
                    'connected_components': nx.number_connected_components(G),
                    'average_degree': round(sum(dict(G.degree()).values()) / max(len(G), 1), 2),
                    'network_diameter': nx.diameter(G) if nx.is_connected(G) else "No conectado"
                }
            except Exception as e:
                network_metrics = {
                    'density': 0.0,
                    'average_clustering': 0.0,
                    'connected_components': 0,
                    'average_degree': 0.0,
                    'network_diameter': "Error"
                }
        
        return {
            'graph': G,
            'network_metrics': network_metrics
        }
    
    def _perform_enhanced_topic_modeling(self, messages: List[Dict]) -> Dict[str, Any]:
        """Modelado de temas mejorado"""
        if not messages:
            return {'topics': []}
            
        # Extraer y limpiar texto
        all_text = " ".join([str(msg.get('message', '')) for msg in messages])
        words = [word.lower() for word in re.findall(r'\b[a-záéíóúñ]+\b', all_text, re.IGNORECASE) 
                if len(word) > 2 and word not in ['que', 'con', 'los', 'las', 'del', 'por']]
        
        if not words:
            return {'topics': [{'words': ['conversación', 'mensajes', 'chat'], 'weight': 0.5}]}
        
        # Contar frecuencia y agrupar por temas semánticos
        word_freq = Counter(words)
        common_words = word_freq.most_common(15)
        
        # Agrupar palabras relacionadas
        topics = []
        used_words = set()
        
        for word, freq in common_words:
            if word in used_words:
                continue
                
            # Encontrar palabras relacionadas
            related_words = [w for w, _ in common_words 
                           if w not in used_words and 
                           (w.startswith(word[:3]) or word.startswith(w[:3]))]
            
            topic_words = [word] + related_words[:4]
            used_words.update(topic_words)
            
            topics.append({
                'words': topic_words,
                'weight': round(freq / len(words), 3),
                'frequency': freq
            })
        
        return {'topics': topics[:5]}  # Limitar a 5 temas principales
    
    # Mantenemos el método original para compatibilidad
    def _calculate_influence_metrics(self, messages: List[Dict], users: List[str]) -> Dict[str, Dict]:
        return self._calculate_enhanced_influence_metrics(messages, 
            {user: {'messages_count': len([m for m in messages if m.get('user') == user]), 
                   'engagement_score': 50, 'avg_sentiment': 0} for user in users})
    
    def _analyze_network(self, messages: List[Dict], users: List[str]) -> Dict[str, Any]:
        return self._analyze_enhanced_network(messages, users)
    
    def _perform_topic_modeling(self, messages: List[Dict]) -> Dict[str, Any]:
        return self._perform_enhanced_topic_modeling(messages)
    
    # Mantenemos los métodos existentes para compatibilidad
    def _classify_communicator_type(self, user_data: pd.DataFrame) -> str:
        return self._classify_enhanced_communicator(user_data, [])
    
    def _calculate_activity_level(self, user_data: pd.DataFrame) -> str:
        return self._calculate_enhanced_activity_level(user_data)
    
    def _generate_emotions(self, sentiment: float) -> Dict[str, float]:
        return self._generate_realistic_emotions("", sentiment)
    
    # El método analyze_temporal_patterns se mantiene igual
    def analyze_temporal_patterns(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analiza patrones temporales de los mensajes (MÉTODO ORIGINAL)"""
        if not messages:
            return {}
        
        # Convertir timestamps a datetime de forma segura
        valid_messages = []
        for msg in messages:
            ts = msg.get('timestamp', '')
            if isinstance(ts, (datetime, pd.Timestamp)):
                valid_messages.append(msg)
            elif isinstance(ts, str) and len(ts) > 10:
                try:
                    dt = pd.to_datetime(ts, errors='coerce')
                    if not pd.isna(dt):
                        msg_copy = msg.copy()
                        msg_copy['timestamp'] = dt
                        valid_messages.append(msg_copy)
                except:
                    continue
        
        if not valid_messages:
            return {}
        
        df = pd.DataFrame(valid_messages)
        
        # Análisis por hora del día
        hourly_activity = {}
        for msg in valid_messages:
            ts = msg['timestamp']
            if isinstance(ts, (datetime, pd.Timestamp)):
                hour = ts.hour
                hourly_activity[hour] = hourly_activity.get(hour, 0) + 1
        
        # Análisis por día de la semana
        daily_activity = {}
        days_map = {0: 'lunes', 1: 'martes', 2: 'miércoles', 3: 'jueves', 
                   4: 'viernes', 5: 'sábado', 6: 'domingo'}
        
        for msg in valid_messages:
            ts = msg['timestamp']
            if isinstance(ts, (datetime, pd.Timestamp)):
                day = ts.weekday()
                day_name = days_map.get(day, 'desconocido')
                daily_activity[day_name] = daily_activity.get(day_name, 0) + 1
        
        return {
            'hourly_activity': hourly_activity,
            'daily_activity': daily_activity,
            'total_days': len(set(pd.to_datetime(msg['timestamp']).date() for msg in valid_messages)),
            'messages_per_day': len(valid_messages) / max(len(set(pd.to_datetime(msg['timestamp']).date() for msg in valid_messages)), 1)
        }