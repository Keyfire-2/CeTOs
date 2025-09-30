import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

class AdvancedUserRoleAnalyzer:
    def __init__(self, df, analysis_data: Dict[str, Any] = None):
        self.df = df
        self.analysis_data = analysis_data
        self.user_profiles = {}
        
    def analyze_advanced_roles(self) -> Dict[str, Dict[str, Any]]:
        """Análisis avanzado de roles con machine learning"""
        if 'user' not in self.df.columns or len(self.df) < 10:
            return self._get_basic_roles()
        
        try:
            # Calcular métricas avanzadas para cada usuario
            user_metrics = self._calculate_user_metrics()
            
            if len(user_metrics) < 3:
                return self._get_rule_based_roles(user_metrics)
            
            # Aplicar clustering para detección automática de roles
            clustered_roles = self._perform_role_clustering(user_metrics)
            
            # Combinar con análisis basado en reglas
            final_roles = self._combine_role_analysis(user_metrics, clustered_roles)
            
            return final_roles
            
        except Exception as e:
            st.error(f"Error en análisis avanzado de roles: {e}")
            return self._get_rule_based_roles(self._calculate_user_metrics())
    
    def _calculate_user_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calcula métricas avanzadas para cada usuario"""
        user_metrics = {}
        
        for user in self.df['user'].unique():
            user_data = self.df[self.df['user'] == user]
            user_messages = user_data['message'].tolist()
            
            # Métricas básicas
            total_messages = len(user_data)
            total_words = user_data['message'].str.split().str.len().sum()
            avg_words = user_data['message'].str.split().str.len().mean()
            
            # Métricas de contenido
            media_count = user_data['message'].str.contains(
                'multimedia|media|omitido|omitted', case=False, na=False
            ).sum()
            
            question_count = user_data['message'].str.contains('\?', na=False).sum()
            exclamation_count = user_data['message'].str.contains('\!', na=False).sum()
            url_count = user_data['message'].str.contains('http', case=False, na=False).sum()
            
            # Métricas de emojis (si están disponibles)
            emoji_count = 0
            if 'emojis' in user_data.columns:
                emoji_count = user_data['emojis'].apply(
                    lambda x: len(x) if isinstance(x, list) else 0
                ).sum()
            
            # Métricas de sentimiento (si están disponibles)
            avg_sentiment = 0
            if 'sentiment' in user_data.columns:
                avg_sentiment = user_data['sentiment'].mean()
            
            # Métricas temporales
            response_time = self._calculate_avg_response_time(user, user_data)
            activity_consistency = self._calculate_activity_consistency(user_data)
            
            # Métricas de engagement
            engagement_score = self._calculate_engagement_score(user, total_messages, len(self.df))
            
            user_metrics[user] = {
                'total_messages': total_messages,
                'total_words': total_words,
                'avg_words_per_message': avg_words,
                'media_ratio': media_count / total_messages if total_messages > 0 else 0,
                'question_ratio': question_count / total_messages if total_messages > 0 else 0,
                'exclamation_ratio': exclamation_count / total_messages if total_messages > 0 else 0,
                'url_ratio': url_count / total_messages if total_messages > 0 else 0,
                'emoji_ratio': emoji_count / total_messages if total_messages > 0 else 0,
                'avg_sentiment': avg_sentiment,
                'avg_response_time': response_time,
                'activity_consistency': activity_consistency,
                'engagement_score': engagement_score,
                'message_share': total_messages / len(self.df)
            }
        
        return user_metrics
    
    def _calculate_avg_response_time(self, user: str, user_data: pd.DataFrame) -> float:
        """Calcula el tiempo promedio de respuesta del usuario"""
        try:
            if 'timestamp' not in user_data.columns:
                return 0.0
            
            # Ordenar mensajes por timestamp
            sorted_data = user_data.sort_values('timestamp')
            response_times = []
            
            for i in range(1, len(sorted_data)):
                current_time = sorted_data.iloc[i]['timestamp']
                prev_time = sorted_data.iloc[i-1]['timestamp']
                
                if pd.notna(current_time) and pd.notna(prev_time):
                    time_diff = (pd.to_datetime(current_time) - pd.to_datetime(prev_time)).total_seconds() / 60  # minutos
                    if time_diff < 240:  # Ignorar diferencias mayores a 4 horas
                        response_times.append(time_diff)
            
            return np.mean(response_times) if response_times else 0.0
        except:
            return 0.0
    
    def _calculate_activity_consistency(self, user_data: pd.DataFrame) -> float:
        """Calcula la consistencia de la actividad del usuario"""
        try:
            if 'timestamp' not in user_data.columns:
                return 0.0
            
            # Calcular desviación estándar de los intervalos entre mensajes
            timestamps = pd.to_datetime(user_data['timestamp'].dropna())
            if len(timestamps) < 2:
                return 0.0
            
            time_diffs = np.diff(timestamps.sort_values()).astype('timedelta64[m]').astype(float)
            if len(time_diffs) == 0:
                return 0.0
            
            # Invertir para que menor desviación = mayor consistencia
            consistency = 1 / (1 + np.std(time_diffs))
            return min(consistency, 1.0)
        except:
            return 0.0
    
    def _calculate_engagement_score(self, user: str, user_messages: int, total_messages: int) -> float:
        """Calcula un score de engagement para el usuario"""
        base_score = (user_messages / total_messages) * 100 if total_messages > 0 else 0
        
        # Bonus por actividad alta
        if user_messages > total_messages * 0.2:
            base_score += 20
        elif user_messages > total_messages * 0.1:
            base_score += 10
        
        return min(base_score, 100)
    
    def _perform_role_clustering(self, user_metrics: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Aplica clustering K-means para detectar roles automáticamente"""
        try:
            # Preparar datos para clustering
            metrics_df = pd.DataFrame.from_dict(user_metrics, orient='index')
            
            # Seleccionar características relevantes
            features = [
                'message_share', 'avg_words_per_message', 'question_ratio',
                'exclamation_ratio', 'engagement_score', 'activity_consistency'
            ]
            
            X = metrics_df[features].fillna(0)
            
            # Estandarizar características
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determinar número óptimo de clusters (3-5 roles)
            n_clusters = min(5, max(3, len(X_scaled) // 3))
            
            # Aplicar K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Mapear clusters a roles basados en centroides
            role_mapping = self._map_clusters_to_roles(kmeans.cluster_centers_, features)
            
            # Asignar roles a usuarios
            user_roles = {}
            for i, (user, _) in enumerate(user_metrics.items()):
                cluster_role = role_mapping.get(clusters[i], "Participante")
                user_roles[user] = cluster_role
            
            return user_roles
            
        except Exception as e:
            st.warning(f"Clustering falló, usando análisis basado en reglas: {e}")
            return {}
    
    def _map_clusters_to_roles(self, centroids: np.ndarray, features: List[str]) -> Dict[int, str]:
        """Mapea clusters de K-means a roles específicos"""
        role_mapping = {}
        
        for cluster_idx, centroid in enumerate(centroids):
            centroid_dict = dict(zip(features, centroid))
            
            # Definir roles basados en características del centroide
            if centroid_dict['message_share'] > 0.5 and centroid_dict['engagement_score'] > 0.6:
                role = "Líder de Conversación"
            elif centroid_dict['avg_words_per_message'] > 0.8 and centroid_dict['question_ratio'] > 0.3:
                role = "Comunicador Activo"
            elif centroid_dict['exclamation_ratio'] > 0.4 and centroid_dict['engagement_score'] > 0.5:
                role = "Entusiasta"
            elif centroid_dict['message_share'] < 0.1 and centroid_dict['activity_consistency'] < 0.3:
                role = "Observador"
            elif centroid_dict['question_ratio'] > 0.5:
                role = "Cuestionador"
            else:
                role = "Participante"
            
            role_mapping[cluster_idx] = role
        
        return role_mapping
    
    def _combine_role_analysis(self, user_metrics: Dict, clustered_roles: Dict) -> Dict[str, Dict[str, Any]]:
        """Combina análisis por clustering con reglas basadas en dominio"""
        final_roles = {}
        
        for user, metrics in user_metrics.items():
            clustered_role = clustered_roles.get(user, "Participante")
            
            # Ajustar rol basado en reglas específicas
            final_role = self._apply_domain_rules(user, metrics, clustered_role)
            
            # Calcular confianza del rol
            confidence = self._calculate_role_confidence(metrics, final_role)
            
            final_roles[user] = {
                'role': final_role,
                'role_confidence': confidence,
                'metrics': metrics,
                'clustered_role': clustered_role,
                'role_category': self._categorize_role(final_role)
            }
        
        return final_roles
    
    def _apply_domain_rules(self, user: str, metrics: Dict, clustered_role: str) -> str:
        """Aplica reglas de dominio específicas para refinar el rol"""
        # Regla: Usuario con mucho multimedia -> Compartidor de Contenido
        if metrics['media_ratio'] > 0.3:
            return "Compartidor de Contenido"
        
        # Regla: Usuario con muchas preguntas -> Cuestionador
        if metrics['question_ratio'] > 0.4:
            return "Cuestionador"
        
        # Regla: Usuario con muchas exclamaciones -> Entusiasta
        if metrics['exclamation_ratio'] > 0.5:
            return "Entusiasta"
        
        # Regla: Usuario con muchos enlaces -> Recursos
        if metrics['url_ratio'] > 0.2:
            return "Proveedor de Recursos"
        
        # Regla: Usuario con sentimiento muy positivo -> Motivador
        if metrics['avg_sentiment'] > 0.5:
            return "Motivador"
        
        # Regla: Usuario con sentimiento muy negativo -> Crítico
        if metrics['avg_sentiment'] < -0.3:
            return "Crítico Constructivo"
        
        return clustered_role
    
    def _calculate_role_confidence(self, metrics: Dict, role: str) -> float:
        """Calcula la confianza en la asignación del rol"""
        confidence = 0.5  # Confianza base
        
        # Aumentar confianza basado en métricas específicas del rol
        role_indicators = {
            "Líder de Conversación": ['message_share', 'engagement_score'],
            "Comunicador Activo": ['avg_words_per_message', 'question_ratio'],
            "Entusiasta": ['exclamation_ratio', 'engagement_score'],
            "Compartidor de Contenido": ['media_ratio'],
            "Cuestionador": ['question_ratio'],
            "Motivador": ['avg_sentiment', 'exclamation_ratio'],
            "Crítico Constructivo": ['avg_sentiment']
        }
        
        if role in role_indicators:
            for indicator in role_indicators[role]:
                if indicator in metrics:
                    confidence += metrics[indicator] * 0.2
        
        return min(confidence, 1.0)
    
    def _categorize_role(self, role: str) -> str:
        """Categoriza el rol en grupos más amplios"""
        leadership_roles = ["Líder de Conversación", "Motivador"]
        active_roles = ["Comunicador Activo", "Entusiasta", "Cuestionador"]
        content_roles = ["Compartidor de Contenido", "Proveedor de Recursos"]
        support_roles = ["Crítico Constructivo"]
        
        if role in leadership_roles:
            return "Liderazgo"
        elif role in active_roles:
            return "Participación Activa"
        elif role in content_roles:
            return "Generación de Contenido"
        elif role in support_roles:
            return "Roles de Soporte"
        else:
            return "Participación General"
    
    def _get_rule_based_roles(self, user_metrics: Dict) -> Dict[str, Dict[str, Any]]:
        """Análisis de roles basado en reglas simples (fallback)"""
        roles = {}
        
        for user, metrics in user_metrics.items():
            if metrics['message_share'] > 0.3:
                role = "Líder"
            elif metrics['avg_words_per_message'] > 15:
                role = "Comunicador"
            elif metrics['media_ratio'] > 0.2:
                role = "Multimedia"
            elif metrics['question_ratio'] > 0.3:
                role = "Cuestionador"
            else:
                role = "Participante"
            
            roles[user] = {
                'role': role,
                'role_confidence': 0.7,
                'metrics': metrics,
                'clustered_role': role,
                'role_category': self._categorize_role(role)
            }
        
        return roles
    
    def _get_basic_roles(self) -> Dict[str, Dict[str, Any]]:
        """Roles muy básicos para datos insuficientes"""
        if 'user' not in self.df.columns:
            return {}
        
        roles = {}
        for user in self.df['user'].unique():
            user_data = self.df[self.df['user'] == user]
            roles[user] = {
                'role': "Participante",
                'role_confidence': 0.5,
                'metrics': {'total_messages': len(user_data)},
                'clustered_role': "Participante",
                'role_category': "Participación General"
            }
        
        return roles

def show_advanced_user_roles(df, analysis_data: Dict[str, Any] = None):
    """Visualización avanzada de roles de usuario"""
    if 'user' not in df.columns or len(df) < 5:
        st.info("👥 Se necesitan más datos para el análisis de roles")
        return
    
    st.markdown("### 👑 Análisis Avanzado de Roles de Usuario")
    
    # Configuración del análisis
    col1, col2 = st.columns(2)
    
    with col1:
        use_advanced = st.checkbox("Usar análisis avanzado con ML", value=True,
                                 help="Usa machine learning para detección más precisa de roles")
    
    with col2:
        if use_advanced:
            st.info("🤖 ML activado - Clustering + Reglas")
        else:
            st.info("📊 Análisis basado en reglas")
    
    # Realizar análisis
    analyzer = AdvancedUserRoleAnalyzer(df, analysis_data)
    
    with st.spinner("🔍 Analizando patrones de comportamiento..."):
        if use_advanced:
            roles_data = analyzer.analyze_advanced_roles()
        else:
            roles_data = analyzer._get_rule_based_roles(analyzer._calculate_user_metrics())
    
    if not roles_data:
        st.error("❌ No se pudieron analizar los roles de usuario")
        return
    
    # Mostrar resultados
    _display_advanced_role_analysis(roles_data)

def _display_advanced_role_analysis(roles_data: Dict[str, Dict[str, Any]]):
    """Muestra el análisis avanzado de roles"""
    
    # Métricas generales
    total_users = len(roles_data)
    role_categories = {}
    confidence_scores = []
    
    for user_data in roles_data.values():
        category = user_data['role_category']
        role_categories[category] = role_categories.get(category, 0) + 1
        confidence_scores.append(user_data['role_confidence'])
    
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Usuarios", total_users)
    
    with col2:
        st.metric("🎯 Roles Únicos", len(set(data['role'] for data in roles_data.values())))
    
    with col3:
        st.metric("📊 Categorías", len(role_categories))
    
    with col4:
        st.metric("✅ Confianza Promedio", f"{avg_confidence:.1%}")
    
    # Pestañas para diferentes visualizaciones
    tab1, tab2, tab3 = st.tabs(["👤 Roles Individuales", "📈 Distribución", "🎯 Análisis Detallado"])
    
    with tab1:
        _display_individual_roles(roles_data)
    
    with tab2:
        _display_role_distribution(roles_data)
    
    with tab3:
        _display_role_analysis_details(roles_data)

def _display_individual_roles(roles_data: Dict[str, Dict[str, Any]]):
    """Muestra los roles individuales de cada usuario"""
    st.markdown("#### 👤 Roles por Usuario")
    
    # Ordenar por confianza del rol
    sorted_roles = sorted(roles_data.items(), 
                         key=lambda x: x[1]['role_confidence'], 
                         reverse=True)
    
    for user, data in sorted_roles:
        confidence_color = "🟢" if data['role_confidence'] > 0.8 else "🟡" if data['role_confidence'] > 0.6 else "🟠"
        
        with st.expander(f"{confidence_color} **{user}** - {data['role']} ({data['role_confidence']:.1%})", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mensajes", f"{data['metrics']['total_messages']:,}")
                st.metric("Palabras/prom", f"{data['metrics']['avg_words_per_message']:.1f}")
            
            with col2:
                st.metric("Engagement", f"{data['metrics']['engagement_score']:.1f}%")
                st.metric("Consistencia", f"{data['metrics']['activity_consistency']:.2f}")
            
            with col3:
                st.metric("Categoría", data['role_category'])
                st.metric("Share", f"{data['metrics']['message_share']:.1%}")
            
            # Indicadores de comportamiento
            st.markdown("**Indicadores Clave:**")
            indicators = []
            
            if data['metrics']['question_ratio'] > 0.3:
                indicators.append("🤔 Hace muchas preguntas")
            if data['metrics']['exclamation_ratio'] > 0.3:
                indicators.append("🎉 Muy expresivo/a")
            if data['metrics']['media_ratio'] > 0.2:
                indicators.append("📷 Comparte multimedia")
            if data['metrics']['url_ratio'] > 0.1:
                indicators.append("🔗 Comparte enlaces")
            if data['metrics']['avg_sentiment'] > 0.3:
                indicators.append("😊 Positivo/a")
            if data['metrics']['avg_sentiment'] < -0.2:
                indicators.append("😐 Crítico/a")
            
            if indicators:
                for indicator in indicators:
                    st.write(f"- {indicator}")
            else:
                st.write("- Comportamiento balanceado")

def _display_role_distribution(roles_data: Dict[str, Dict[str, Any]]):
    """Muestra la distribución de roles"""
    st.markdown("#### 📈 Distribución de Roles")
    
    # Preparar datos para gráficos
    role_counts = {}
    category_counts = {}
    
    for data in roles_data.values():
        role = data['role']
        category = data['role_category']
        
        role_counts[role] = role_counts.get(role, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        if role_counts:
            fig = px.pie(
                values=list(role_counts.values()),
                names=list(role_counts.keys()),
                title="Distribución de Roles Específicos",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if category_counts:
            fig = px.bar(
                x=list(category_counts.values()),
                y=list(category_counts.keys()),
                orientation='h',
                title="Distribución por Categorías",
                color=list(category_counts.values()),
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_title="Número de Usuarios", yaxis_title="Categoría")
            st.plotly_chart(fig, use_container_width=True)

def _display_role_analysis_details(roles_data: Dict[str, Dict[str, Any]]):
    """Muestra análisis detallado de los roles"""
    st.markdown("#### 🎯 Análisis Detallado de Comportamiento")
    
    # Crear DataFrame para análisis
    analysis_data = []
    for user, data in roles_data.items():
        analysis_data.append({
            'Usuario': user,
            'Rol': data['role'],
            'Categoría': data['role_category'],
            'Confianza': data['role_confidence'],
            'Mensajes': data['metrics']['total_messages'],
            'Engagement': data['metrics']['engagement_score'],
            'Palabras_Promedio': data['metrics']['avg_words_per_message'],
            'Ratio_Preguntas': data['metrics']['question_ratio'],
            'Ratio_Exclamaciones': data['metrics']['exclamation_ratio'],
            'Sentimiento_Promedio': data['metrics']['avg_sentiment']
        })
    
    df_analysis = pd.DataFrame(analysis_data)
    
    # Mostrar tabla interactiva
    st.dataframe(
        df_analysis.sort_values('Confianza', ascending=False),
        use_container_width=True,
        height=400
    )
    
    # Recomendaciones basadas en el análisis
    st.markdown("#### 💡 Recomendaciones")
    
    role_recommendations = {
        "Líder de Conversación": "Reconocer su liderazgo y potencialmente asignar responsabilidades",
        "Comunicador Activo": "Involucrar en discusiones importantes y solicitar su opinión",
        "Entusiasta": "Utilizar su energía positiva para motivar al grupo",
        "Cuestionador": "Valorar su pensamiento crítico y usarlo para mejorar procesos",
        "Compartidor de Contenido": "Fomentar que continúe compartiendo recursos valiosos",
        "Motivador": "Reconocer su impacto positivo en la moral del grupo",
        "Crítico Constructivo": "Escuchar sus comentarios para identificar áreas de mejora",
        "Observador": "Involucrar gradualmente y hacer sentir incluido/a"
    }
    
    for role, recommendation in role_recommendations.items():
        if any(data['role'] == role for data in roles_data.values()):
            st.write(f"**{role}:** {recommendation}")

# Función de compatibilidad con versión anterior
class UserRoleAnalyzer(AdvancedUserRoleAnalyzer):
    """Clase de compatibilidad con la versión anterior"""
    def analyze_roles(self):
        """Método de compatibilidad con la versión anterior"""
        return self._get_rule_based_roles(self._calculate_user_metrics())

def show_user_roles(df):
    """Función de compatibilidad con la versión anterior"""
    analyzer = UserRoleAnalyzer(df)
    roles = analyzer.analyze_roles()
    
    if roles:
        st.subheader("👥 Roles de Usuario")
        
        for user, data in roles.items():
            st.markdown(f"""
            <div class="user-role-card">
                <h4>{user}</h4>
                <p><strong>Rol:</strong> {data['role']}</p>
                <p><strong>Mensajes:</strong> {data['metrics']['total_messages']} | <strong>Palabras promedio:</strong> {data['metrics']['avg_words_per_message']:.1f}</p>
            </div>
            """, unsafe_allow_html=True)