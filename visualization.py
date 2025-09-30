# visualization.py - Motor de visualizaciones avanzado mejorado
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import base64
from io import BytesIO

class VisualizationEngine:
    def __init__(self):
        # Paleta de colores mejorada
        self.color_palette = px.colors.qualitative.Bold
        self.sentiment_colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']  # Neg, Neu, Pos
        self.emotion_colors = px.colors.qualitative.Set3
        self.network_colors = px.colors.sequential.Viridis
        
    def display_key_metrics(self, metrics: Dict[str, Any]):
        """Muestra métricas clave en tarjetas visuales mejoradas"""
        st.markdown("### 📊 Dashboard de Métricas Principales")
        
        # Primera fila - Métricas principales
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            self._create_metric_card(
                "💬 Mensajes Totales", 
                f"{metrics.get('total_messages', 0):,}",
                "green" if metrics.get('total_messages', 0) > 50 else "orange"
            )
        
        with col2:
            self._create_metric_card(
                "👥 Usuarios Únicos", 
                str(metrics.get('total_users', 0)),
                "green" if metrics.get('total_users', 0) > 2 else "orange"
            )
        
        with col3:
            sentiment = metrics.get('avg_sentiment', 0)
            sentiment_icon = "😊" if sentiment > 0.1 else "😐" if sentiment > -0.1 else "😞"
            sentiment_color = "green" if sentiment > 0.1 else "orange" if sentiment > -0.1 else "red"
            self._create_metric_card(
                f"{sentiment_icon} Sentimiento", 
                f"{sentiment:.3f}",
                sentiment_color
            )
        
        with col4:
            density = metrics.get('message_density', 0)
            self._create_metric_card(
                "📈 Densidad/hora", 
                f"{density:.1f}",
                "green" if density > 2 else "orange"
            )
        
        with col5:
            engagement = metrics.get('engagement_rate', 0)
            self._create_metric_card(
                "🎯 Engagement", 
                f"{engagement:.1f}%",
                "green" if engagement > 50 else "orange"
            )
        
        # Segunda fila - Métricas adicionales
        if any(key in metrics for key in ['conversation_health', 'response_rate', 'messages_per_user']):
            col6, col7, col8, col9 = st.columns(4)
            
            with col6:
                health = metrics.get('conversation_health', 'N/A')
                health_color = "green" if health == "Excelente" else "orange" if health == "Buena" else "red"
                self._create_metric_card("❤️ Salud Conversación", health, health_color)
            
            with col7:
                response_rate = metrics.get('response_rate', 0)
                self._create_metric_card("⚡ Tasa Respuesta", f"{response_rate:.1f}%", 
                                       "green" if response_rate > 50 else "orange")
            
            with col8:
                msgs_per_user = metrics.get('messages_per_user', 0)
                self._create_metric_card("📨 Msgs/Usuario", f"{msgs_per_user:.1f}",
                                       "green" if msgs_per_user > 10 else "orange")
            
            with col9:
                words_per_msg = metrics.get('words_per_message', 0)
                self._create_metric_card("📝 Palabras/Msg", f"{words_per_msg:.1f}",
                                       "green" if words_per_msg > 5 else "orange")
    
    def _create_metric_card(self, title: str, value: str, color: str):
        """Crea una tarjeta de métrica visualmente atractiva"""
        color_map = {
            "green": "#10b981",
            "orange": "#f59e0b", 
            "red": "#ef4444"
        }
        
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, {color_map.get(color, '#6b7280')}, #1f2937);
            padding: 1.2rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin: 0.3rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-left: 4px solid {color_map.get(color, '#6b7280')};
            transition: transform 0.2s ease;
        '>
            <div style='font-size: 0.9rem; opacity: 0.9;'>{title}</div>
            <div style='font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;'>{value}</div>
        </div>
        """, unsafe_allow_html=True)
    
    def display_sentiment_analysis(self, messages: List[Dict[str, Any]], metrics: Dict[str, Any]):
        """Visualización avanzada de análisis de sentimiento"""
        st.markdown("### 😊 Análisis de Sentimiento Avanzado")
        
        if not messages:
            st.info("No hay datos para mostrar análisis de sentimiento")
            return
            
        df = pd.DataFrame(messages)
        
        # Crear pestañas para diferentes visualizaciones
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribución", "📈 Evolución", "🎭 Emociones", "👤 Por Usuario"])
        
        with tab1:
            self._display_sentiment_distribution(df)
        
        with tab2:
            self._display_sentiment_evolution(df)
        
        with tab3:
            self._display_emotion_analysis(messages)
        
        with tab4:
            self._display_user_sentiment_analysis(df, messages)
    
    def _display_sentiment_distribution(self, df: pd.DataFrame):
        """Muestra distribución de sentimiento"""
        fig = px.histogram(df, x='sentiment', nbins=25, 
                          title='Distribución de Sentimiento en Mensajes',
                          color_discrete_sequence=['#6366f1'],
                          opacity=0.8)
        
        # Añadir líneas de referencia
        fig.add_vline(x=0.1, line_dash="dash", line_color="green", annotation_text="Positivo")
        fig.add_vline(x=-0.1, line_dash="dash", line_color="red", annotation_text="Negativo")
        
        fig.update_layout(
            xaxis_title='Puntuación de Sentimiento', 
            yaxis_title='Número de Mensajes',
            showlegend=False,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas rápidas
        col1, col2, col3 = st.columns(3)
        with col1:
            positive = len(df[df['sentiment'] > 0.1])
            st.metric("Mensajes Positivos", f"{positive} ({positive/len(df)*100:.1f}%)")
        with col2:
            neutral = len(df[(df['sentiment'] >= -0.1) & (df['sentiment'] <= 0.1)])
            st.metric("Mensajes Neutrales", f"{neutral} ({neutral/len(df)*100:.1f}%)")
        with col3:
            negative = len(df[df['sentiment'] < -0.1])
            st.metric("Mensajes Negativos", f"{negative} ({negative/len(df)*100:.1f}%)")
    
    def _display_sentiment_evolution(self, df: pd.DataFrame):
        """Muestra evolución temporal del sentimiento"""
        if len(df) < 10:
            st.info("Se necesitan más mensajes para mostrar la evolución temporal")
            return
            
        df_sorted = df.sort_values('timestamp')
        
        # Crear subplots
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Evolución del Sentimiento', 'Volumen de Mensajes por Día'),
                           vertical_spacing=0.1)
        
        # Sentimiento rolling
        window_size = min(20, len(df_sorted) // 10)
        df_sorted['rolling_sentiment'] = df_sorted['sentiment'].rolling(window=window_size, center=True).mean()
        df_sorted['date'] = pd.to_datetime(df_sorted['timestamp']).dt.date
        
        # Gráfico de línea para sentimiento
        fig.add_trace(
            go.Scatter(x=df_sorted['timestamp'], y=df_sorted['rolling_sentiment'],
                      mode='lines', name='Sentimiento (Media Móvil)',
                      line=dict(color='#00f2fe', width=3)),
            row=1, col=1
        )
        
        # Gráfico de barras para volumen
        daily_volume = df_sorted.groupby('date').size()
        fig.add_trace(
            go.Bar(x=list(daily_volume.index), y=daily_volume.values,
                  name='Mensajes por Día', marker_color='#4facfe'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=False, template='plotly_white')
        fig.update_xaxes(title_text="Fecha", row=2, col=1)
        fig.update_yaxes(title_text="Sentimiento", row=1, col=1)
        fig.update_yaxes(title_text="Número de Mensajes", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_emotion_analysis(self, messages: List[Dict[str, Any]]):
        """Muestra análisis de emociones"""
        emotion_data = []
        for msg in messages:
            emotions = msg.get('emotions', {})
            for emotion, score in emotions.items():
                if score > 0.05:  # Umbral más bajo para incluir más emociones
                    emotion_data.append({
                        'user': msg['user'],
                        'emotion': emotion,
                        'score': score
                    })
        
        if not emotion_data:
            st.info("No se detectaron emociones significativas en los mensajes")
            return
            
        df_emotions = pd.DataFrame(emotion_data)
        
        # Heatmap de emociones por usuario
        emotion_pivot = df_emotions.pivot_table(
            index='user', columns='emotion', values='score', aggfunc='mean'
        ).fillna(0)
        
        # Ordenar por frecuencia total de emociones
        emotion_totals = emotion_pivot.sum(axis=0)
        emotion_pivot = emotion_pivot[emotion_totals.sort_values(ascending=False).index]
        
        fig = px.imshow(emotion_pivot, 
                       title='Mapa de Calor de Emociones por Usuario',
                       color_continuous_scale='Viridis',
                       aspect="auto")
        
        fig.update_layout(
            xaxis_title='Emociones',
            yaxis_title='Usuario',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribución general de emociones
        emotion_dist = df_emotions.groupby('emotion')['score'].mean().sort_values(ascending=False)
        fig2 = px.bar(x=emotion_dist.index, y=emotion_dist.values,
                     title='Distribución Promedio de Emociones',
                     color=emotion_dist.values,
                     color_continuous_scale='Viridis')
        fig2.update_layout(xaxis_title='Emoción', yaxis_title='Intensidad Promedio')
        st.plotly_chart(fig2, use_container_width=True)
    
    def _display_user_sentiment_analysis(self, df: pd.DataFrame, messages: List[Dict[str, Any]]):
        """Muestra análisis de sentimiento por usuario"""
        user_sentiment = df.groupby('user')['sentiment'].agg(['mean', 'count', 'std']).reset_index()
        user_sentiment = user_sentiment[user_sentiment['count'] >= 3]  # Filtrar usuarios con pocos mensajes
        
        if len(user_sentiment) == 0:
            st.info("No hay suficientes datos para análisis por usuario")
            return
            
        # Bubble chart: sentimiento vs actividad con tamaño por consistencia
        fig = px.scatter(user_sentiment, x='count', y='mean', size='count',
                        color='mean', hover_name='user',
                        title='Sentimiento vs Actividad por Usuario',
                        labels={'count': 'Número de Mensajes', 'mean': 'Sentimiento Promedio'},
                        color_continuous_scale='RdYlGn',
                        size_max=40)
        
        fig.update_layout(template='plotly_white')
        fig.update_traces(marker=dict(sizemode='diameter', line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig, use_container_width=True)
    
    def display_user_analysis(self, user_profiles: Dict[str, Dict], metrics: Dict[str, Any]):
        """Visualización mejorada de análisis de usuarios"""
        st.markdown("### 👥 Análisis de Participación de Usuarios")
        
        if not user_profiles:
            st.info("No hay datos de usuarios para analizar")
            return
            
        tab1, tab2, tab3 = st.tabs(["📈 Actividad", "🎭 Perfiles", "📊 Detalles"])
        
        with tab1:
            self._display_user_activity(user_profiles)
        
        with tab2:
            self._display_user_profiles(user_profiles)
        
        with tab3:
            self._display_user_details(user_profiles)
    
    def _display_user_activity(self, user_profiles: Dict[str, Dict]):
        """Muestra visualizaciones de actividad de usuarios"""
        users = list(user_profiles.keys())
        message_counts = [up['messages_count'] for up in user_profiles.values()]
        avg_sentiments = [up['avg_sentiment'] for up in user_profiles.values()]
        engagement_scores = [up.get('engagement_score', 50) for up in user_profiles.values()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de barras horizontal para mejor legibilidad
            fig = px.bar(x=message_counts, y=users, orientation='h',
                        title='Mensajes por Usuario',
                        color=message_counts,
                        color_continuous_scale='Viridis')
            fig.update_layout(yaxis_title='Usuario', xaxis_title='Número de Mensajes')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot mejorado
            fig = px.scatter(x=message_counts, y=avg_sentiments, text=users,
                            size=engagement_scores,
                            title='Actividad vs Sentimiento por Usuario',
                            labels={'x': 'Número de Mensajes', 'y': 'Sentimiento Promedio'},
                            color=engagement_scores,
                            color_continuous_scale='Viridis')
            fig.update_traces(textposition='top center', marker=dict(sizemode='diameter', sizeref=2.*max(engagement_scores)/(40.**2)))
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def _display_user_profiles(self, user_profiles: Dict[str, Dict]):
        """Muestra perfiles de usuario detallados"""
        st.subheader("Perfiles de Comunicación")
        
        for user, profile in user_profiles.items():
            with st.expander(f"👤 {user} - {profile.get('communicator_type', 'Usuario')}", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mensajes", profile['messages_count'])
                
                with col2:
                    st.metric("Sentimiento", f"{profile['avg_sentiment']:.3f}")
                
                with col3:
                    st.metric("Engagement", f"{profile.get('engagement_score', 0):.1f}%")
                
                with col4:
                    st.metric("Actividad", profile.get('activity_level', 'N/A'))
                
                # Emociones principales
                top_emotions = profile.get('top_emotions', {})
                if top_emotions:
                    st.write("**Emociones principales:**")
                    for emotion, score in list(top_emotions.items())[:3]:
                        st.progress(score, text=f"{emotion}: {score:.2f}")
    
    def _display_user_details(self, user_profiles: Dict[str, Dict]):
        """Muestra tabla detallada de usuarios"""
        user_data = []
        for user, profile in user_profiles.items():
            user_data.append({
                'Usuario': user,
                'Mensajes': profile['messages_count'],
                'Palabras': profile.get('words_count', 0),
                'Sentimiento': f"{profile['avg_sentiment']:.3f}",
                'Tipo': profile.get('communicator_type', 'N/A'),
                'Actividad': profile.get('activity_level', 'N/A'),
                'Engagement': f"{profile.get('engagement_score', 0):.1f}%",
                'Respuesta (min)': profile.get('response_time', 'N/A')
            })
        
        df_users = pd.DataFrame(user_data)
        st.dataframe(df_users, use_container_width=True)
    
    def display_influence_analysis(self, influence_metrics: Dict[str, Dict], user_profiles: Dict[str, Dict]):
        """Visualización mejorada de análisis de influencia"""
        st.markdown("### 👑 Análisis de Influencia y Liderazgo")
        
        if not influence_metrics:
            st.info("No hay datos de influencia para analizar")
            return
            
        tab1, tab2, tab3 = st.tabs(["📊 Métricas", "🏆 Ranking", "🎯 Roles"])
        
        with tab1:
            self._display_influence_metrics(influence_metrics)
        
        with tab2:
            self._display_influence_ranking(influence_metrics, user_profiles)
        
        with tab3:
            self._display_network_roles(influence_metrics)
    
    def _display_influence_metrics(self, influence_metrics: Dict[str, Dict]):
        """Muestra métricas de influencia en gráfico radar"""
        users = list(influence_metrics.keys())[:6]  # Máximo 6 usuarios para claridad
        metrics_list = ['degree_centrality', 'betweenness_centrality', 'pagerank', 'activity_factor']
        
        fig = go.Figure()
        
        for user in users:
            metrics_values = [
                influence_metrics[user]['degree_centrality'],
                influence_metrics[user]['betweenness_centrality'] * 2,  # Escalar para mejor visualización
                influence_metrics[user]['pagerank'] * 3,
                influence_metrics[user]['activity_factor'] * 2
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=metrics_values,
                theta=['Centralidad', 'Intermediación', 'PageRank', 'Actividad'],
                fill='toself',
                name=user
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            title='Comparación de Métricas de Influencia',
            template='plotly_white',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_influence_ranking(self, influence_metrics: Dict[str, Dict], user_profiles: Dict[str, Dict]):
        """Muestra ranking de influencia mejorado"""
        sorted_influence = sorted(influence_metrics.items(), 
                                key=lambda x: x[1]['composite_score'], reverse=True)
        
        st.subheader("🏆 Ranking de Influencia")
        
        for i, (user, metrics) in enumerate(sorted_influence[:10], 1):
            profile = user_profiles.get(user, {})
            
            col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
            
            with col1:
                # Icono de medalla para top 3
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
                st.markdown(f"<h3 style='text-align: center;'>{medal}</h3>", unsafe_allow_html=True)
            
            with col2:
                st.write(f"**{user}**")
                st.write(f"*{metrics['influence_level']}* - {profile.get('communicator_type', 'Usuario')}")
                st.progress(float(metrics['composite_score']))
            
            with col3:
                st.write(f"**Score:** {metrics['composite_score']:.3f}")
                st.write(f"**Mensajes:** {profile.get('messages_count', 0)}")
            
            with col4:
                st.write(f"**Rol:** {metrics.get('network_role', 'N/A')}")
                st.write(f"**Actividad:** {metrics['activity_factor']:.1%}")
            
            st.markdown("---")
    
    def _display_network_roles(self, influence_metrics: Dict[str, Dict]):
        """Muestra distribución de roles en la red"""
        roles = {}
        for metrics in influence_metrics.values():
            role = metrics.get('network_role', 'Participante')
            roles[role] = roles.get(role, 0) + 1
        
        if roles:
            fig = px.pie(values=list(roles.values()), names=list(roles.keys()),
                        title='Distribución de Roles en la Red',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
    
    def display_network_analysis(self, network_data: Dict[str, Any], user_profiles: Dict[str, Dict], influence_metrics: Dict[str, Dict]):
        """Visualización mejorada de análisis de red"""
        st.markdown("### 🌐 Análisis de Red de Conversación")
        
        G = network_data['graph']
        
        if len(G.nodes()) == 0:
            st.warning("No hay suficientes interacciones para visualizar la red")
            return
        
        tab1, tab2 = st.tabs(["🕸️ Visualización", "📈 Métricas"])
        
        with tab1:
            self._display_network_graph(G, user_profiles, influence_metrics)
        
        with tab2:
            self._display_network_metrics(network_data)
    
    def _display_network_graph(self, G: nx.Graph, user_profiles: Dict[str, Dict], influence_metrics: Dict[str, Dict]):
        """Muestra gráfico de red interactivo mejorado"""
        # Usar layout de resorte mejorado
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        
        # Preparar datos de edges
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2].get('weight', 1))
        
        # Trace para edges con grosor basado en peso
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Preparar datos de nodes
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        node_names = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_names.append(node)
            
            influence = influence_metrics.get(node, {}).get('composite_score', 0)
            messages = user_profiles.get(node, {}).get('messages_count', 0)
            role = influence_metrics.get(node, {}).get('network_role', 'Participante')
            
            node_text.append(
                f"<b>{node}</b><br>"
                f"Mensajes: {messages}<br>"
                f"Influencia: {influence:.3f}<br>"
                f"Rol: {role}"
            )
            node_size.append(15 + min(messages / 5, 35))  # Tamaño más balanceado
            node_color.append(influence)
        
        # Trace para nodes
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_names,
            textposition="middle center",
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Influencia'),
                line=dict(width=2, color='darkblue')
            ),
            textfont=dict(size=10, color='white')
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='<b>Red de Conversación</b><br><i>Tamaño: Actividad | Color: Influencia</i>',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=80),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white',
                           height=600
                       ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_network_metrics(self, network_data: Dict[str, Any]):
        """Muestra métricas de red en tarjetas"""
        metrics = network_data.get('network_metrics', {})
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            self._create_metric_card("Densidad", f"{metrics.get('density', 0):.3f}", 
                                   "green" if metrics.get('density', 0) > 0.3 else "orange")
        
        with col2:
            self._create_metric_card("Agrupamiento", f"{metrics.get('average_clustering', 0):.3f}",
                                   "green" if metrics.get('average_clustering', 0) > 0.5 else "orange")
        
        with col3:
            self._create_metric_card("Componentes", str(metrics.get('connected_components', 0)),
                                   "green" if metrics.get('connected_components', 0) == 1 else "orange")
        
        with col4:
            self._create_metric_card("Grado Promedio", f"{metrics.get('average_degree', 0):.1f}",
                                   "green" if metrics.get('average_degree', 0) > 2 else "orange")
        
        with col5:
            diameter = metrics.get('network_diameter', 'N/A')
            if isinstance(diameter, (int, float)):
                self._create_metric_card("Diámetro", str(diameter),
                                       "green" if diameter <= 3 else "orange")
            else:
                self._create_metric_card("Diámetro", str(diameter), "orange")
        
        # Interpretación de métricas
        st.info("""
        **Interpretación:**
        - **Densidad alta**: Muchas conexiones entre usuarios
        - **Agrupamiento alto**: Formación de subgrupos o comunidades  
        - **Un componente**: Todos conectados directamente o indirectamente
        - **Grado alto**: Usuarios muy conectados
        - **Diámetro bajo**: Información fluye rápidamente
        """)
    
    def display_topic_analysis(self, topic_modeling: Dict[str, Any]):
        """Visualización mejorada de análisis de temas"""
        st.markdown("### 🎯 Modelado de Temas de Conversación")
        
        if not topic_modeling.get('topics'):
            st.info("No se pudieron identificar temas significativos en la conversación")
            return
        
        topics = topic_modeling['topics']
        
        # Mostrar temas en tarjetas expandibles
        cols = st.columns(2)
        for i, topic in enumerate(topics):
            with cols[i % 2]:
                with st.expander(f"**Tema {i+1}** - {', '.join(topic['words'][:3])}", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write("**Palabras clave:**")
                        for word in topic['words'][:8]:
                            st.write(f"• {word}")
                    
                    with col2:
                        st.metric("Relevancia", f"{topic.get('weight', 0):.3f}")
                        if 'frequency' in topic:
                            st.metric("Frecuencia", topic['frequency'])
        
        # Word cloud simulation mejorada
        st.subheader("📊 Frecuencia de Palabras Clave")
        all_words = []
        for topic in topics:
            all_words.extend(topic['words'][:5])
        
        word_freq = pd.Series(all_words).value_counts()
        
        fig = px.bar(x=word_freq.values, y=word_freq.index, orientation='h',
                    title='Palabras más Frecuentes en los Temas Identificados',
                    color=word_freq.values,
                    color_continuous_scale='Viridis')
        fig.update_layout(xaxis_title='Frecuencia', yaxis_title='Palabra')
        st.plotly_chart(fig, use_container_width=True)
    
    def display_temporal_analysis(self, messages: List[Dict[str, Any]]):
        """Visualización mejorada de análisis temporal"""
        st.markdown("### ⏰ Análisis Temporal de Actividad")
        
        if not messages:
            st.info("No hay datos para análisis temporal")
            return
            
        df = pd.DataFrame(messages)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day_name()
        df['date'] = df['timestamp'].dt.date
        
        tab1, tab2, tab3 = st.tabs(["🕒 Por Hora", "📅 Por Día", "🔥 Mapa de Calor"])
        
        with tab1:
            self._display_hourly_activity(df)
        
        with tab2:
            self._display_daily_activity(df)
        
        with tab3:
            self._display_activity_heatmap(df)
    
    def _display_hourly_activity(self, df: pd.DataFrame):
        """Muestra actividad por hora del día"""
        hourly_activity = df.groupby('hour').size()
        
        fig = px.area(x=hourly_activity.index, y=hourly_activity.values,
                     title='Actividad por Hora del Día',
                     labels={'x': 'Hora del Día', 'y': 'Número de Mensajes'})
        fig.update_traces(line=dict(color='#4facfe', width=3), fill='tozeroy')
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas de horas pico
        peak_hour = hourly_activity.idxmax()
        peak_activity = hourly_activity.max()
        st.metric("⏰ Hora de Mayor Actividad", f"{peak_hour}:00 - {peak_hour+1}:00", 
                 f"{peak_activity} mensajes")
    
    def _display_daily_activity(self, df: pd.DataFrame):
        """Muestra actividad por día de la semana"""
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_names_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        
        daily_activity = df.groupby('day').size().reindex(day_order, fill_value=0)
        
        fig = px.bar(x=day_names_es, y=daily_activity.values,
                    title='Actividad por Día de la Semana',
                    color=daily_activity.values,
                    color_continuous_scale='Viridis')
        fig.update_layout(xaxis_title='Día de la Semana', yaxis_title='Número de Mensajes')
        st.plotly_chart(fig, use_container_width=True)
        
        # Día más activo
        peak_day_idx = daily_activity.values.argmax()
        peak_day = day_names_es[peak_day_idx]
        peak_activity = daily_activity.values[peak_day_idx]
        st.metric("📅 Día Más Activo", peak_day, f"{peak_activity} mensajes")
    
    def _display_activity_heatmap(self, df: pd.DataFrame):
        """Muestra heatmap de actividad (hora vs día)"""
        if len(df) < 50:
            st.info("Se necesitan más mensajes para generar el mapa de calor")
            return
            
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_names_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        
        df_heatmap = df.groupby(['day', 'hour']).size().unstack(fill_value=0)
        df_heatmap = df_heatmap.reindex(day_order)
        df_heatmap.index = day_names_es
        
        fig = px.imshow(df_heatmap, 
                       title='Mapa de Calor: Actividad por Día y Hora',
                       color_continuous_scale='Viridis',
                       labels=dict(x="Hora del Día", y="Día de la Semana", color="Mensajes"))
        st.plotly_chart(fig, use_container_width=True)
    
    def display_raw_data(self, messages: List[Dict[str, Any]], user_profiles: Dict[str, Dict]):
        """Muestra datos crudos y opciones de exportación mejoradas"""
        st.markdown("### 📋 Datos y Exportación")
        
        tab1, tab2, tab3 = st.tabs(["📨 Mensajes", "👥 Usuarios", "💾 Exportar"])
        
        with tab1:
            self._display_messages_data(messages)
        
        with tab2:
            self._display_users_data(user_profiles)
        
        with tab3:
            self._display_export_options(messages, user_profiles)
    
    def _display_messages_data(self, messages: List[Dict[str, Any]]):
        """Muestra tabla de mensajes con filtros"""
        st.subheader("Mensajes Analizados")
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        with col1:
            min_sentiment = st.slider("Sentimiento mínimo", -1.0, 1.0, -1.0, 0.1)
        with col2:
            max_sentiment = st.slider("Sentimiento máximo", -1.0, 1.0, 1.0, 0.1)
        with col3:
            users = list(set(msg.get('user', '') for msg in messages))
            selected_user = st.selectbox("Filtrar por usuario", ["Todos"] + users)
        
        # Aplicar filtros
        filtered_messages = messages
        if selected_user != "Todos":
            filtered_messages = [msg for msg in filtered_messages if msg.get('user') == selected_user]
        
        filtered_messages = [msg for msg in filtered_messages 
                           if min_sentiment <= msg.get('sentiment', 0) <= max_sentiment]
        
        # Mostrar tabla
        df_display = pd.DataFrame(filtered_messages[:200])[['timestamp', 'user', 'message', 'sentiment']]  # Limitar para rendimiento
        st.dataframe(df_display, use_container_width=True, height=400)
        
        st.write(f"Mostrando {len(filtered_messages)} de {len(messages)} mensajes")
    
    def _display_users_data(self, user_profiles: Dict[str, Dict]):
        """Muestra tabla detallada de usuarios"""
        st.subheader("Estadísticas de Usuarios")
        
        user_data = []
        for user, profile in user_profiles.items():
            user_data.append({
                'Usuario': user,
                'Mensajes': profile['messages_count'],
                'Palabras': profile.get('words_count', 0),
                'Sentimiento Prom': f"{profile['avg_sentiment']:.3f}",
                'Tipo Comunicador': profile.get('communicator_type', 'N/A'),
                'Nivel Actividad': profile.get('activity_level', 'N/A'),
                'Engagement': f"{profile.get('engagement_score', 0):.1f}%",
                'Tiempo Respuesta (min)': profile.get('response_time', 'N/A')
            })
        
        st.dataframe(pd.DataFrame(user_data), use_container_width=True)
    
    def _display_export_options(self, messages: List[Dict[str, Any]], user_profiles: Dict[str, Dict]):
        """Muestra opciones de exportación mejoradas"""
        st.subheader("Opciones de Exportación")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 CSV Mensajes", use_container_width=True):
                self._export_messages_csv(messages)
        
        with col2:
            if st.button("👥 CSV Usuarios", use_container_width=True):
                self._export_users_csv(user_profiles)
        
        with col3:
            if st.button("📈 Reporte Completo", use_container_width=True):
                self._export_complete_report(messages, user_profiles)
        
        with col4:
            if st.button("🔄 Nuevo Análisis", use_container_width=True):
                st.session_state.analysis_data = None
                st.rerun()
    
    def _export_messages_csv(self, messages: List[Dict[str, Any]]):
        """Exporta mensajes a CSV"""
        df_export = pd.DataFrame(messages)
        csv = df_export.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="⬇️ Descargar CSV de Mensajes",
            data=csv,
            file_name="whatsapp_mensajes.csv",
            mime="text/csv",
            key="download_messages_csv"
        )
    
    def _export_users_csv(self, user_profiles: Dict[str, Dict]):
        """Exporta estadísticas de usuarios a CSV"""
        user_data = []
        for user, profile in user_profiles.items():
            user_data.append({
                'Usuario': user,
                'Mensajes': profile['messages_count'],
                'Palabras': profile.get('words_count', 0),
                'Sentimiento_Promedio': profile['avg_sentiment'],
                'Tipo_Comunicador': profile.get('communicator_type', ''),
                'Nivel_Actividad': profile.get('activity_level', ''),
                'Engagement': profile.get('engagement_score', 0),
                'Tiempo_Respuesta_Min': profile.get('response_time', 0)
            })
        
        df_users = pd.DataFrame(user_data)
        csv = df_users.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="⬇️ Descargar CSV de Usuarios",
            data=csv,
            file_name="whatsapp_usuarios.csv",
            mime="text/csv",
            key="download_users_csv"
        )
    
    def _export_complete_report(self, messages: List[Dict[str, Any]], user_profiles: Dict[str, Dict]):
        """Genera y exporta reporte completo"""
        report = self.generate_enhanced_report(messages, user_profiles)
        st.download_button(
            label="⬇️ Descargar Reporte Completo",
            data=report,
            file_name="analisis_whatsapp_completo.txt",
            mime="text/plain",
            key="download_complete_report"
        )
    
    def generate_enhanced_report(self, messages: List[Dict[str, Any]], user_profiles: Dict[str, Dict]) -> str:
        """Genera reporte de análisis mejorado en texto plano"""
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("📊 REPORTE COMPLETO DE ANÁLISIS DE WHATSAPP")
        report_lines.append("=" * 70)
        report_lines.append(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total mensajes analizados: {len(messages):,}")
        report_lines.append(f"Usuarios únicos: {len(user_profiles)}")
        report_lines.append("")
        
        # Resumen de métricas generales
        if messages:
            avg_sentiment = np.mean([msg.get('sentiment', 0) for msg in messages])
            total_words = sum([msg.get('words_count', 0) for msg in messages])
            report_lines.append("📈 MÉTRICAS GENERALES:")
            report_lines.append("-" * 50)
            report_lines.append(f"Sentimiento promedio: {avg_sentiment:.3f}")
            report_lines.append(f"Palabras totales: {total_words:,}")
            report_lines.append(f"Palabras por mensaje: {total_words/len(messages):.1f}")
            report_lines.append("")
        
        # Resumen de usuarios
        report_lines.append("👥 RESUMEN DE USUARIOS:")
        report_lines.append("-" * 50)
        
        for user, profile in sorted(user_profiles.items(), 
                                  key=lambda x: x[1]['messages_count'], reverse=True):
            report_lines.append(
                f"• {user}: "
                f"{profile['messages_count']} mensajes | "
                f"{profile.get('words_count', 0)} palabras | "
                f"Sentimiento: {profile['avg_sentiment']:.3f} | "
                f"Tipo: {profile.get('communicator_type', 'N/A')}"
            )
        
        report_lines.append("")
        report_lines.append("🎯 RECOMENDACIONES:")
        report_lines.append("-" * 50)
        
        # Recomendaciones basadas en el análisis
        if len(user_profiles) > 5:
            report_lines.append("• Grupo muy activo con buena diversidad de participantes")
        elif len(user_profiles) > 2:
            report_lines.append("• Grupo de tamaño moderado, buena para conversaciones enfocadas")
        else:
            report_lines.append("• Conversación uno a uno, muy personalizada")
        
        avg_sent = np.mean([p['avg_sentiment'] for p in user_profiles.values()])
        if avg_sent > 0.2:
            report_lines.append("• Ambiente muy positivo en la conversación")
        elif avg_sent > 0:
            report_lines.append("• Ambiente generalmente positivo")
        else:
            report_lines.append("• Se detectó sentimiento negativo, considerar intervención")
        
        return "\n".join(report_lines)

    # Método original mantenido para compatibilidad
    def generate_report(self, messages: List[Dict[str, Any]], user_profiles: Dict[str, Dict]) -> str:
        return self.generate_enhanced_report(messages, user_profiles)