import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import traceback
import time
import plotly.express as px

sys.path.append(os.path.dirname(__file__))

from visualization import VisualizationEngine
from analysis import AdvancedAnalyzer
from database import DataManager
from data_cleaner import WhatsAppDataCleaner

# Configuración de la página
st.set_page_config(
    page_title="CeTOs Deep - WhatsApp Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema personalizado mejorado
def apply_custom_theme():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #84B026, #217373, #173540);
        -webkit-background-clip: text;
        -moz-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        -moz-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #84B026;
        text-align: center;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    .metric-card {
        background: linear-gradient(135deg, #173540, #161F30);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.3rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        border: 1px solid #217373;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(132, 176, 38, 0.1), transparent);
        transition: left 0.5s;
    }
    .metric-card:hover::before {
        left: 100%;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #84B026;
        box-shadow: 0 12px 35px rgba(132, 176, 38, 0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        margin-top: 0.5rem;
        color: #84B026;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #84B026;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #217373;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }
    .stButton>button {
        background: linear-gradient(135deg, #217373, #173540);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #84B026, #217373);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(132, 176, 38, 0.3);
    }
    
    /* Mejoras para la sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #161F30, #173540);
    }
    
    /* Mejoras para las pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #173540;
        border-radius: 12px;
        padding: 8px;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #161F30;
        color: white;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid #217373;
        flex: 1;
        text-align: center;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #84B026, #217373);
        color: #161F30;
        border-color: #84B026;
        transform: scale(1.02);
    }
    
    /* Progress bar personalizada */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #84B026, #217373);
    }
    
    /* Mejoras para los expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #173540, #161F30);
        color: white;
        border-radius: 8px;
        border: 1px solid #217373;
    }
    
    /* Loading spinner personalizado */
    .stSpinner > div {
        border-color: #84B026 transparent transparent transparent;
    }
    
    /* Mejoras para las tarjetas de información */
    .info-card {
        background: linear-gradient(135deg, #173540, #161F30);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #217373;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

class WhatsAppDashboard:
    def __init__(self):
        self.viz_engine = VisualizationEngine()
        self.analyzer = AdvancedAnalyzer()
        self.data_cleaner = WhatsAppDataCleaner()
        self.data_manager = DataManager()
        apply_custom_theme()
        
        # Inicializar session state
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = None
        if 'data_source' not in st.session_state:
            st.session_state.data_source = None
        if 'cleaning_stats' not in st.session_state:
            st.session_state.cleaning_stats = None
        if 'processing_stage' not in st.session_state:
            st.session_state.processing_stage = None
    
    def show_sidebar(self):
        """Sidebar con configuración mejorada"""
        st.sidebar.markdown('<div class="section-header">⚙️ Configuración</div>', unsafe_allow_html=True)
        
        # Selector de modo con iconos
        analysis_mode = st.sidebar.radio(
            "**Modo de análisis:**",
            ["📁 Cargar Archivo", "💾 Base de Datos", "🎮 Datos Demo"],
            help="Elige cómo quieres obtener los datos para analizar"
        )
        
        if analysis_mode == "📁 Cargar Archivo":
            return self._file_upload_controls()
        elif analysis_mode == "💾 Base de Datos":
            return self._database_controls()
        else:
            return self._demo_data_controls()
    
    def _file_upload_controls(self):
        """Controles mejorados para carga de archivos"""
        st.sidebar.markdown("### 📤 Subir Archivo")
        st.sidebar.info("""
        **Formatos soportados:**
        - 📝 **TXT**: Exportación de WhatsApp
        - 📊 **JSON**: Análisis previos
        """)
        
        uploaded_file = st.sidebar.file_uploader(
            "Selecciona tu archivo",
            type=['txt', 'json'],
            help="Arrastra o haz clic para seleccionar tu archivo de WhatsApp"
        )
        
        if uploaded_file is not None:
            # Información del archivo en tarjeta
            file_size = len(uploaded_file.getvalue()) / 1024
            st.sidebar.markdown(f"""
            <div class="info-card">
                <div style='font-weight: bold;'>📄 {uploaded_file.name}</div>
                <div>📊 Tamaño: {file_size:.1f} KB</div>
                <div>🔤 Tipo: {uploaded_file.type}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Botón de análisis con estado
            if st.sidebar.button("🚀 **Iniciar Análisis**", use_container_width=True, type="primary"):
                return self._process_uploaded_file(uploaded_file)
        
        return st.session_state.get('analysis_data')
    
    def _process_uploaded_file(self, uploaded_file):
        """Procesa archivo subido con mejor manejo de estado"""
        try:
            # Inicializar progreso
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            # Etapa 1: Limpieza de datos
            status_text.text("🔄 Limpiando datos...")
            st.session_state.processing_stage = "Limpieza"
            cleaned_messages = self.data_cleaner.process_uploaded_file(uploaded_file)
            progress_bar.progress(25)
            
            if not cleaned_messages:
                st.sidebar.error("❌ No se pudieron procesar los mensajes del archivo")
                return None
            
            # Etapa 2: Análisis de sentimiento
            status_text.text("😊 Analizando sentimiento...")
            st.session_state.processing_stage = "Sentimiento"
            enhanced_messages = self.data_cleaner.enhance_messages_with_sentiment(cleaned_messages)
            progress_bar.progress(50)
            
            # Etapa 3: Estadísticas de limpieza
            status_text.text("📊 Calculando estadísticas...")
            st.session_state.processing_stage = "Estadísticas"
            cleaning_stats = self.data_cleaner.get_cleaning_stats(enhanced_messages)
            st.session_state.cleaning_stats = cleaning_stats
            progress_bar.progress(75)
            
            # Etapa 4: Análisis completo
            status_text.text("🔍 Realizando análisis avanzado...")
            st.session_state.processing_stage = "Análisis"
            analyzed_data = self.analyzer.comprehensive_analysis(enhanced_messages)
            progress_bar.progress(100)
            
            # Finalizar
            status_text.text("✅ ¡Análisis completado!")
            st.session_state.analysis_data = analyzed_data
            st.session_state.data_source = "file"
            
            # Mostrar estadísticas de limpieza
            self._show_enhanced_cleaning_stats(cleaning_stats)
            
            # Pequeña pausa para mostrar el 100%
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            st.sidebar.success("🎉 ¡Análisis completado exitosamente!")
            return analyzed_data
            
        except Exception as e:
            st.sidebar.error(f"❌ Error procesando archivo: {str(e)}")
            st.sidebar.error(f"🔧 Detalles: {traceback.format_exc()}")
            return None
    
    def _show_enhanced_cleaning_stats(self, stats):
        """Muestra estadísticas de limpieza mejoradas"""
        if not stats:
            return
            
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Estadísticas de Procesamiento")
        
        # Métricas principales
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.metric("📨 Mensajes", stats.get('total_messages', 0))
            st.metric("👥 Usuarios", stats.get('unique_users', 0))
            st.metric("📅 Días", stats.get('conversation_days', 0))
        
        with col2:
            st.metric("📝 Palabras", f"{stats.get('total_words', 0):,}")
            st.metric("😊 Emojis", stats.get('total_emojis', 0))
            st.metric("📈 Msgs/día", f"{stats.get('messages_per_day', 0):.1f}")
        
        # Información adicional
        if stats.get('date_range', {}).get('start'):
            st.sidebar.info(f"**Período:** {stats['date_range']['start'].split()[0]} a {stats['date_range']['end'].split()[0]}")
    
    def _database_controls(self):
        """Controles mejorados para base de datos"""
        st.sidebar.markdown("### 💾 Análisis Guardados")
        
        saved_analyses = self.data_manager.get_saved_analyses()
        
        if not saved_analyses:
            st.sidebar.info("""
            **No hay análisis guardados**
            
            Después de realizar un análisis, podrás guardarlo aquí para acceder más tarde.
            """)
            return st.session_state.get('analysis_data')
        
        # Selector de análisis con información
        analysis_options = {k: f"{v['name']} ({v['timestamp'][:10]})" for k, v in saved_analyses.items()}
        selected_analysis = st.sidebar.selectbox(
            "Selecciona un análisis:",
            options=list(analysis_options.keys()),
            format_func=lambda x: analysis_options[x]
        )
        
        if selected_analysis:
            analysis_info = saved_analyses[selected_analysis]
            st.sidebar.markdown(f"""
            <div class="info-card">
                <div style='font-weight: bold;'>📊 {analysis_info['name']}</div>
                <div>📅 {analysis_info['timestamp'][:16]}</div>
                <div>💬 {analysis_info['metadata']['total_messages']} mensajes</div>
                <div>👥 {analysis_info['metadata']['total_users']} usuarios</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Botones de acción
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("📂 Cargar", use_container_width=True, key="load_db"):
                with st.spinner("Cargando análisis..."):
                    try:
                        analysis_data = self.data_manager.load_analysis(selected_analysis)
                        st.session_state.analysis_data = analysis_data
                        st.session_state.data_source = "database"
                        st.sidebar.success("✅ Análisis cargado correctamente")
                    except Exception as e:
                        st.sidebar.error(f"❌ Error cargando análisis: {e}")
        
        with col2:
            if st.button("🗑️ Eliminar", use_container_width=True, key="delete_db"):
                try:
                    self.data_manager.delete_analysis(selected_analysis)
                    st.sidebar.success("✅ Análisis eliminado correctamente")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"❌ Error eliminando análisis: {e}")
        
        return st.session_state.get('analysis_data')
    
    def _demo_data_controls(self):
        """Controles para generar datos de demostración"""
        st.sidebar.markdown("### 🎮 Datos de Demostración")
        st.sidebar.info("""
        **Genera datos de ejemplo** para explorar las funcionalidades del sistema sin necesidad de archivos.
        """)
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            num_messages = st.slider("Mensajes", 50, 500, 100, 50)
        
        with col2:
            num_users = st.slider("Usuarios", 3, 10, 5)
        
        scenario = st.sidebar.selectbox(
            "Escenario:",
            ["Equipo de Trabajo", "Grupo Familiar", "Amigos", "Comunidad"],
            help="Selecciona el tipo de conversación a simular"
        )
        
        if st.sidebar.button("🎪 Generar Datos Demo", use_container_width=True, type="primary"):
            with st.spinner(f"Generando conversación de {scenario}..."):
                try:
                    demo_data = self.analyzer.generate_demo_data(num_messages, num_users, scenario)
                    st.session_state.analysis_data = demo_data
                    st.session_state.data_source = "demo"
                    st.sidebar.success(f"✅ {scenario} generado con {num_messages} mensajes")
                except Exception as e:
                    st.sidebar.error(f"❌ Error generando datos demo: {e}")
        
        return st.session_state.get('analysis_data')
    
    def show_main_content(self, analysis_data):
        """Contenido principal mejorado del dashboard"""
        # Header principal con información contextual
        st.markdown('<div class="main-header">🔍 CeTOs Deep Analysis</div>', unsafe_allow_html=True)
        
        # Información del análisis actual
        self._show_analysis_context(analysis_data)
        
        # Métricas rápidas mejoradas
        self._show_enhanced_quick_metrics(analysis_data['metrics'])
        
        # Pestañas mejoradas para organizar el contenido
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 **Análisis General**", 
            "👥 **Red Social**", 
            "💬 **Contenido**", 
            "⏰ **Temporal**", 
            "📈 **Datos**"
        ])
        
        with tab1:
            self._show_enhanced_general_analysis(analysis_data)
        
        with tab2:
            self._show_enhanced_social_analysis(analysis_data)
        
        with tab3:
            self._show_enhanced_content_analysis(analysis_data)
        
        with tab4:
            self._show_enhanced_temporal_analysis(analysis_data)
        
        with tab5:
            self._show_enhanced_data_explorer(analysis_data)
    
    def _show_analysis_context(self, analysis_data):
        """Muestra información contextual del análisis actual"""
        source_info = {
            "file": "📁 Archivo subido",
            "database": "💾 Análisis guardado", 
            "demo": "🎮 Datos demo"
        }
        
        source = st.session_state.get('data_source', 'desconocido')
        message_count = len(analysis_data['messages'])
        user_count = len(analysis_data['user_profiles'])
        
        st.markdown(f"""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <div class="sub-header">
                {source_info.get(source, '🔍 Análisis')} • 
                💬 {message_count} mensajes • 
                👥 {user_count} usuarios
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _show_enhanced_quick_metrics(self, metrics):
        """Muestra métricas rápidas mejoradas"""
        st.markdown("### 📈 Resumen Ejecutivo")
        
        # Primera fila - Métricas principales
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics_config = [
            ("💬 Mensajes", metrics.get('total_messages', 0), f"{metrics.get('total_messages', 0):,}"),
            ("👥 Usuarios", metrics.get('total_users', 0), str(metrics.get('total_users', 0))),
            ("😊 Sentimiento", metrics.get('avg_sentiment', 0), f"{metrics.get('avg_sentiment', 0):.3f}"),
            ("📊 Densidad", metrics.get('message_density', 0), f"{metrics.get('message_density', 0):.1f}/h"),
            ("🎯 Engagement", metrics.get('engagement_rate', 0), f"{metrics.get('engagement_rate', 0):.1f}%")
        ]
        
        for i, (label, value, display_value) in enumerate(metrics_config):
            with [col1, col2, col3, col4, col5][i]:
                self._create_enhanced_metric_card(label, display_value, value)
        
        # Segunda fila - Métricas adicionales si existen
        if any(key in metrics for key in ['conversation_health', 'response_rate', 'messages_per_user']):
            st.markdown("---")
            col6, col7, col8, col9, col10 = st.columns(5)
            
            additional_metrics = [
                ("❤️ Salud", metrics.get('conversation_health', 'N/A'), None),
                ("⚡ Respuesta", metrics.get('response_rate', 0), f"{metrics.get('response_rate', 0):.1f}%"),
                ("📨 Msgs/User", metrics.get('messages_per_user', 0), f"{metrics.get('messages_per_user', 0):.1f}"),
                ("📝 Words/Msg", metrics.get('words_per_message', 0), f"{metrics.get('words_per_message', 0):.1f}"),
                ("🕒 Horas", metrics.get('time_range_hours', 0), f"{metrics.get('time_range_hours', 0):.0f}h")
            ]
            
            for i, (label, value, display_value) in enumerate(additional_metrics):
                with [col6, col7, col8, col9, col10][i]:
                    if display_value:
                        self._create_enhanced_metric_card(label, display_value, value)
                    else:
                        self._create_enhanced_metric_card(label, str(value), value)
    
    def _create_enhanced_metric_card(self, label: str, value: str, raw_value):
        """Crea una tarjeta de métrica visualmente mejorada"""
        # Determinar color basado en el valor
        if isinstance(raw_value, (int, float)):
            if raw_value > 0:
                color = "#10b981"  # verde
            elif raw_value < 0:
                color = "#ef4444"  # rojo
            else:
                color = "#6b7280"  # gris
        else:
            color = "#84B026"  # color principal
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)
    
    def _show_enhanced_general_analysis(self, data):
        """Análisis general mejorado"""
        st.markdown("### 📊 Vista General de la Conversación")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Análisis de sentimiento
            self.viz_engine.display_sentiment_analysis(data['messages'], data['metrics'])
            
            # Análisis de usuarios
            self.viz_engine.display_user_analysis(data['user_profiles'], data['metrics'])
        
        with col2:
            # Métricas clave
            self.viz_engine.display_key_metrics(data['metrics'])
            
            # Información adicional
            self._show_conversation_insights(data)
    
    def _show_conversation_insights(self, data):
        """Muestra insights adicionales de la conversación"""
        st.markdown("### 💡 Insights")
        
        metrics = data['metrics']
        user_profiles = data['user_profiles']
        
        # Insights basados en métricas
        insights = []
        
        if metrics.get('avg_sentiment', 0) > 0.3:
            insights.append("✅ **Conversación muy positiva** - Excelente ambiente")
        elif metrics.get('avg_sentiment', 0) > 0:
            insights.append("👍 **Conversación positiva** - Buen ambiente general")
        else:
            insights.append("⚠️ **Conversación neutral/negativa** - Podría necesitar atención")
        
        if metrics.get('engagement_rate', 0) > 70:
            insights.append("🔥 **Alto engagement** - Participación muy activa")
        elif metrics.get('engagement_rate', 0) > 40:
            insights.append("📈 **Engagement moderado** - Buena participación")
        else:
            insights.append("💤 **Engagement bajo** - Podría necesitar más interacción")
        
        # Usuario más activo
        if user_profiles:
            top_user = max(user_profiles.items(), key=lambda x: x[1]['messages_count'])
            insights.append(f"👑 **Líder de conversación**: {top_user[0]} ({top_user[1]['messages_count']} mensajes)")
        
        # Mostrar insights
        for insight in insights:
            st.markdown(f"- {insight}")
    
    def _show_enhanced_social_analysis(self, data):
        """Análisis de red social mejorado"""
        st.markdown("### 👥 Dinámicas Sociales y de Influencia")
        
        tab1, tab2 = st.tabs(["🎯 Influencia", "🕸️ Red"])
        
        with tab1:
            self.viz_engine.display_influence_analysis(data['influence_metrics'], data['user_profiles'])
        
        with tab2:
            self.viz_engine.display_network_analysis(
                data['network_data'], data['user_profiles'], data['influence_metrics']
            )
    
    def _show_enhanced_content_analysis(self, data):
        """Análisis de contenido mejorado"""
        st.markdown("### 💬 Análisis de Contenido")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.viz_engine.display_topic_analysis(data['topic_modeling'])
        
        with col2:
            self._show_enhanced_word_analysis(data)
    
    def _show_enhanced_word_analysis(self, data):
        """Análisis de palabras mejorado"""
        st.markdown("#### 📝 Frecuencia de Palabras")
        
        try:
            all_messages = " ".join([msg.get('message', '') for msg in data['messages']])
            words = [word.lower() for word in all_messages.split() if len(word) > 3]
            
            if words:
                word_freq = pd.Series(words).value_counts().head(15)
                
                # Crear gráfico de barras horizontal
                fig = px.bar(
                    x=word_freq.values,
                    y=word_freq.index,
                    orientation='h',
                    title='Palabras Más Frecuentes',
                    color=word_freq.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    xaxis_title='Frecuencia',
                    yaxis_title='Palabra',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay suficientes palabras significativas para analizar")
                
        except Exception as e:
            st.info("🔧 Análisis de palabras en desarrollo")
    
    def _show_enhanced_temporal_analysis(self, data):
        """Análisis temporal mejorado"""
        st.markdown("### ⏰ Patrones Temporales")
        
        # Análisis temporal principal
        self.viz_engine.display_temporal_analysis(data['messages'])
        
        # Información adicional de patrones
        self._show_temporal_insights(data)
    
    def _show_temporal_insights(self, data):
        """Muestra insights temporales"""
        if 'temporal_patterns' in data:
            patterns = data['temporal_patterns']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'hourly_activity' in patterns:
                    peak_hour = max(patterns['hourly_activity'].items(), key=lambda x: x[1])[0]
                    st.metric("🕐 Hora Pico", f"{peak_hour}:00")
            
            with col2:
                if 'daily_activity' in patterns:
                    peak_day = max(patterns['daily_activity'].items(), key=lambda x: x[1])[0]
                    st.metric("📅 Día Más Activo", peak_day.capitalize())
    
    def _show_enhanced_data_explorer(self, data):
        """Explorador de datos mejorado"""
        st.markdown("### 📈 Exploración de Datos")
        
        # Usar el visualizador mejorado para datos crudos
        self.viz_engine.display_raw_data(data['messages'], data['user_profiles'])
    
    def show_welcome(self):
        """Pantalla de bienvenida completamente mejorada"""
        st.markdown('<div class="main-header">🔍 CeTOs Deep Analysis</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style='text-align: center; margin-bottom: 3rem;'>
            <h3 style='color: #84B026;'>Analítica Avanzada para Conversaciones de WhatsApp</h3>
            <p style='color: #6b7280; font-size: 1.1rem;'>
                Descubre insights profundos en tus conversaciones con inteligencia artificial
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tarjetas de características
        col1, col2, col3 = st.columns(3)
        
        features = [
            ("🧹", "Limpieza Inteligente", "Procesamiento automático de archivos TXT y JSON de WhatsApp"),
            ("📈", "Análisis Avanzado", "Sentimiento, emociones, redes sociales y patrones temporales"),
            ("💾", "Exportación Completa", "Múltiples formatos para compartir y almacenar resultados")
        ]
        
        for i, (icon, title, description) in enumerate(features):
            with [col1, col2, col3][i]:
                st.markdown(f"""
                <div class="info-card" style='text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;'>
                    <div style='font-size: 2.5rem; margin-bottom: 1rem;'>{icon}</div>
                    <h4 style='color: #84B026; margin: 0.5rem 0;'>{title}</h4>
                    <p style='color: #d1d5db; font-size: 0.9rem;'>{description}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Guía de inicio rápido
        st.markdown("""
        ---
        
        ### 🚀 Comienza en 30 Segundos
        
        ```python
        1. 📁 Selecciona "Cargar Archivo" en el sidebar
        2. ⬆️ Sube tu exportación de WhatsApp (.txt)
        3. 🚀 Haz clic en "Iniciar Análisis" 
        4. 📊 Explora los insights generados automáticamente
        ```
        
        ### 🎮 ¿Quieres probar primero?
        
        Selecciona **"Datos Demo"** en el sidebar para generar una conversación de ejemplo 
        y explorar todas las funcionalidades sin necesidad de archivos.
        
        ### 📁 Formatos Soportados
        
        | Formato | Descripción | Recomendado |
        |---------|-------------|-------------|
        | **TXT** | Exportación directa de WhatsApp | ✅ Ideal |
        | **JSON** | Análisis previos guardados | ✅ Compatible |
        | **Demo** | Datos de ejemplo generados | 🎯 Para pruebas |
        
        ### 🛠️ Características Técnicas
        
        - ✅ **Procesamiento multilingüe** (Español/Inglés)
        - ✅ **Análisis de sentimiento avanzado**
        - ✅ **Detección de emociones contextual**
        - ✅ **Visualización de redes sociales**
        - ✅ **Patrones temporales inteligentes**
        - ✅ **Exportación en múltiples formatos**
        - ✅ **Interfaz responsive y moderna**
        """)
    
    def run(self):
        """Ejecutar la aplicación mejorada"""
        try:
            # Sidebar
            analysis_data = self.show_sidebar()
            
            # Main content
            if analysis_data:
                self.show_main_content(analysis_data)
            else:
                self.show_welcome()
                
        except Exception as e:
            st.error("❌ Ocurrió un error inesperado en la aplicación")
            st.error(f"**Detalles técnicos:** {str(e)}")
            st.info("""
            **Solución de problemas:**
            - Verifica que el archivo sea una exportación válida de WhatsApp
            - Intenta recargar la página
            - Si el problema persiste, contacta al soporte técnico
            """)

# Ejecutar la aplicación
if __name__ == "__main__":
    dashboard = WhatsAppDashboard()
    dashboard.run()