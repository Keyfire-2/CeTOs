import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import List, Dict, Any, Optional

class AdvancedDynamicFilters:
    def __init__(self, df, analysis_data: Dict[str, Any] = None):
        self.df = df
        self.analysis_data = analysis_data
        self.filtered_df = df.copy()
        self.active_filters = {}
    
    def setup_advanced_filters(self):
        """Filtros avanzados con múltiples opciones"""
        st.sidebar.header("🎛️ Filtros Avanzados")
        
        # Pestañas para diferentes tipos de filtros
        tab1, tab2, tab3 = st.sidebar.tabs(["👥 Básicos", "📊 Análisis", "💬 Contenido"])
        
        with tab1:
            self._setup_basic_filters()
        
        with tab2:
            self._setup_analysis_filters()
        
        with tab3:
            self._setup_content_filters()
        
        # Mostrar estadísticas de filtros
        self._show_advanced_stats()
        
        return self.filtered_df
    
    def _setup_basic_filters(self):
        """Filtros básicos por usuario, fecha y tipo"""
        # Filtro por usuario con búsqueda
        if 'user' in self.df.columns:
            users = sorted(self.df['user'].dropna().unique().tolist())
            if users:
                selected_users = st.multiselect(
                    "👥 Filtrar por usuarios:",
                    options=users,
                    default=[],
                    help="Selecciona uno o más usuarios"
                )
                
                if selected_users:
                    self.filtered_df = self.filtered_df[self.filtered_df['user'].isin(selected_users)]
                    self.active_filters['usuarios'] = f"{len(selected_users)} usuarios"
        
        # Filtro por fecha mejorado
        self._setup_date_filters()
        
        # Filtro por tipo de mensaje
        if 'message_type' in self.df.columns:
            message_types = sorted(self.df['message_type'].dropna().unique().tolist())
            if message_types:
                selected_types = st.multiselect(
                    "📨 Tipo de mensaje:",
                    options=message_types,
                    default=[],
                    help="Filtrar por tipo de mensaje"
                )
                
                if selected_types:
                    self.filtered_df = self.filtered_df[self.filtered_df['message_type'].isin(selected_types)]
                    self.active_filters['tipos_mensaje'] = f"{len(selected_types)} tipos"
    
    def _setup_date_filters(self):
        """Filtros de fecha avanzados"""
        date_columns = [col for col in self.df.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
        
        if not date_columns:
            return
        
        date_col = date_columns[0]  # Usar la primera columna de fecha encontrada
        
        try:
            # Convertir a datetime
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
            self.filtered_df[date_col] = pd.to_datetime(self.filtered_df[date_col], errors='coerce')
            
            # Eliminar valores NaT
            valid_dates = self.df[date_col].dropna()
            if valid_dates.empty:
                return
            
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            
            # Opciones de rango de fecha predefinidas
            date_options = {
                "Todo el período": (min_date, max_date),
                "Últimos 7 días": (max_date - timedelta(days=7), max_date),
                "Últimos 30 días": (max_date - timedelta(days=30), max_date),
                "Últimos 90 días": (max_date - timedelta(days=90), max_date),
                "Personalizado": None
            }
            
            selected_range = st.selectbox(
                "📅 Rango de fechas:",
                options=list(date_options.keys())
            )
            
            if selected_range == "Personalizado":
                date_range = st.date_input(
                    "Selecciona rango personalizado:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    self.filtered_df = self.filtered_df[
                        (self.filtered_df[date_col].dt.date >= start_date) & 
                        (self.filtered_df[date_col].dt.date <= end_date)
                    ]
                    self.active_filters['fechas'] = f"{start_date} a {end_date}"
            else:
                start_date, end_date = date_options[selected_range]
                if selected_range != "Todo el período":
                    self.filtered_df = self.filtered_df[
                        (self.filtered_df[date_col].dt.date >= start_date) & 
                        (self.filtered_df[date_col].dt.date <= end_date)
                    ]
                    self.active_filters['fechas'] = selected_range
            
            # Filtro adicional por día de la semana
            if 'day_of_week' in self.df.columns or date_col:
                days_of_week = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
                selected_days = st.multiselect(
                    "📆 Días de la semana:",
                    options=days_of_week,
                    default=[]
                )
                
                if selected_days:
                    if 'day_of_week' in self.df.columns:
                        self.filtered_df = self.filtered_df[self.filtered_df['day_of_week'].isin(selected_days)]
                    else:
                        # Calcular día de la semana desde la fecha
                        self.filtered_df = self.filtered_df[
                            self.filtered_df[date_col].dt.day_name().str.lower().isin(
                                [day.capitalize() for day in selected_days]
                            )
                        ]
                    self.active_filters['dias_semana'] = f"{len(selected_days)} días"
                    
        except Exception as e:
            st.sidebar.warning(f"⚠️ No se pudieron aplicar filtros de fecha: {e}")
    
    def _setup_analysis_filters(self):
        """Filtros basados en análisis de sentimiento y métricas"""
        if not self.analysis_data:
            st.info("ℹ️ Carga un análisis para habilitar estos filtros")
            return
        
        # Filtro por sentimiento
        if any('sentiment' in str(col).lower() for col in self.df.columns):
            sentiment_col = next((col for col in self.df.columns if 'sentiment' in str(col).lower()), None)
            
            if sentiment_col:
                min_sentiment = float(self.df[sentiment_col].min())
                max_sentiment = float(self.df[sentiment_col].max())
                
                sentiment_range = st.slider(
                    "😊 Rango de sentimiento:",
                    min_value=min_sentiment,
                    max_value=max_sentiment,
                    value=(min_sentiment, max_sentiment),
                    step=0.1,
                    help="Filtrar mensajes por puntuación de sentimiento"
                )
                
                if sentiment_range != (min_sentiment, max_sentiment):
                    self.filtered_df = self.filtered_df[
                        (self.filtered_df[sentiment_col] >= sentiment_range[0]) & 
                        (self.filtered_df[sentiment_col] <= sentiment_range[1])
                    ]
                    self.active_filters['sentimiento'] = f"{sentiment_range[0]:.2f} a {sentiment_range[1]:.2f}"
        
        # Filtro por longitud del mensaje
        if 'words_count' in self.df.columns:
            min_words = int(self.df['words_count'].min())
            max_words = int(self.df['words_count'].max())
            
            word_range = st.slider(
                "📝 Rango de palabras por mensaje:",
                min_value=min_words,
                max_value=min(max_words, 1000),  # Limitar máximo para rendimiento
                value=(min_words, min(max_words, 100)),
                help="Filtrar por número de palabras"
            )
            
            if word_range != (min_words, min(max_words, 100)):
                self.filtered_df = self.filtered_df[
                    (self.filtered_df['words_count'] >= word_range[0]) & 
                    (self.filtered_df['words_count'] <= word_range[1])
                ]
                self.active_filters['palabras'] = f"{word_range[0]}-{word_range[1]} palabras"
        
        # Filtro por hora del día
        date_columns = [col for col in self.df.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
        if date_columns:
            date_col = date_columns[0]
            try:
                hours = list(range(24))
                selected_hours = st.multiselect(
                    "🕐 Horas del día:",
                    options=hours,
                    default=[],
                    format_func=lambda x: f"{x:02d}:00"
                )
                
                if selected_hours:
                    self.filtered_df = self.filtered_df[
                        self.filtered_df[date_col].dt.hour.isin(selected_hours)
                    ]
                    self.active_filters['horas'] = f"{len(selected_hours)} horas"
            except:
                pass
    
    def _setup_content_filters(self):
        """Filtros avanzados de contenido"""
        # Búsqueda de texto avanzada
        col1, col2 = st.columns(2)
        
        with col1:
            keyword = st.text_input(
                "🔍 Palabra clave:",
                help="Buscar en el contenido de los mensajes"
            )
        
        with col2:
            search_mode = st.selectbox(
                "Modo de búsqueda:",
                options=["Contiene", "Comienza con", "Termina con", "Expresión regular"]
            )
        
        if keyword:
            if search_mode == "Contiene":
                self.filtered_df = self.filtered_df[
                    self.filtered_df['message'].astype(str).str.contains(keyword, case=False, na=False)
                ]
            elif search_mode == "Comienza con":
                self.filtered_df = self.filtered_df[
                    self.filtered_df['message'].astype(str).str.startswith(keyword, na=False)
                ]
            elif search_mode == "Termina con":
                self.filtered_df = self.filtered_df[
                    self.filtered_df['message'].astype(str).str.endswith(keyword, na=False)
                ]
            elif search_mode == "Expresión regular":
                try:
                    self.filtered_df = self.filtered_df[
                        self.filtered_df['message'].astype(str).str.contains(keyword, case=False, na=False, regex=True)
                    ]
                except:
                    st.error("❌ Expresión regular inválida")
            
            self.active_filters['palabra_clave'] = f"'{keyword}' ({search_mode})"
        
        # Filtro por presencia de emojis
        if 'emojis' in self.df.columns:
            emoji_filter = st.selectbox(
                "😊 Contenido con emojis:",
                options=["Todos", "Con emojis", "Sin emojis"]
            )
            
            if emoji_filter == "Con emojis":
                self.filtered_df = self.filtered_df[
                    self.filtered_df['emojis'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
                ]
                self.active_filters['emojis'] = "Solo con emojis"
            elif emoji_filter == "Sin emojis":
                self.filtered_df = self.filtered_df[
                    self.filtered_df['emojis'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True)
                ]
                self.active_filters['emojis'] = "Sin emojis"
        
        # Filtro por URLs
        if 'urls' in self.df.columns:
            url_filter = st.selectbox(
                "🌐 Contenido con enlaces:",
                options=["Todos", "Con enlaces", "Sin enlaces"]
            )
            
            if url_filter == "Con enlaces":
                self.filtered_df = self.filtered_df[
                    self.filtered_df['urls'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
                ]
                self.active_filters['enlaces'] = "Solo con enlaces"
            elif url_filter == "Sin enlaces":
                self.filtered_df = self.filtered_df[
                    self.filtered_df['urls'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True)
                ]
                self.active_filters['enlaces'] = "Sin enlaces"
    
    def _show_advanced_stats(self):
        """Muestra estadísticas avanzadas de los filtros"""
        if len(self.df) == 0:
            return
        
        original_count = len(self.df)
        filtered_count = len(self.filtered_df)
        filtered_percentage = (filtered_count / original_count * 100) if original_count > 0 else 0
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Estadísticas de Filtros")
        
        # Métricas principales
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.metric("Mensajes totales", f"{original_count:,}")
        
        with col2:
            st.metric("Mensajes filtrados", f"{filtered_count:,}", 
                     delta=f"{filtered_percentage:.1f}%")
        
        # Barra de progreso para visualizar el filtrado
        if original_count > 0:
            progress = filtered_count / original_count
            st.sidebar.progress(progress)
            st.sidebar.caption(f"Mostrando {filtered_count} de {original_count} mensajes ({filtered_percentage:.1f}%)")
        
        # Filtros activos
        if self.active_filters:
            st.sidebar.markdown("#### 🎛️ Filtros Activos")
            for filter_name, filter_value in self.active_filters.items():
                st.sidebar.write(f"• **{filter_name}**: {filter_value}")
            
            # Botón para limpiar filtros
            if st.sidebar.button("🧹 Limpiar todos los filtros", use_container_width=True):
                self.filtered_df = self.df.copy()
                self.active_filters = {}
                st.rerun()
        else:
            st.sidebar.info("ℹ️ No hay filtros activos")
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """Retorna un resumen de los filtros aplicados"""
        return {
            'original_count': len(self.df),
            'filtered_count': len(self.filtered_df),
            'filtered_percentage': (len(self.filtered_df) / len(self.df) * 100) if len(self.df) > 0 else 0,
            'active_filters': self.active_filters,
            'remaining_users': self.filtered_df['user'].nunique() if 'user' in self.filtered_df.columns else 0
        }
    
    def export_filtered_data(self, format: str = 'csv') -> str:
        """Exporta los datos filtrados"""
        if format == 'csv':
            return self.filtered_df.to_csv(index=False)
        elif format == 'json':
            return self.filtered_df.to_json(orient='records', indent=2)
        else:
            raise ValueError(f"Formato no soportado: {format}")

# Clase de compatibilidad con versión anterior
class DynamicFilters(AdvancedDynamicFilters):
    """Clase de compatibilidad con la versión anterior"""
    def setup_filters(self):
        """Método de compatibilidad con la versión anterior"""
        st.sidebar.header("🔧 Filtros")
        
        # Filtro por usuario
        if 'user' in self.df.columns:
            users = ['Todos'] + sorted(self.df['user'].dropna().unique().tolist())
            selected_user = st.sidebar.selectbox("Usuario", users)
            
            if selected_user != 'Todos':
                self.filtered_df = self.filtered_df[self.filtered_df['user'] == selected_user]
                self.active_filters['usuario'] = selected_user
        
        # Filtro por palabra clave
        keyword = st.sidebar.text_input("Palabra clave")
        if keyword:
            self.filtered_df = self.filtered_df[
                self.filtered_df['message'].astype(str).str.contains(keyword, case=False, na=False)
            ]
            self.active_filters['palabra_clave'] = keyword
        
        # Filtro por fecha
        if 'date' in self.df.columns:
            try:
                dates = pd.to_datetime(self.df['date']).dt.date
                min_date = dates.min()
                max_date = dates.max()
                
                if min_date != max_date:
                    date_range = st.sidebar.date_input(
                        "Rango de Fechas",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
                    
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        self.filtered_df = self.filtered_df[
                            (pd.to_datetime(self.filtered_df['date']).dt.date >= start_date) &
                            (pd.to_datetime(self.filtered_df['date']).dt.date <= end_date)
                        ]
                        self.active_filters['fechas'] = f"{start_date} a {end_date}"
            except:
                pass
        
        self._show_advanced_stats()
        return self.filtered_df
    
    def show_stats(self):
        """Método de compatibilidad con la versión anterior"""
        original_count = len(self.df)
        filtered_count = len(self.filtered_df)
        
        st.sidebar.markdown("---")
        st.sidebar.metric("Mensajes totales", original_count)
        st.sidebar.metric("Mensajes filtrados", filtered_count)

# Función de utilidad para uso rápido
def setup_dynamic_filters(df, analysis_data=None, advanced=True):
    """Función de conveniencia para configurar filtros rápidamente"""
    if advanced:
        filter_engine = AdvancedDynamicFilters(df, analysis_data)
        return filter_engine.setup_advanced_filters()
    else:
        filter_engine = DynamicFilters(df, analysis_data)
        return filter_engine.setup_filters()