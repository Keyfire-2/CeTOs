import streamlit as st
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class AdvancedThemeManager:
    def __init__(self):
        self.current_theme = "Cyber Deep"
        self.themes = self._initialize_themes()
        self.user_preferences = self._load_user_preferences()
    
    def _initialize_themes(self) -> Dict[str, Dict[str, str]]:
        """Inicializa todos los temas disponibles"""
        return {
            "Cyber Deep": {
                "primary": "#84B026",
                "secondary": "#217373", 
                "accent1": "#173540",
                "accent2": "#161F30",
                "dark": "#0D0D0D",
                "background": "#0A0A0A",
                "text": "#FFFFFF",
                "success": "#10b981",
                "warning": "#f59e0b",
                "error": "#ef4444",
                "info": "#3b82f6",
                "card_bg": "#1a1a1a",
                "border": "#333333",
                "gradient_start": "#84B026",
                "gradient_end": "#217373",
                "shadow": "rgba(132, 176, 38, 0.3)"
            },
            "Ocean Blue": {
                "primary": "#0066cc",
                "secondary": "#0099ff", 
                "accent1": "#66ccff",
                "accent2": "#99ddff",
                "dark": "#003366",
                "background": "#f0f8ff",
                "text": "#003366",
                "success": "#00cc99",
                "warning": "#ffaa00",
                "error": "#ff6666",
                "info": "#3399ff",
                "card_bg": "#ffffff",
                "border": "#cce5ff",
                "gradient_start": "#0066cc",
                "gradient_end": "#0099ff",
                "shadow": "rgba(0, 102, 204, 0.2)"
            },
            "Sunset Purple": {
                "primary": "#8B5FBF",
                "secondary": "#6A3093", 
                "accent1": "#A45DE2",
                "accent2": "#C78BFF",
                "dark": "#4A235A",
                "background": "#FAF5FF",
                "text": "#4A235A",
                "success": "#9FDFA5",
                "warning": "#FFD966",
                "error": "#FF8A8A",
                "info": "#8FAADC",
                "card_bg": "#FFFFFF",
                "border": "#E8DAEF",
                "gradient_start": "#8B5FBF",
                "gradient_end": "#6A3093",
                "shadow": "rgba(139, 95, 191, 0.3)"
            },
            "Forest Green": {
                "primary": "#2E8B57",
                "secondary": "#3CB371", 
                "accent1": "#228B22",
                "accent2": "#32CD32",
                "dark": "#006400",
                "background": "#F5FFF5",
                "text": "#006400",
                "success": "#90EE90",
                "warning": "#FFD700",
                "error": "#FF6B6B",
                "info": "#87CEEB",
                "card_bg": "#FFFFFF",
                "border": "#D4EDDA",
                "gradient_start": "#2E8B57",
                "gradient_end": "#3CB371",
                "shadow": "rgba(46, 139, 87, 0.3)"
            },
            "Dark Matrix": {
                "primary": "#00FF41",
                "secondary": "#008F11", 
                "accent1": "#003B00",
                "accent2": "#005F00",
                "dark": "#001100",
                "background": "#000000",
                "text": "#00FF41",
                "success": "#00FF41",
                "warning": "#FFFF00",
                "error": "#FF0033",
                "info": "#00FFFF",
                "card_bg": "#001100",
                "border": "#003300",
                "gradient_start": "#00FF41",
                "gradient_end": "#008F11",
                "shadow": "rgba(0, 255, 65, 0.3)"
            },
            "Warm Sunset": {
                "primary": "#FF6B35",
                "secondary": "#FF8E53", 
                "accent1": "#FFB399",
                "accent2": "#FFD6CC",
                "dark": "#CC5500",
                "background": "#FFF5F0",
                "text": "#663300",
                "success": "#66BB6A",
                "warning": "#FFA726",
                "error": "#EF5350",
                "info": "#42A5F5",
                "card_bg": "#FFFFFF",
                "border": "#FFE0B2",
                "gradient_start": "#FF6B35",
                "gradient_end": "#FF8E53",
                "shadow": "rgba(255, 107, 53, 0.3)"
            }
        }
    
    def _load_user_preferences(self) -> Dict[str, Any]:
        """Carga las preferencias del usuario desde session state"""
        if 'theme_preferences' not in st.session_state:
            st.session_state.theme_preferences = {
                'current_theme': 'Cyber Deep',
                'font_size': 'medium',
                'animation_enabled': True,
                'dark_mode': False,
                'last_updated': datetime.now().isoformat()
            }
        return st.session_state.theme_preferences
    
    def _save_user_preferences(self):
        """Guarda las preferencias del usuario en session state"""
        st.session_state.theme_preferences = self.user_preferences
    
    def apply_advanced_theme(self, theme_name: str = None):
        """Aplica el tema avanzado con CSS personalizado"""
        if theme_name:
            self.current_theme = theme_name
            self.user_preferences['current_theme'] = theme_name
            self._save_user_preferences()
        
        theme = self.themes[self.current_theme]
        
        custom_css = self._generate_advanced_css(theme)
        st.markdown(custom_css, unsafe_allow_html=True)
    
    def _generate_advanced_css(self, theme: Dict[str, str]) -> str:
        """Genera CSS avanzado basado en el tema seleccionado"""
        return f"""
        <style>
            /* Configuración base */
            .main {{
                background-color: {theme['background']};
                color: {theme['text']};
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            
            /* Header principal */
            .main-header {{
                font-size: 3.5rem;
                background: linear-gradient(135deg, {theme['gradient_start']}, {theme['gradient_end']});
                -webkit-background-clip: text;
                -moz-background-clip: text;
                background-clip: text;
                -webkit-text-fill-color: transparent;
                -moz-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 1rem;
                font-weight: 800;
                text-shadow: 0 2px 10px {theme['shadow']};
            }}
            
            /* Botones */
            .stButton>button {{
                background: linear-gradient(135deg, {theme['primary']}, {theme['secondary']});
                color: white;
                border: none;
                border-radius: 12px;
                padding: 0.75rem 1.5rem;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px {theme['shadow']};
                border: 2px solid transparent;
            }}
            
            .stButton>button:hover {{
                background: linear-gradient(135deg, {theme['secondary']}, {theme['primary']});
                transform: translateY(-2px);
                box-shadow: 0 6px 20px {theme['shadow']};
                border-color: {theme['accent1']};
            }}
            
            /* Sidebar */
            .css-1d391kg {{
                background: linear-gradient(180deg, {theme['accent1']}, {theme['accent2']});
                border-right: 3px solid {theme['primary']};
            }}
            
            .sidebar .sidebar-content {{
                background: transparent !important;
            }}
            
            /* Tarjetas de métricas */
            .metric-card {{
                background: linear-gradient(135deg, {theme['card_bg']}, {theme['accent1']});
                padding: 1.5rem;
                border-radius: 15px;
                color: {theme['text']};
                text-align: center;
                margin: 0.5rem;
                box-shadow: 0 8px 25px {theme['shadow']};
                border: 2px solid {theme['border']};
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }}
            
            .metric-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
                transition: left 0.5s;
            }}
            
            .metric-card:hover::before {{
                left: 100%;
            }}
            
            .metric-card:hover {{
                transform: translateY(-5px);
                border-color: {theme['primary']};
                box-shadow: 0 12px 35px {theme['shadow']};
            }}
            
            .metric-value {{
                font-size: 2.2rem;
                font-weight: bold;
                margin-top: 0.5rem;
                color: {theme['primary']};
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }}
            
            .metric-label {{
                font-size: 0.9rem;
                opacity: 0.9;
                margin-bottom: 0.5rem;
                color: {theme['text']};
            }}
            
            /* Tarjetas de roles de usuario */
            .user-role-card {{
                background: linear-gradient(135deg, {theme['card_bg']}, {theme['accent2']});
                padding: 1.5rem;
                border-radius: 12px;
                margin: 0.5rem 0;
                border: 2px solid {theme['border']};
                box-shadow: 0 4px 15px {theme['shadow']};
                transition: all 0.3s ease;
            }}
            
            .user-role-card:hover {{
                transform: translateX(5px);
                border-color: {theme['primary']};
                box-shadow: 0 6px 20px {theme['shadow']};
            }}
            
            /* Pestañas */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
                background: {theme['accent1']};
                border-radius: 12px;
                padding: 8px;
                margin-bottom: 1rem;
                border: 2px solid {theme['border']};
            }}
            
            .stTabs [data-baseweb="tab"] {{
                background: {theme['card_bg']};
                color: {theme['text']};
                border-radius: 10px;
                padding: 1rem 1.5rem;
                font-weight: 600;
                transition: all 0.3s ease;
                border: 2px solid {theme['border']};
                flex: 1;
                text-align: center;
            }}
            
            .stTabs [aria-selected="true"] {{
                background: linear-gradient(135deg, {theme['primary']}, {theme['secondary']});
                color: white;
                border-color: {theme['primary']};
                transform: scale(1.02);
                box-shadow: 0 4px 12px {theme['shadow']};
            }}
            
            /* Progress bars */
            .stProgress > div > div > div > div {{
                background: linear-gradient(90deg, {theme['primary']}, {theme['secondary']});
            }}
            
            /* Expanders */
            .streamlit-expanderHeader {{
                background: linear-gradient(135deg, {theme['card_bg']}, {theme['accent1']});
                color: {theme['text']};
                border-radius: 8px;
                border: 2px solid {theme['border']};
                font-weight: 600;
            }}
            
            .streamlit-expanderHeader:hover {{
                background: linear-gradient(135deg, {theme['accent1']}, {theme['accent2']});
                border-color: {theme['primary']};
            }}
            
            /* Input fields */
            .stTextInput>div>div>input, 
            .stSelectbox>div>div>select,
            .stTextArea>div>div>textarea {{
                background: {theme['card_bg']};
                color: {theme['text']};
                border: 2px solid {theme['border']};
                border-radius: 8px;
            }}
            
            .stTextInput>div>div>input:focus, 
            .stSelectbox>div>div>select:focus,
            .stTextArea>div>div>textarea:focus {{
                border-color: {theme['primary']};
                box-shadow: 0 0 0 2px {theme['shadow']};
            }}
            
            /* Dataframes y tablas */
            .dataframe {{
                background: {theme['card_bg']};
                color: {theme['text']};
                border: 2px solid {theme['border']};
                border-radius: 8px;
            }}
            
            /* Mensajes de información */
            .stAlert {{
                border-radius: 8px;
                border: 2px solid {theme['border']};
            }}
            
            /* Spinners */
            .stSpinner > div {{
                border-color: {theme['primary']} transparent transparent transparent;
            }}
            
            /* Secciones y headers */
            .section-header {{
                font-size: 1.8rem;
                color: {theme['primary']};
                margin: 2rem 0 1rem 0;
                border-bottom: 3px solid {theme['secondary']};
                padding-bottom: 0.5rem;
                font-weight: 600;
            }}
            
            .sub-header {{
                font-size: 1.2rem;
                color: {theme['secondary']};
                margin-bottom: 1rem;
                opacity: 0.9;
            }}
            
            /* Tarjetas de información */
            .info-card {{
                background: linear-gradient(135deg, {theme['card_bg']}, {theme['accent1']});
                padding: 1.5rem;
                border-radius: 12px;
                border: 2px solid {theme['border']};
                margin: 1rem 0;
                box-shadow: 0 4px 15px {theme['shadow']};
            }}
            
            /* Estados de éxito, warning y error */
            .stSuccess {{
                background: linear-gradient(135deg, {theme['success']}, #0f9d58) !important;
                color: white !important;
            }}
            
            .stWarning {{
                background: linear-gradient(135deg, {theme['warning']}, #f57c00) !important;
                color: white !important;
            }}
            
            .stError {{
                background: linear-gradient(135deg, {theme['error']}, #c62828) !important;
                color: white !important;
            }}
            
            .stInfo {{
                background: linear-gradient(135deg, {theme['info']}, #1565c0) !important;
                color: white !important;
            }}
            
            /* Scrollbar personalizada */
            ::-webkit-scrollbar {{
                width: 8px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: {theme['accent1']};
                border-radius: 4px;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: linear-gradient(135deg, {theme['primary']}, {theme['secondary']});
                border-radius: 4px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: linear-gradient(135deg, {theme['secondary']}, {theme['primary']});
            }}
        </style>
        """
    
    def show_theme_selector(self):
        """Muestra el selector de temas en el sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎨 Personalización")
        
        # Selector de tema
        theme_names = list(self.themes.keys())
        selected_theme = st.sidebar.selectbox(
            "Selecciona un tema:",
            options=theme_names,
            index=theme_names.index(self.user_preferences.get('current_theme', 'Cyber Deep')),
            help="Elige un tema visual para la aplicación"
        )
        
        # Aplicar tema seleccionado
        if selected_theme != self.current_theme:
            self.apply_advanced_theme(selected_theme)
            st.sidebar.success(f"✅ Tema cambiado a: {selected_theme}")
        
        # Previsualización de colores del tema actual
        st.sidebar.markdown("#### 🎨 Paleta del Tema")
        current_theme = self.themes[selected_theme]
        
        colors_to_show = ['primary', 'secondary', 'accent1', 'accent2']
        for color_name in colors_to_show:
            color_value = current_theme[color_name]
            st.sidebar.markdown(
                f"<div style='display: flex; align-items: center; margin: 5px 0;'>"
                f"<div style='width: 20px; height: 20px; background: {color_value}; "
                f"border-radius: 4px; margin-right: 10px; border: 1px solid {current_theme['border']};'></div>"
                f"<span style='font-size: 0.8rem;'>{color_name}: {color_value}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Configuraciones adicionales
        st.sidebar.markdown("#### ⚙️ Configuración")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            font_size = st.selectbox(
                "Tamaño de texto:",
                ["Pequeño", "Mediano", "Grande"],
                index=1
            )
        
        with col2:
            animations = st.checkbox("Animaciones", value=True)
        
        # Actualizar preferencias
        self.user_preferences.update({
            'font_size': font_size,
            'animation_enabled': animations,
            'last_updated': datetime.now().isoformat()
        })
        self._save_user_preferences()
        
        # Botón para resetear a tema por defecto
        if st.sidebar.button("🔄 Resetear a Default", use_container_width=True):
            self.apply_advanced_theme("Cyber Deep")
            st.rerun()
    
    def get_theme_colors(self, theme_name: str = None) -> Dict[str, str]:
        """Retorna los colores del tema especificado"""
        if theme_name is None:
            theme_name = self.current_theme
        return self.themes.get(theme_name, self.themes["Cyber Deep"])
    
    def create_custom_theme(self, theme_data: Dict[str, str], theme_name: str):
        """Crea un tema personalizado"""
        required_keys = ['primary', 'secondary', 'accent1', 'accent2', 'background', 'text']
        
        if all(key in theme_data for key in required_keys):
            self.themes[theme_name] = theme_data
            st.success(f"🎨 Tema personalizado '{theme_name}' creado exitosamente!")
        else:
            st.error("❌ El tema personalizado debe incluir todos los colores requeridos")
    
    def export_theme(self, theme_name: str) -> str:
        """Exporta un tema como JSON"""
        if theme_name in self.themes:
            return json.dumps(self.themes[theme_name], indent=2)
        else:
            raise ValueError(f"Tema '{theme_name}' no encontrado")

# Clase de compatibilidad con versión anterior
class ThemeManager(AdvancedThemeManager):
    """Clase de compatibilidad con la versión anterior"""
    def apply_custom_css(self):
        """Método de compatibilidad con la versión anterior"""
        self.apply_advanced_theme("Cyber Deep")
    
    def get_theme_selector(self):
        """Método de compatibilidad con la versión anterior"""
        return "Cyber Deep"

# Función de utilidad para uso rápido
def setup_theme(theme_name: str = "Cyber Deep", show_selector: bool = True):
    """Función de conveniencia para configurar el tema rápidamente"""
    theme_manager = AdvancedThemeManager()
    theme_manager.apply_advanced_theme(theme_name)
    
    if show_selector:
        theme_manager.show_theme_selector()
    
    return theme_manager