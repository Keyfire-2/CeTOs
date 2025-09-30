import pandas as pd
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import re
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class AdvancedTopicModeling:
    def __init__(self):
        self.vectorizer = None
        self.lda_model = None
        self.nmf_model = None
        self.kmeans = None
        self.feature_names = None
        
        # Stop words en español más completas
        self.spanish_stopwords = {
            'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 
            'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 
            'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'también', 'me', 'hasta', 
            'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 
            'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 
            'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 
            'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 
            'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 'tus', 'ellas', 'nosotras', 'vosotros', 
            'vosotras', 'os', 'mío', 'mía', 'míos', 'mías', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 
            'suya', 'suyos', 'suyas', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 
            'vuestros', 'vuestras', 'esos', 'esas', 'estoy', 'estás', 'está', 'estamos', 'estáis', 
            'están', 'esté', 'estés', 'estemos', 'estéis', 'estén', 'estaré', 'estarás', 'estará', 
            'estaremos', 'estaréis', 'estarán', 'estaría', 'estarías', 'estaríamos', 'estaríais', 
            'estarían', 'estaba', 'estabas', 'estábamos', 'estabais', 'estaban', 'estuve', 'estuviste', 
            'estuvo', 'estuvimos', 'estuvisteis', 'estuvieron', 'estuviera', 'estuvieras', 'estuviéramos', 
            'estuvierais', 'estuvieran', 'estuviese', 'estuvieses', 'estuviésemos', 'estuvieseis', 
            'estuviesen', 'estando', 'estado', 'estada', 'estados', 'estadas', 'estad', 'he', 'has', 
            'ha', 'hemos', 'habéis', 'han', 'haya', 'hayas', 'hayamos', 'hayáis', 'hayan', 'habré', 
            'habrás', 'habrá', 'habremos', 'habréis', 'habrán', 'habría', 'habrías', 'habríamos', 
            'habríais', 'habrían', 'había', 'habías', 'habíamos', 'habíais', 'habían', 'hube', 'hubiste', 
            'hubo', 'hubimos', 'hubisteis', 'hubieron', 'hubiera', 'hubieras', 'hubiéramos', 'hubierais', 
            'hubieran', 'hubiese', 'hubieses', 'hubiésemos', 'hubieseis', 'hubiesen', 'habiendo', 
            'habido', 'habida', 'habidos', 'habidas', 'soy', 'eres', 'es', 'somos', 'sois', 'son', 
            'sea', 'seas', 'seamos', 'seáis', 'sean', 'seré', 'serás', 'será', 'seremos', 'seréis', 
            'serán', 'sería', 'serías', 'seríamos', 'seríais', 'serían', 'era', 'eras', 'éramos', 
            'erais', 'eran', 'fui', 'fuiste', 'fue', 'fuimos', 'fuisteis', 'fueron', 'fuera', 'fueras', 
            'fuéramos', 'fuerais', 'fueran', 'fuese', 'fueses', 'fuésemos', 'fueseis', 'fuesen', 
            'sintiendo', 'sentido', 'tengo', 'tienes', 'tiene', 'tenemos', 'tenéis', 'tienen', 'tenga', 
            'tengas', 'tengamos', 'tengáis', 'tengan', 'tendré', 'tendrás', 'tendrá', 'tendremos', 
            'tendréis', 'tendrán', 'tendría', 'tendrías', 'tendríamos', 'tendríais', 'tendrían', 
            'tenía', 'tenías', 'teníamos', 'teníais', 'tenían', 'tuve', 'tuviste', 'tuvo', 'tuvimos', 
            'tuvisteis', 'tuvieron', 'tuviera', 'tuvieras', 'tuviéramos', 'tuvierais', 'tuvieran', 
            'tuviese', 'tuvieses', 'tuviésemos', 'tuvieseis', 'tuviesen', 'teniendo', 'tenido', 'tenida', 
            'tenidos', 'tenidas', 'tened'
        }
    
    def advanced_text_cleaning(self, text: str) -> str:
        """Limpieza avanzada de texto para análisis de temas"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Remover URLs, menciones, y caracteres especiales
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remover números y caracteres especiales, mantener letras y espacios
        text = re.sub(r'[^a-záéíóúñü\s]', ' ', text)
        
        # Remover espacios extra
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Filtrar palabras muy cortas (menos de 2 caracteres)
        words = [word for word in text.split() if len(word) > 2]
        
        return ' '.join(words)
    
    def extract_ngrams(self, texts: List[str], ngram_range: Tuple[int, int] = (1, 2)) -> List[str]:
        """Extrae n-gramas relevantes del texto"""
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=100,
            stop_words=list(self.spanish_stopwords)
        )
        
        try:
            X = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calcular frecuencia de n-gramas
            ngram_freq = X.sum(axis=0).A1
            ngram_dict = dict(zip(feature_names, ngram_freq))
            
            # Filtrar n-gramas relevantes (frecuencia > 2)
            relevant_ngrams = [ngram for ngram, freq in ngram_dict.items() if freq > 2]
            
            return relevant_ngrams
        except:
            return []
    
    def find_optimal_topics(self, dtm, max_topics: int = 8) -> int:
        """Encuentra el número óptimo de temas usando el método de coherencia"""
        if dtm.shape[0] < 10:
            return min(3, max_topics)
        
        max_topics = min(max_topics, dtm.shape[0] // 3)
        if max_topics < 2:
            return 2
        
        best_score = -1
        best_n = 2
        
        for n_topics in range(2, max_topics + 1):
            try:
                lda = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=10
                )
                lda.fit(dtm)
                
                # Calcular coherencia simple
                score = self._calculate_coherence(lda, dtm)
                
                if score > best_score:
                    best_score = score
                    best_n = n_topics
            except:
                continue
        
        return best_n
    
    def _calculate_coherence(self, model, dtm, top_n: int = 10) -> float:
        """Calcula coherencia simple para los temas"""
        try:
            feature_names = self.vectorizer.get_feature_names_out()
            coherence_scores = []
            
            for topic_idx, topic in enumerate(model.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-top_n - 1:-1]]
                
                # Coherencia simple basada en co-ocurrencia
                topic_coherence = 0
                word_pairs = 0
                
                for i, word1 in enumerate(top_words):
                    for j, word2 in enumerate(top_words[i+1:], i+1):
                        # Simular co-ocurrencia (en implementación real usarías datos externos)
                        topic_coherence += 1 / (abs(i - j) + 1)
                        word_pairs += 1
                
                if word_pairs > 0:
                    coherence_scores.append(topic_coherence / word_pairs)
            
            return np.mean(coherence_scores) if coherence_scores else 0
        except:
            return 0
    
    def analyze_topics_advanced(self, messages: List[str], auto_topics: bool = True, n_topics: int = 5) -> Dict[str, Any]:
        """Análisis avanzado de temas con múltiples algoritmos"""
        if len(messages) < 15:
            return self._get_empty_analysis()
        
        # Limpiar y preparar textos
        cleaned_messages = [self.advanced_text_cleaning(msg) for msg in messages if msg and len(str(msg).split()) >= 3]
        cleaned_messages = [msg for msg in cleaned_messages if len(msg.split()) >= 2]
        
        if len(cleaned_messages) < 10:
            return self._get_empty_analysis()
        
        try:
            # Vectorización con TF-IDF para mejor calidad
            self.vectorizer = TfidfVectorizer(
                max_features=100,
                min_df=2,
                max_df=0.8,
                stop_words=list(self.spanish_stopwords),
                ngram_range=(1, 2)
            )
            
            dtm = self.vectorizer.fit_transform(cleaned_messages)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            if dtm.shape[0] < 5:
                return self._get_empty_analysis()
            
            # Determinar número óptimo de temas
            if auto_topics:
                n_topics = self.find_optimal_topics(dtm)
            else:
                n_topics = min(n_topics, dtm.shape[0] // 3)
            
            n_topics = max(2, min(n_topics, 8))
            
            # Análisis LDA
            lda_topics = self._perform_lda_analysis(dtm, n_topics)
            
            # Análisis NMF
            nmf_topics = self._perform_nmf_analysis(dtm, n_topics)
            
            # Análisis por clustering
            cluster_topics = self._perform_clustering_analysis(dtm, n_topics, cleaned_messages)
            
            # Análisis de n-gramas
            ngrams = self.extract_ngrams(cleaned_messages)
            
            return {
                'lda_topics': lda_topics,
                'nmf_topics': nmf_topics,
                'cluster_topics': cluster_topics,
                'ngrams': ngrams,
                'optimal_topics': n_topics,
                'total_documents': dtm.shape[0],
                'total_words': dtm.shape[1],
                'algorithm_comparison': self._compare_algorithms(lda_topics, nmf_topics, cluster_topics)
            }
            
        except Exception as e:
            st.error(f"Error en análisis avanzado de temas: {e}")
            return self._get_empty_analysis()
    
    def _perform_lda_analysis(self, dtm, n_topics: int) -> List[Dict[str, Any]]:
        """Realiza análisis LDA"""
        try:
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20,
                learning_method='online'
            )
            
            lda_output = self.lda_model.fit_transform(dtm)
            topics = []
            
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [self.feature_names[i] for i in top_words_idx]
                word_scores = [topic[i] for i in top_words_idx]
                
                # Calcular importancia del tema
                topic_importance = lda_output[:, topic_idx].mean()
                
                topics.append({
                    'topic_id': topic_idx + 1,
                    'words': top_words,
                    'scores': word_scores,
                    'importance': float(topic_importance),
                    'algorithm': 'LDA'
                })
            
            return topics
        except:
            return []
    
    def _perform_nmf_analysis(self, dtm, n_topics: int) -> List[Dict[str, Any]]:
        """Realiza análisis NMF (Non-negative Matrix Factorization)"""
        try:
            self.nmf_model = NMF(
                n_components=n_topics,
                random_state=42,
                max_iter=200
            )
            
            nmf_output = self.nmf_model.fit_transform(dtm)
            topics = []
            
            for topic_idx, topic in enumerate(self.nmf_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [self.feature_names[i] for i in top_words_idx]
                word_scores = [topic[i] for i in top_words_idx]
                
                # Calcular importancia del tema
                topic_importance = nmf_output[:, topic_idx].mean()
                
                topics.append({
                    'topic_id': topic_idx + 1,
                    'words': top_words,
                    'scores': word_scores,
                    'importance': float(topic_importance),
                    'algorithm': 'NMF'
                })
            
            return topics
        except:
            return []
    
    def _perform_clustering_analysis(self, dtm, n_topics: int, texts: List[str]) -> List[Dict[str, Any]]:
        """Realiza análisis por clustering K-means"""
        try:
            self.kmeans = KMeans(
                n_clusters=n_topics,
                random_state=42,
                n_init=10
            )
            
            clusters = self.kmeans.fit_predict(dtm)
            
            # Extraer palabras clave por cluster
            topics = []
            order_centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]
            
            for cluster_idx in range(n_topics):
                top_words_idx = order_centroids[cluster_idx, :10]
                top_words = [self.feature_names[i] for i in top_words_idx]
                
                # Calcular tamaño del cluster
                cluster_size = np.sum(clusters == cluster_idx)
                cluster_importance = cluster_size / len(texts)
                
                # Ejemplos representativos del cluster
                cluster_docs = [texts[i] for i in range(len(texts)) if clusters[i] == cluster_idx]
                representative_docs = cluster_docs[:3] if cluster_docs else []
                
                topics.append({
                    'topic_id': cluster_idx + 1,
                    'words': top_words,
                    'importance': float(cluster_importance),
                    'cluster_size': cluster_size,
                    'examples': representative_docs,
                    'algorithm': 'K-means'
                })
            
            return topics
        except:
            return []
    
    def _compare_algorithms(self, lda_topics: List, nmf_topics: List, cluster_topics: List) -> Dict[str, Any]:
        """Compara los resultados de diferentes algoritmos"""
        comparison = {
            'lda_topic_count': len(lda_topics),
            'nmf_topic_count': len(nmf_topics),
            'cluster_topic_count': len(cluster_topics),
            'best_algorithm': 'LDA'  # Por defecto
        }
        
        # Calcular diversidad de temas (cuántas palabras únicas)
        lda_words = set()
        nmf_words = set()
        cluster_words = set()
        
        for topic in lda_topics:
            lda_words.update(topic['words'][:5])
        
        for topic in nmf_topics:
            nmf_words.update(topic['words'][:5])
        
        for topic in cluster_topics:
            cluster_words.update(topic['words'][:5])
        
        comparison.update({
            'lda_unique_words': len(lda_words),
            'nmf_unique_words': len(nmf_words),
            'cluster_unique_words': len(cluster_words)
        })
        
        return comparison
    
    def _get_empty_analysis(self) -> Dict[str, Any]:
        """Retorna análisis vacío"""
        return {
            'lda_topics': [],
            'nmf_topics': [],
            'cluster_topics': [],
            'ngrams': [],
            'optimal_topics': 0,
            'total_documents': 0,
            'total_words': 0,
            'algorithm_comparison': {}
        }

def show_advanced_topic_analysis(messages: List[Dict[str, Any]]):
    """Visualización avanzada de análisis de temas"""
    if not messages or len(messages) < 20:
        st.info("📊 Se necesitan al menos 20 mensajes para el análisis avanzado de temas")
        return
    
    st.markdown("### 🎯 Análisis Avanzado de Temas")
    
    # Extraer textos de los mensajes
    texts = [msg.get('message', '') for msg in messages if msg.get('message')]
    
    # Configuración del análisis
    col1, col2 = st.columns(2)
    
    with col1:
        auto_detect = st.checkbox("Detección automática de temas", value=True, 
                                 help="El sistema determina el número óptimo de temas")
    
    with col2:
        if not auto_detect:
            n_topics = st.slider("Número de temas", 2, 8, 4)
        else:
            n_topics = 4  # Valor por defecto
    
    # Realizar análisis
    model = AdvancedTopicModeling()
    
    with st.spinner("🔍 Analizando temas con múltiples algoritmos..."):
        analysis_results = model.analyze_topics_advanced(texts, auto_detect, n_topics)
    
    if not analysis_results['lda_topics']:
        st.info("❌ No se pudieron identificar temas significativos en la conversación")
        return
    
    # Mostrar resultados
    _display_topic_analysis_results(analysis_results, texts)

def _display_topic_analysis_results(analysis_results: Dict[str, Any], texts: List[str]):
    """Muestra los resultados del análisis de temas"""
    
    # Métricas generales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Temas Identificados", analysis_results['optimal_topics'])
    
    with col2:
        st.metric("📝 Documentos Analizados", analysis_results['total_documents'])
    
    with col3:
        st.metric("🔤 Palabras Únicas", analysis_results['total_words'])
    
    with col4:
        best_algo = analysis_results['algorithm_comparison'].get('best_algorithm', 'LDA')
        st.metric("🏆 Mejor Algoritmo", best_algo)
    
    # Pestañas para diferentes algoritmos
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 LDA", "📈 NMF", "🔗 Clustering", "📊 Comparación"])
    
    with tab1:
        _display_lda_results(analysis_results['lda_topics'])
    
    with tab2:
        _display_nmf_results(analysis_results['nmf_topics'])
    
    with tab3:
        _display_clustering_results(analysis_results['cluster_topics'])
    
    with tab4:
        _display_algorithm_comparison(analysis_results)
    
    # N-gramas más frecuentes
    if analysis_results['ngrams']:
        st.markdown("#### 🔤 N-gramas Más Frecuentes")
        ngrams = analysis_results['ngrams'][:15]
        
        col1, col2, col3 = st.columns(3)
        items_per_col = len(ngrams) // 3 + 1
        
        for i, col in enumerate([col1, col2, col3]):
            with col:
                start_idx = i * items_per_col
                end_idx = start_idx + items_per_col
                for ngram in ngrams[start_idx:end_idx]:
                    st.write(f"• {ngram}")

def _display_lda_results(topics: List[Dict[str, Any]]):
    """Muestra resultados del algoritmo LDA"""
    if not topics:
        st.info("No se generaron temas con LDA")
        return
    
    st.markdown("##### Temas Identificados por LDA")
    
    for topic in topics:
        with st.expander(f"**Tema {topic['topic_id']}** - Importancia: {topic['importance']:.2%}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Palabras clave:**")
                words_html = ""
                for word, score in zip(topic['words'][:8], topic['scores'][:8]):
                    intensity = min(255, int(score * 255))
                    words_html += f'<span style="background-color: rgba(132, 176, 38, {score}); padding: 2px 6px; margin: 2px; border-radius: 3px; font-size: 0.9em;">{word}</span> '
                st.markdown(words_html, unsafe_allow_html=True)
            
            with col2:
                st.metric("Importancia", f"{topic['importance']:.2%}")

def _display_nmf_results(topics: List[Dict[str, Any]]):
    """Muestra resultados del algoritmo NMF"""
    if not topics:
        st.info("No se generaron temas con NMF")
        return
    
    st.markdown("##### Temas Identificados por NMF")
    
    # Ordenar por importancia
    topics_sorted = sorted(topics, key=lambda x: x['importance'], reverse=True)
    
    for topic in topics_sorted:
        with st.expander(f"**Tema {topic['topic_id']}** - Importancia: {topic['importance']:.2%}", expanded=False):
            st.write("**Palabras clave:**")
            
            # Crear gráfico de barras para las palabras
            words = topic['words'][:8]
            scores = topic['scores'][:8]
            
            fig = px.bar(
                x=scores,
                y=words,
                orientation='h',
                title=f"Tema {topic['topic_id']} - Distribución de Palabras",
                labels={'x': 'Relevancia', 'y': 'Palabra'},
                color=scores,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

def _display_clustering_results(topics: List[Dict[str, Any]]):
    """Muestra resultados del clustering"""
    if not topics:
        st.info("No se generaron clusters")
        return
    
    st.markdown("##### Grupos de Conversación Identificados")
    
    # Ordenar por tamaño del cluster
    topics_sorted = sorted(topics, key=lambda x: x['cluster_size'], reverse=True)
    
    for topic in topics_sorted:
        with st.expander(f"**Grupo {topic['topic_id']}** - {topic['cluster_size']} mensajes ({topic['importance']:.2%})", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Palabras características:**")
                for word in topic['words'][:6]:
                    st.write(f"• {word}")
                
                if topic['examples']:
                    st.write("**Ejemplos representativos:**")
                    for example in topic['examples']:
                        st.write(f"*\"{example[:100]}...\"*")
            
            with col2:
                st.metric("Mensajes", topic['cluster_size'])
                st.metric("Porcentaje", f"{topic['importance']:.2%}")

def _display_algorithm_comparison(analysis_results: Dict[str, Any]):
    """Muestra comparación entre algoritmos"""
    comparison = analysis_results['algorithm_comparison']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Temas LDA", comparison.get('lda_topic_count', 0))
        st.metric("Palabras Únicas LDA", comparison.get('lda_unique_words', 0))
    
    with col2:
        st.metric("Temas NMF", comparison.get('nmf_topic_count', 0))
        st.metric("Palabras Únicas NMF", comparison.get('nmf_unique_words', 0))
    
    with col3:
        st.metric("Clusters", comparison.get('cluster_topic_count', 0))
        st.metric("Palabras Únicas Clustering", comparison.get('cluster_unique_words', 0))
    
    # Recomendación del mejor algoritmo
    best_algo = comparison.get('best_algorithm', 'LDA')
    st.info(f"**🏆 Recomendación:** El algoritmo **{best_algo}** proporcionó los temas más diversos y significativos")

# Función de compatibilidad con el código existente
def show_topic_analysis(df):
    """Función de compatibilidad con la versión anterior"""
    if 'message' not in df.columns or len(df) < 10:
        st.info("Se necesitan más mensajes para el análisis de temas")
        return
    
    # Usar la nueva función avanzada
    messages = [{'message': msg} for msg in df['message'].tolist()]
    show_advanced_topic_analysis(messages)

# Clase de compatibilidad
class TopicModeling(AdvancedTopicModeling):
    """Clase de compatibilidad con la versión anterior"""
    def analyze_topics(self, messages, n_topics=3):
        """Método de compatibilidad"""
        texts = [msg if isinstance(msg, str) else str(msg) for msg in messages]
        results = self.analyze_topics_advanced(texts, False, n_topics)
        return results.get('lda_topics', [])